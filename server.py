"""
Flask Backend für Tirol DGM Selector
Empfängt BBOX-Parameter und generiert STL/OBJ-Dateien.
"""

from flask import Flask, request, jsonify, send_file, render_template_string
from flask_cors import CORS
import threading
import os
import uuid
import traceback

app = Flask(__name__)
CORS(app)

# Output-Verzeichnis
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Job-Status speichern
jobs = {}

# ── Import der Terrain-Logik ──────────────────────────────────────────────
import xml.etree.ElementTree as ET
import requests
import numpy as np

WCS_ENDPOINT = (
    "https://gis.tirol.gv.at/arcgis/services/Service_Public/terrain/"
    "MapServer/WCSServer"
)

NS = {
    "gml":    "http://www.opengis.net/gml/3.2",
    "gmlcov": "http://www.opengis.net/gmlcov/1.0",
    "wcs":    "http://www.opengis.net/wcs/2.0",
}

Z_SCALE  = 2.0
Z_OFFSET = 50.0


def fetch_multipart(bbox, coverage_name, crs):
    ymin, xmin, ymax, xmax = bbox
    params = [
        ("SERVICE",    "WCS"),
        ("VERSION",    "2.0.1"),
        ("REQUEST",    "GetCoverage"),
        ("COVERAGEID", coverage_name),
        ("SUBSET",     f"y({ymin},{ymax})"),
        ("SUBSET",     f"x({xmin},{xmax})"),
        ("FORMAT",     "image/tiff"),
        ("OUTPUTCRS",  crs),
    ]
    r = requests.get(WCS_ENDPOINT, params=params, timeout=(10, 5 * 60))
    r.raise_for_status()

    content_type = r.headers.get("Content-Type", "")
    boundary = None
    for chunk in content_type.split(";"):
        chunk = chunk.strip()
        if chunk.lower().startswith("boundary="):
            boundary = chunk.split("=", 1)[1].strip('"').strip("'").encode()
            break

    if boundary is None:
        raise RuntimeError(f"No Multipart-Boundary found. Content-Type: {content_type}")

    raw = r.content
    delimiter = b"--" + boundary + b"\n"
    raw_parts = raw.split(delimiter)

    parts = []
    for part in raw_parts:
        if not part or part.strip(b"\n-") == b"":
            continue
        pos = 0
        header_lines = []
        while pos < len(part):
            cr_end = part.find(b"\r\n", pos)
            n_end  = part.find(b"\n",   pos)
            if n_end == -1:
                break
            if cr_end != -1 and cr_end == n_end - 1:
                eol = cr_end
                next_pos = cr_end + 2
            else:
                eol = n_end
                next_pos = n_end + 1
            line = part[pos:eol].decode(errors="ignore").strip()
            if line.startswith("Content-"):
                header_lines.append(line)
                pos = next_pos
            elif line == "" and header_lines:
                pos = next_pos
                break
            elif not header_lines:
                pos = next_pos
            else:
                break
        body = part[pos:]
        if body:
            parts.append(("\n".join(header_lines), body))
    return parts


def parse_gml(gml_bytes):
    root = ET.fromstring(gml_bytes)
    grid_env = root.find(".//gml:GridEnvelope", NS)
    high = grid_env.find("gml:high", NS).text.split()
    rows = int(high[0]) + 1
    cols = int(high[1]) + 1
    pos = root.find(".//gml:origin/gml:Point/gml:pos", NS).text.split()
    origin_y = float(pos[0])
    origin_x = float(pos[1])
    offset_vectors = root.findall(".//gml:offsetVector", NS)
    ov0 = [float(v) for v in offset_vectors[0].text.split()]
    ov1 = [float(v) for v in offset_vectors[1].text.split()]
    dy = abs(ov0[0]) if ov0[0] != 0 else abs(ov0[1])
    dx = abs(ov1[1]) if ov1[1] != 0 else abs(ov1[0])
    return rows, cols, origin_y, origin_x, dy, dx


def raw_to_array(tiff_bytes):
    from rasterio.io import MemoryFile
    with MemoryFile(tiff_bytes) as memfile:
        with memfile.open() as dataset:
            arr = dataset.read(1).astype(np.float32)
            nodata = dataset.nodata
    if nodata is not None:
        arr[nodata == arr] = np.nan
    arr[arr < -1000] = np.nan
    arr[arr > 9000]  = np.nan
    if np.isnan(arr).any():
        flat = arr.ravel()
        mask = np.isnan(flat)
        idx  = np.where(~mask, np.arange(len(flat)), 0)
        np.maximum.accumulate(idx, out=idx)
        arr  = flat[idx].reshape(arr.shape)
    return arr


def downsample_array(arr, dy, dx, factor):
    if factor <= 1:
        return arr, dy, dx
    new_arr = arr[::factor, ::factor]
    return new_arr, dy * factor, dx * factor


def build_mesh(arr, origin_y, origin_x, dy, dx, scale_factor=1.0, closed_body=False):
    rows, cols = arr.shape
    ys = origin_y - np.arange(rows) * dy
    xs = origin_x + np.arange(cols) * dx
    xx, yy = np.meshgrid(xs, ys)
    zz = arr * Z_SCALE
    xx = xx * scale_factor
    yy = yy * scale_factor
    zz = zz * scale_factor
    verts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    r = np.arange(rows - 1)
    c = np.arange(cols - 1)
    rr, cc = np.meshgrid(r, c, indexing="ij")
    rr, cc = rr.ravel(), cc.ravel()
    v00 = rr * cols + cc
    v10 = (rr + 1) * cols + cc
    v01 = rr * cols + (cc + 1)
    v11 = (rr + 1) * cols + (cc + 1)
    faces = np.vstack([
        np.column_stack([v00, v10, v01]),
        np.column_stack([v10, v11, v01]),
    ])
    if closed_body:
        min_z = np.min(zz) - Z_OFFSET * scale_factor
        base_verts = verts.copy()
        base_verts[:, 2] = min_z
        offset = len(verts)
        verts = np.vstack([verts, base_verts])
        base_faces = faces.copy() + offset
        base_faces = np.column_stack([base_faces[:, 0], base_faces[:, 2], base_faces[:, 1]])
        side_faces = []
        for col in range(cols - 1):
            v1, v2 = col, col + 1
            v3, v4 = v2 + offset, v1 + offset
            side_faces += [[v1, v2, v3], [v1, v3, v4]]
        for col in range(cols - 1):
            v1 = (rows - 1) * cols + col
            v2 = (rows - 1) * cols + col + 1
            v3, v4 = v2 + offset, v1 + offset
            side_faces += [[v1, v3, v2], [v1, v4, v3]]
        for row in range(rows - 1):
            v1 = row * cols
            v2 = (row + 1) * cols
            v3, v4 = v2 + offset, v1 + offset
            side_faces += [[v1, v3, v2], [v1, v4, v3]]
        for row in range(rows - 1):
            v1 = row * cols + (cols - 1)
            v2 = (row + 1) * cols + (cols - 1)
            v3, v4 = v2 + offset, v1 + offset
            side_faces += [[v1, v2, v3], [v1, v3, v4]]
        faces = np.vstack([faces, base_faces, np.array(side_faces)])
    return verts, faces


def export_stl(verts, faces, path):
    from stl import mesh as stl_mesh
    verts_mm = verts * 1000.0
    m = stl_mesh.Mesh(np.zeros(len(faces), dtype=stl_mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            m.vectors[i][j] = verts_mm[f[j]]
    m.save(path)


def export_obj(verts, faces, path):
    verts_mm = verts * 1000.0
    with open(path, "w") as f:
        f.write("# Datenquelle: Land Tirol\n# Units: millimeters\n")
        for v in verts_mm:
            f.write(f"v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def run_job(job_id, params):
    try:
        jobs[job_id]["status"] = "running"
        jobs[job_id]["log"] = []

        def log(msg):
            jobs[job_id]["log"].append(msg)
            print(f"[{job_id[:8]}] {msg}")

        bbox          = tuple(params["bbox"])
        coverage_name = params["coverage_name"]
        downsample    = int(params.get("downsample", 4))
        closed_body   = bool(params.get("closed_body", False))
        target_mm     = float(params.get("target_mm", 100.0))
        fmt           = params.get("format", "stl")  # stl | obj | both

        log(f"Fetch data: {coverage_name}  BBOX={bbox}")
        parts = fetch_multipart(bbox, coverage_name, "EPSG:31254")

        gml_bytes = raw_bytes = None
        for header, body in parts:
            h = header.lower()
            if "text/xml" in h or "gml" in h:
                gml_bytes = body
            else:
                raw_bytes = body
        if gml_bytes is None or raw_bytes is None:
            gml_bytes, raw_bytes = parts[0][1], parts[1][1]

        log("Parse GML ...")
        rows, cols, origin_y, origin_x, dy, dx = parse_gml(gml_bytes)
        log(f"Grid: {cols}×{rows}  dx={dx}m dy={dy}m")

        log("Convert raster data ...")
        arr = raw_to_array(raw_bytes)

        if downsample > 1:
            log(f"Downsampling ×{downsample} ...")
            arr, dy, dx = downsample_array(arr, dy, dx, downsample)
            rows, cols = arr.shape
            log(f"Reduced grid: {cols}×{rows}")

        actual_width_m  = (cols - 1) * dx
        actual_height_m = (rows - 1) * dy
        max_dim_m       = max(actual_width_m, actual_height_m)
        scale_factor    = target_mm / max_dim_m / 1000.0

        log(f"Area: {actual_width_m:.0f}m × {actual_height_m:.0f}m → {target_mm}mm")
        log("Build mesh ...")
        verts, faces = build_mesh(arr, origin_y, origin_x, dy, dx, scale_factor, closed_body)
        log(f"Mesh: {len(verts):,} vertices, {len(faces):,} triangles")

        out_files = []
        base = os.path.join(OUTPUT_DIR, job_id)

        if fmt in ("stl", "both"):
            path = base + ".stl"
            log("Export STL ...")
            export_stl(verts, faces, path)
            size_mb = os.path.getsize(path) / 1024 / 1024
            log(f"STL: {size_mb:.1f} MB")
            out_files.append({"name": f"{job_id}.stl", "size_mb": round(size_mb, 1)})

        if fmt in ("obj", "both"):
            path = base + ".obj"
            log("Export OBJ ...")
            export_obj(verts, faces, path)
            size_mb = os.path.getsize(path) / 1024 / 1024
            log(f"OBJ: {size_mb:.1f} MB")
            out_files.append({"name": f"{job_id}.obj", "size_mb": round(size_mb, 1)})

        jobs[job_id]["status"] = "done"
        jobs[job_id]["files"]  = out_files
        log("Done!")

    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"]  = str(e)
        jobs[job_id]["log"].append(f"ERROR: {e}")
        traceback.print_exc()


# ── ROUTES ────────────────────────────────────────────────────────────────

@app.route("/generate", methods=["POST"])
def generate():
    params = request.get_json()
    if not params or "bbox" not in params:
        return jsonify({"error": "Missing bbox"}), 400

    job_id = uuid.uuid4().hex
    jobs[job_id] = {"status": "queued", "log": [], "files": []}
    t = threading.Thread(target=run_job, args=(job_id, params), daemon=True)
    t.start()
    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id):
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(jobs[job_id])


@app.route("/download/<filename>")
def download(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        return jsonify({"error": "File not found"}), 404
    return send_file(path, as_attachment=True)



@app.route("/")
def index():
    with open("index.html", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    print("Tirol DGM Server läuft auf http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
