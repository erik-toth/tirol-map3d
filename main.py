"""
Tirol Map 3D
--------------------------
Dieses Skript extrahiert digitale Geländemodelle (DGM) über die WCS-Schnittstelle 
des Landes Tirol und konvertiert diese STL- oder OBJ-Dateien.

Rechtlicher Hinweis:
Die Geodaten werden vom Land Tirol bereitgestellt. Bei jeglicher Art der 
Darstellung oder Erstellung von Folgeprodukten ist auf das Urheberrecht des 
Landes Tirol hinzuweisen (z. B. "Datenquelle: Land Tirol").
Die Verwendung erfolgt gemäß den Standardbedingungen des Landes Tirol (Dez. 2017).
(https://www.tirol.gv.at/data/informationsweiterverwendung/)

Autor: Erik Tóth
Lizenz: MIT
"""

import os
import struct
import xml.etree.ElementTree as ET
import requests
import numpy as np

# config

WCS_ENDPOINT = (
    "https://gis.tirol.gv.at/arcgis/services/Service_Public/terrain/"
    "MapServer/WCSServer"
)

# layer model:
#   'Gelaendemodell_5m_M28'     -> 5 m resolution
#   'Gelaendemodell_50cm_M28'   -> 50 cm resolution (files get very large!!!)
COVERAGE_NAME = "Gelaendemodell_5m_M28"

# Bounding Box in EPSG:31254 system
# BBOX = (y_min, x_min, y_max, x_max)
# covered area by the layers: y 167802–293567 / x -18752–202157
# How to get coords: https://epsg.io/map , projection to EPSG:31254
# x-mid:80256 y-mid:236783 for Old Town, Innsbruck (AT)
BBOX = (235783, 79256, 237783, 81256) # ~ 2x2 km area of Innsbruck

# CRS (M28 -> EPSG:31254)
CRS = "EPSG:31254"

# Z-axis scaling (1.0 -> 1:1 scale, 2–5 better 3D look (useful for 3d printing))
Z_SCALE = 2.0

# output file names
OUTPUT_STL = "terrain.stl"
OUTPUT_OBJ  = "terrain.obj"


# namespace GML-Parsing
NS = {
    "gml":    "http://www.opengis.net/gml/3.2",
    "gmlcov": "http://www.opengis.net/gmlcov/1.0",
    "wcs":    "http://www.opengis.net/wcs/2.0",
}


# fetch multipart and split

def fetch_multipart(bbox, coverage_name, crs):
    """
    Sens GetCoverage-Request and returns all multipart-parts as a
    list of (header_str, body_bytes) tuples.
    """
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

    print(f"Load data from '{coverage_name}' ...")
    print(f"  BBox (y_min, x_min, y_max, x_max): {bbox}")

    r = requests.get(WCS_ENDPOINT, params=params, timeout=(10, 5 * 60))
    r.raise_for_status()

    content_type = r.headers.get("Content-Type", "")
    print(f"  Content-Type: {content_type}")

    # get boundary - strip quotes: boundary="wcs" → wcs
    boundary = None
    for chunk in content_type.split(";"):
        chunk = chunk.strip()
        if chunk.lower().startswith("boundary="):
            boundary = chunk.split("=", 1)[1].strip('"').strip("'").encode()
            break

    if boundary is None:
        raise RuntimeError(f"No Multipart-Boundary fund. Content-Type: {content_type}")

    # work with raw byte values!!!! 
    # do not normalize!
    # Example format response from server:
    #   --wcs\n
    #   Content-Type: text/xml\n
    #   Content-ID: GML-Part\n
    #   <?xml...         ← no space
    #   ...
    #   --wcs\n
    #   Content-Type: image/tiff\n
    #   Content-ID: ...\n
    #   II*\x00...      ← raw TIFF-Bytes
    #   --wcs--
    raw = r.content
    delimiter = b"--" + boundary + b"\n"
    raw_parts = raw.split(delimiter)

    parts = []
    for part in raw_parts:
        if not part or part.strip(b"\n-") == b"":
            continue

        # read header lines (end with \n)
        # body starts with the first line that doesnt start with "Content-"
        pos = 0
        header_lines = []
        while pos < len(part):
            # line ending (\r\n oder \n)
            cr_end = part.find(b"\r\n", pos)
            n_end  = part.find(b"\n",   pos)
            if n_end == -1:
                break
            # use \r\n if available
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
                # space after header
                pos = next_pos
                break
            elif not header_lines:
                pos = next_pos  # skip leading space
            else:
                break  # no space

        body = part[pos:]  # raw bytes

        if body:
            parts.append(("\n".join(header_lines), body))

    print(f"  {len(parts)} Part(s) received")
    return parts


# GML parse to grid infos

def parse_gml(gml_bytes: bytes):
    """
    Reads data from GML-Part
      - cols, rows          (number of pixels)
      - origin_y, origin_x  (origin point in meters)
      - dy, dx              (pixel size, form offsetVector)
    """
    root = ET.fromstring(gml_bytes)

    # gml:GridEnvelope -> high yields (rows-1, cols-1)
    grid_env = root.find(".//gml:GridEnvelope", NS)
    high = grid_env.find("gml:high", NS).text.split()
    # axisLabels="y x" -> high[0]=rows-1, high[1]=cols-1
    rows = int(high[0]) + 1
    cols = int(high[1]) + 1

    # gml:origin
    pos = root.find(".//gml:origin/gml:Point/gml:pos", NS).text.split()
    origin_y = float(pos[0])
    origin_x = float(pos[1])

    # gml:offsetVector - two vector
    offset_vectors = root.findall(".//gml:offsetVector", NS)
    # first vector: 1 dy-direction, 2 dx-Richtung
    ov0 = [float(v) for v in offset_vectors[0].text.split()]
    ov1 = [float(v) for v in offset_vectors[1].text.split()]
    dy = abs(ov0[0]) if ov0[0] != 0 else abs(ov0[1])
    dx = abs(ov1[1]) if ov1[1] != 0 else abs(ov1[0])

    print(f"  Grid: {cols} × {rows} Pixel  |  dx={dx} m, dy={dy} m")
    print(f"  Origin: y={origin_y}, x={origin_x}")

    return rows, cols, origin_y, origin_x, dy, dx


# TIFF-bytes to numpy array

def raw_to_array(tiff_bytes: bytes, rows: int = None, cols: int = None):
    """
    Reads a TIFF (including compressed) directly from memory via rasterio.
    rows/cols are read from the TIFF itself (GML values for reference only).
    """
    from rasterio.io import MemoryFile

    with MemoryFile(tiff_bytes) as memfile:
        with memfile.open() as dataset:
            arr = dataset.read(1).astype(np.float32)
            nodata = dataset.nodata
            print(f"  Raster size: {arr.shape[1]} × {arr.shape[0]} Pixel")

    # Nodata maskieren
    if nodata is not None:
        arr[nodata == arr] = np.nan
    arr[arr < -1000] = np.nan
    arr[arr > 9000]  = np.nan

    # NaN-Lücken füllen (Vorwärts-Fill)
    if np.isnan(arr).any():
        flat = arr.ravel()
        mask = np.isnan(flat)
        idx  = np.where(~mask, np.arange(len(flat)), 0)
        np.maximum.accumulate(idx, out=idx)
        arr  = flat[idx].reshape(arr.shape)

    valid = arr[~np.isnan(arr)]
    print(f"  Altitude range: {valid.min():.1f} – {valid.max():.1f} m")
    return arr


# ── 4. Mesh bauen ────────────────────────────────────────────────────────────

def build_mesh(arr, origin_y, origin_x, dy, dx):
    """Vertices + triangles from hight-array."""
    rows, cols = arr.shape

    # build grid
    # origin is upper left corner (biggest y, smallest x)
    ys = origin_y - np.arange(rows) * dy
    xs = origin_x + np.arange(cols) * dx

    xx, yy = np.meshgrid(xs, ys)
    zz = arr * Z_SCALE

    verts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    # two tris pro quad
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

    print(f"  Mesh: {len(verts):,} Vertices, {len(faces):,} Triangles")
    return verts, faces


# STL export

def export_stl(verts, faces, path):
    from stl import mesh as stl_mesh
    m = stl_mesh.Mesh(np.zeros(len(faces), dtype=stl_mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            m.vectors[i][j] = verts[f[j]]
    m.save(path)
    print(f"STL saved: {path}  ({os.path.getsize(path)/1024/1024:.1f} MB)")


# OBJ export

def export_obj(verts, faces, path):
    with open(path, "w") as f:
        f.write("#Datenquelle: Land Tirol\n")
        for v in verts:
            f.write(f"v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    print(f"OBJ saved: {path}  ({os.path.getsize(path)/1024/1024:.1f} MB)")


# main

if __name__ == "__main__":

    # fetch data
    parts = fetch_multipart(BBOX, COVERAGE_NAME, CRS)

    # identify parts
    gml_bytes = None
    raw_bytes  = None
    for header, body in parts:
        h = header.lower()
        if "text/xml" in h or "gml" in h:
            gml_bytes = body
        else:
            raw_bytes = body

    if gml_bytes is None or raw_bytes is None:
        # Fallback
        gml_bytes = parts[0][1]
        raw_bytes  = parts[1][1]

    # Debugging:
    # with open("debug.gml", "wb") as f: f.write(gml_bytes)
    # with open("debug.bin", "wb") as f: f.write(raw_bytes)

    # parse GML
    print("\nParse GML …")
    rows, cols, origin_y, origin_x, dy, dx = parse_gml(gml_bytes)

    # 3. Raw to array
    print("\nConverting raw data ...")
    arr = raw_to_array(raw_bytes)

    # 4. build mesh
    print("\nBuildign mesh ...")
    verts, faces = build_mesh(arr, origin_y, origin_x, dy, dx)

    # 5. export
    while True:
        match(int(input("Export mesh to...\n   [0] STL\n   [1] OBJ\n   [2] STL and OBJ\nEnter a number: "))):
            case 0:
                print("\nThis may take a while. Exporting...")
                export_stl(verts, faces, OUTPUT_STL)
                break
            case 1:
                print("\nThis may take a while. Exporting...")
                export_obj(verts, faces, OUTPUT_OBJ)
                break
            case 2:
                print("\nThis may take a while. Exporting...")
                export_stl(verts, faces, OUTPUT_STL)
                export_obj(verts, faces, OUTPUT_OBJ)
            case _: break
    

    print("\nDone!")
