"""
Tirol Map 3D
--------------------------
Dieses Skript extrahiert digitale Geländemodelle (DGM) über die WCS-Schnittstelle 
des Landes Tirol und konvertiert diese in STL- oder OBJ-Dateien.

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
import xml.etree.ElementTree as ET
import requests
import numpy as np
import math
import shutil

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
# covered area by the layers: y 167802–293567 / x -18752–202086
# How to get coords: https://epsg.io/map , projection to EPSG:31254
# x-mid:80256 y-mid:236783 for Old Town, Innsbruck (AT)
# kauns 26647.097431 215983.895577
# 327675
BBOX = (167802.5, -18752.5, 293567.5, 202157.5) # (167802.5, -18752.5, 293567.5, 202157.5)

# tiling settings
TILE_SIZE = 25_000 # length of one tile
TILING_THRESHOLD = 25_000 # tiling threshold

# CRS (M28 -> EPSG:31254)
CRS = "EPSG:31254"

# Z-axis scaling (1.0 -> 1:1 scale, 2–5 better 3D look (useful for 3d printing))
Z_SCALE = 2.0

Z_OFFSET = 50.0

# output file names
OUTPUT_STL = "terrain.stl"
OUTPUT_OBJ  = "terrain.obj"

# cache file path
CACHE_DIR = ".tile_cache"


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
    #   <?xml...         <- no space
    #   ...
    #   --wcs\n
    #   Content-Type: image/tiff\n
    #   Content-ID: ...\n
    #   II*\x00...      <- raw TIFF-Bytes
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
            print(f"  Raster size: {arr.shape[1]} x {arr.shape[0]} Pixel")

    # mask nodata
    if nodata is not None:
        arr[nodata == arr] = np.nan
    arr[arr < -1000] = np.nan
    arr[arr > 9000]  = np.nan

    # fill NaN
    if np.isnan(arr).any():
        flat = arr.ravel()
        mask = np.isnan(flat)
        idx  = np.where(~mask, np.arange(len(flat)), 0)
        np.maximum.accumulate(idx, out=idx)
        arr  = flat[idx].reshape(arr.shape)

    valid = arr[~np.isnan(arr)]
    # print only if data is present
    if len(valid) > 0:
        print(f"  Altitude range: {valid.min()-Z_OFFSET:.1f} - {valid.max()-Z_OFFSET:.1f} m")
    return arr


# downsample array

def downsample_array(arr, dy, dx, factor):
    """
    Reduces array resolution by taking every n-th point.
    Returns downsampled array and adjusted pixel sizes.
    """
    if factor <= 1:
        return arr, dy, dx
    
    rows, cols = arr.shape
    new_arr = arr[::factor, ::factor]
    new_dy = dy * factor
    new_dx = dx * factor
    
    new_rows, new_cols = new_arr.shape
    reduction = (rows * cols) / (new_rows * new_cols)
    
    print(f"  Downsampling by factor {factor}")
    print(f"  Original: {cols} x {rows} pixels")
    print(f"  Reduced:  {new_cols} x {new_rows} pixels")
    print(f"  Reduction: {reduction:.1f}x fewer points")
    print(f"  New resolution: dx={new_dx:.1f} m, dy={new_dy:.1f} m")
    
    return new_arr, new_dy, new_dx


# mesh simplification (curvature-aware vertex clustering)

def simplify_mesh(verts: np.ndarray, faces: np.ndarray, target_ratio: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """
    Generic mesh simplification using Quadric Error Metric (QEM) approximation
    via vertex clustering on a uniform grid.
    """

    if target_ratio >= 1.0:
        return verts, faces

    n_verts = len(verts)
    target_verts = max(4, int(n_verts * target_ratio))

    # grid resolution
    bbox_min = verts.min(axis=0)
    bbox_max = verts.max(axis=0)
    bbox_range = bbox_max - bbox_min
    # avoid division by zero for flat axis
    bbox_range = np.where(bbox_range == 0, 1e-9, bbox_range)

    # cells per axis
    cells_per_axis = max(2, int(round(target_verts ** (1.0 / 3.0))))
    cell_size = bbox_range / cells_per_axis

    # assign each vertex to a cell
    norm_verts = (verts - bbox_min) / bbox_range
    cell_idx_f = norm_verts * (cells_per_axis - 1e-9)
    cell_idx   = cell_idx_f.astype(np.int32)
    cell_idx   = np.clip(cell_idx, 0, cells_per_axis - 1)

    stride = np.array([cells_per_axis * cells_per_axis, cells_per_axis, 1], dtype=np.int64)
    flat_cell = (cell_idx * stride).sum(axis=1)

    # quadric per cell and representative vertex
    unique_cells, inv_map = np.unique(flat_cell, return_inverse=True)
    n_cells = len(unique_cells)
    cell_map = {c: i for i, c in enumerate(unique_cells)}

    # accumulate triangle quadrics
    # Q per cell stored as (A 3x3, b 3, c scalar) for the error form:
    #   E(v) = v^T A v - 2 b^T v + c
    A_acc = np.zeros((n_cells, 3, 3), dtype=np.float64)
    b_acc = np.zeros((n_cells, 3),    dtype=np.float64)
    c_acc = np.zeros(n_cells,         dtype=np.float64)

    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    edges1 = (v1 - v0).astype(np.float64)
    edges2 = (v2 - v0).astype(np.float64)
    normals = np.cross(edges1, edges2)
    norms   = np.linalg.norm(normals, axis=1, keepdims=True)
    norms   = np.where(norms == 0, 1e-15, norms)
    normals = normals / norms

    d = -(normals * v0.astype(np.float64)).sum(axis=1)

    # for each face corner add the quadric to its cell
    for corner in range(3):
        vi_idx = faces[:, corner]
        ci     = inv_map[vi_idx]
        n      = normals
        # A += n n^T
        np.add.at(A_acc, ci, n[:, :, None] * n[:, None, :])
        # b += d * n
        np.add.at(b_acc, ci, (d[:, None] * n))
        # c += d^2
        np.add.at(c_acc, ci, d ** 2)
    # note: b should be -A·p for centroid; using plane-based b here is correct
    # the representative minimises E(v) = v^T A v + 2 b^T v + c

    # solve for representative vertex per cell
    new_verts_list = []
    for i in range(n_cells):
        A = A_acc[i]
        b = -b_acc[i]   # A v = -b_acc  (from ∂E/∂v = 2Av + 2b_acc = 0)
        try:
            cond = np.linalg.cond(A)
            if cond < 1e10:
                v_opt = np.linalg.solve(A, b)
            else:
                raise np.linalg.LinAlgError("ill-conditioned")
        except np.linalg.LinAlgError:
            # fallback
            mask  = inv_map == i
            v_opt = verts[mask].mean(axis=0).astype(np.float64)
        new_verts_list.append(v_opt)

    new_verts = np.array(new_verts_list, dtype=np.float32)

    # remap faces and remove degenerate triangles
    new_face_idx = inv_map[faces]

    valid = (new_face_idx[:, 0] != new_face_idx[:, 1]) & \
            (new_face_idx[:, 1] != new_face_idx[:, 2]) & \
            (new_face_idx[:, 0] != new_face_idx[:, 2])
    new_faces = new_face_idx[valid].astype(np.int32)

    print(f"  QEM simplification: {n_verts:,} → {len(new_verts):,} vertices  "
          f"({len(faces):,} → {len(new_faces):,} triangles)  "
          f"[target ratio {target_ratio:.0%}]")

    return new_verts, new_faces


# build mesh

def build_mesh(arr, origin_y, origin_x, dy, dx, scale_factor=1.0, closed_body=False):
    """Vertices + triangles from hight-array."""
    rows, cols = arr.shape

    # build grid
    # origin is upper left corner (biggest y, smallest x)
    ys = origin_y - np.arange(rows) * dy
    xs = origin_x + np.arange(cols) * dx

    xx, yy = np.meshgrid(xs, ys)
    zz = arr * Z_SCALE

    # apply scale
    xx = xx * scale_factor
    yy = yy * scale_factor
    zz = zz * scale_factor

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

    if closed_body:
        print("  Creating closed volume...")
        
        # min z height
        min_z = np.min(zz) - Z_OFFSET * scale_factor  # 1m
        
        # create floor verts
        base_verts = verts.copy()
        base_verts[:, 2] = min_z
        
        # add floor vertices
        offset = len(verts)
        verts = np.vstack([verts, base_verts])
        
        # bottom face (mirrored)
        base_faces = faces.copy() + offset
        base_faces = np.column_stack([base_faces[:, 0], base_faces[:, 2], base_faces[:, 1]])
        
        # side face
        side_faces = []
        
        # edge (row 0)
        for col in range(cols - 1):
            v1 = col
            v2 = col + 1
            v3 = v2 + offset
            v4 = v1 + offset
            side_faces.append([v1, v2, v3])
            side_faces.append([v1, v3, v4])
        
        # edge (row rows-1)
        for col in range(cols - 1):
            v1 = (rows - 1) * cols + col
            v2 = (rows - 1) * cols + col + 1
            v3 = v2 + offset
            v4 = v1 + offset
            side_faces.append([v1, v3, v2])
            side_faces.append([v1, v4, v3])
        
        #  edge (col 0)
        for row in range(rows - 1):
            v1 = row * cols
            v2 = (row + 1) * cols
            v3 = v2 + offset
            v4 = v1 + offset
            side_faces.append([v1, v3, v2])
            side_faces.append([v1, v4, v3])
        
        # edge (col cols-1)
        for row in range(rows - 1):
            v1 = row * cols + (cols - 1)
            v2 = (row + 1) * cols + (cols - 1)
            v3 = v2 + offset
            v4 = v1 + offset
            side_faces.append([v1, v2, v3])
            side_faces.append([v1, v3, v4])
        
        side_faces = np.array(side_faces)
        
        # combine all faces
        faces = np.vstack([faces, base_faces, side_faces])

    print(f"  Mesh: {len(verts):,} Vertices, {len(faces):,} Triangles")
    return verts, faces


# STL export

def export_stl(verts, faces, path):
    from stl import mesh as stl_mesh
    verts_mm = verts * 1000.0
    m = stl_mesh.Mesh(np.zeros(len(faces), dtype=stl_mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            m.vectors[i][j] = verts_mm[f[j]]
    m.save(path)
    print(f"STL saved: {path}  ({os.path.getsize(path)/1024/1024:.1f} MB)")


# OBJ export

def export_obj(verts, faces, path):
    verts_mm = verts * 1000.0
    with open(path, "w") as f:
        f.write("# Datenquelle: Land Tirol\n")
        f.write("# Units: millimeters\n")
        for v in verts_mm:
            f.write(f"v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    print(f"OBJ saved: {path}  ({os.path.getsize(path)/1024/1024:.1f} MB)")


# main

if __name__ == "__main__":

    # check if tiling is nessecery
    ymin, xmin, ymax, xmax = BBOX
    delta_y = abs(ymax - ymin)
    delta_x = abs(xmax - xmin)

    if delta_y > TILING_THRESHOLD or delta_x > TILING_THRESHOLD:
        # tiling mode
        print(f"\nTiling is active: Area ({delta_x/1000:.1f}km x {delta_y/1000:.1f}km) is over {TILING_THRESHOLD/1000}km.")
        
        cols_count = math.ceil(delta_x / TILE_SIZE)
        rows_count = math.ceil(delta_y / TILE_SIZE)
        print(f"Spliting in {cols_count * rows_count} tiles ({cols_count}x{rows_count}).")

        # load first tile before
        test_bbox = (ymin, xmin, min(ymin + 100, ymax), min(xmin + 100, xmax))
        parts = fetch_multipart(test_bbox, COVERAGE_NAME, CRS)
        _, _, _, _, dy, dx = parse_gml(parts[0][1])

        # init the resulting array
        total_rows_px = int(round(delta_y / dy))
        total_cols_px = int(round(delta_x / dx))
        arr = np.full((total_rows_px, total_cols_px), np.nan, dtype=np.float32)
        
        # origin point
        origin_y, origin_x = ymax, xmin

        # cache folder
        os.makedirs(CACHE_DIR, exist_ok=True)
        print(f"\nCache folder: '{CACHE_DIR}'")

        # loade tiles and add
        for r in range(rows_count):
            for c in range(cols_count):
                t_ymin = ymin + r * TILE_SIZE
                t_xmin = xmin + c * TILE_SIZE
                t_ymax = min(t_ymin + TILE_SIZE, ymax)
                t_xmax = min(t_xmin + TILE_SIZE, xmax)
                
                print(f"\nLoad tile in row {r+1}, column {c+1}...")
                tile_parts = fetch_multipart((t_ymin, t_xmin, t_ymax, t_xmax), COVERAGE_NAME, CRS)
                
                # identify parts
                t_gml = None
                t_raw = None
                for header, body in tile_parts:
                    h = header.lower()
                    if "text/xml" in h or "gml" in h: t_gml = body
                    else: t_raw = body
                
                if t_raw:
                    tile_arr = raw_to_array(t_raw)
                    th, tw = tile_arr.shape
                    
                    # save tile in cache instead of memory
                    cache_path = os.path.join(CACHE_DIR, f"tile_{r}_{c}.npy")
                    np.save(cache_path, tile_arr)
                    print(f"  Tile saved: '{cache_path}'")
                    
                    # free tile from memory
                    del tile_arr
                    
                    # load tile from cache
                    tile_arr = np.load(cache_path)
                    
                    # calc the resulting index
                    # y-axis is invertet --> ymax
                    start_row = max(0, total_rows_px - int(round((t_ymax - ymin) / dy)))
                    start_col = int(round((t_xmin - xmin) / dx))
                    
                    # paste with saftey check
                    end_row = min(start_row + th, total_rows_px)
                    end_col = min(start_col + tw, total_cols_px)
                    arr[start_row:end_row, start_col:end_col] = tile_arr[:(end_row-start_row), :(end_col-start_col)]
                    
                    # free tile from memory
                    del tile_arr

        # delete cache dir
        shutil.rmtree(CACHE_DIR)
        print(f"\nCache '{CACHE_DIR}' is removed.")

    else:
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
            # fallback
            gml_bytes = parts[0][1]
            raw_bytes  = parts[1][1]

        # parse GML
        print("\nParse GML ...")
        rows, cols, origin_y, origin_x, dy, dx = parse_gml(gml_bytes)

        # raw to array
        print("\nConverting raw data ...")
        arr = raw_to_array(raw_bytes)

    # downsampling
    print("\nMesh simplification:")
    rows, cols = arr.shape
    estimated_verts = rows * cols
    estimated_faces = (rows - 1) * (cols - 1) * 2
    
    print(f"  Current grid: {cols} x {rows} pixels")
    print(f"  Estimated mesh: {estimated_verts:,} vertices, {estimated_faces:,} triangles")
    
    while True:
        try:
            downsample_factor = int(input("\nDownsampling factor (No scaling = 1; recommended = 4): "))
            if downsample_factor < 1:
                print("Factor must be at least 1.")
                continue
            
            if downsample_factor > 1:
                test_rows = rows // downsample_factor
                test_cols = cols // downsample_factor
                new_verts = test_rows * test_cols
                new_faces = (test_rows - 1) * (test_cols - 1) * 2
                print(f"\n  With factor {downsample_factor}:")
                print(f"  New grid: {test_cols} x {test_rows} pixels")
                print(f"  New mesh: {new_verts:,} vertices, {new_faces:,} triangles")
                print(f"  Reduction: {estimated_verts/new_verts:.1f}x fewer vertices")
            
            confirm = input("\nConfirm (y/n): ").strip().lower()
            if confirm == "y":
                if downsample_factor > 1:
                    print("\nApplying downsampling...")
                    arr, dy, dx = downsample_array(arr, dy, dx, downsample_factor)
                    rows, cols = arr.shape
                break
        except ValueError:
            print("Enter a valid integer.")

    # getting true values in meters
    actual_width_m = (cols - 1) * dx
    actual_height_m = (rows - 1) * dy
    max_dimension_m = max(actual_width_m, actual_height_m)
    
    print(f"\nArea: {actual_width_m:.1f} m x {actual_height_m:.1f} m")
    print(f"Max extend: {max_dimension_m:.1f} m")

    # closed volume
    while True:
        closed_input = input("\nCreate closed volume? (y/n): ").strip().lower()
        if closed_input in ["n", "y"]:
            closed_body = closed_input in ["y"]
            break
        print("\nEnter 'y' for yes or 'n' for no!")

    # scaling
    while True:
        try:
            target_size_mm = float(input("\nSide length of square in mm (float): "))
            if target_size_mm <= 0:
                print("Only positive values are allowed.")
                continue
            
            # scale factor (mm to m)
            scale_factor = target_size_mm / max_dimension_m / 1000.0
            
            # new height width
            result_width = actual_width_m * scale_factor * 1000.0
            result_height = actual_height_m * scale_factor * 1000.0
            
            print(f"\nOutput size: {result_width:.1f} mm x {result_height:.1f} mm")
            print(f"Scale: 1:{int(1/scale_factor)}")
            
            confirm = input("Confirm (y/n): ").strip().lower()
            if confirm in ["y"]:
                break
        except ValueError:
            print("Enter a valid number.")

    # mesh simplification
    simplify_ratio = 1.0
    while True:
        simp_input = input(
            "\nMesh simplification after build? "
            "(reduces file size via QEM; 0.0-1.0, 1.0 = no simplification): "
        ).strip()
        try:
            simplify_ratio = float(simp_input)
            if 0.0 < simplify_ratio <= 1.0:
                if simplify_ratio < 1.0:
                    estimated_new_verts = int(rows * cols * simplify_ratio)
                    print(f"  Target: ~{estimated_new_verts:,} vertices "
                          f"(from {rows * cols:,})")
                break
            print("Please enter a value between 0.0 (exclusive) and 1.0 (inclusive).")
        except ValueError:
            print("Enter a valid decimal number, e.g. 0.3")

    # build mesh
    print("\nBuilding mesh ...")
    verts, faces = build_mesh(arr, origin_y, origin_x, dy, dx, scale_factor, closed_body)

    # apply simplification
    if simplify_ratio < 1.0:
        print("\nSimplifying mesh ...")
        verts, faces = simplify_mesh(verts, faces, target_ratio=simplify_ratio)

    # export
    while True:
        try:
            choice = int(input("\nExport mesh to...\n   [0] STL\n   [1] OBJ\n   [2] STL and OBJ\nEnter a number: "))
            match(choice):
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
                    break
                case _: 
                    print("Invalid input.")
        except ValueError:
            print("Please enter a number.")
    

    print("\nDone!")