"""Convert a generated .ply Gaussian splat into a clean triangle mesh & Binary FBX 7.4.

Pipeline (all CPU — no GPU required):
  1. Load the Gaussian PLY (recovered to Z-up world: xyz = (x, z, -y)_ply)
  2. Filter splats by opacity threshold + optional statistical outlier removal
  3. Surface reconstruction (pyvista marching cubes or open3d Poisson)
  4. Mesh stabilization: remove degenerate/duplicate faces, drop tiny
     disconnected floater clusters, and apply optional Taubin smoothing
  5. Optional quadric edge-collapse decimation to target triangle count
  6. Export Binary FBX 7.4 (Blender headless + round-trip verification when
     available, else pure-Python Binary FBX 7.4 writer) and save:
       {stem}_mesh.ply   — FBX-verified binary PLY (x,y,z + rgb + faces)
       {stem}.fbx        — Binary FBX 7.4 for Blender / Unity / Unreal
       {stem}_mesh.json  — Persistent mesh/FBX metadata (faces, verts, settings)

CLI:
  .venv/bin/python convert_fbx.py --ply outputs/X.ply --out outputs/X \
      [--target_faces 30000] [--opacity 0.3] [--quality standard] \
      [--method marching_cubes] [--no_clean] [--no_smooth] [--no_fbx]
"""
import argparse
import json
import os
import struct
import time

import numpy as np
import plyfile

# 3D Gaussian SH DC coefficient -> linear color (same as render_preview.py)
_SH0 = 0.28209479177387814

QUALITY_MAX_POINTS = {
    "fast": 20000,
    "standard": 35000,
    "high": 55000,
}

QUALITY_GRID_RES = {
    "fast": 128,
    "standard": 160,
    "high": 208,
}


def load_gaussians(ply_path):
    """Load Gaussian splats as (xyz, colors, opacity) in Z-up world space."""
    ply = plyfile.PlyData.read(ply_path)
    el = ply["vertex"]
    # PLY is written rotated +90 deg about X: (x, y, z)_zup = (x, z, -y)_ply
    xyz = np.stack([el["x"], el["z"], -el["y"]], axis=1).astype(np.float64)
    sh = np.stack([el["f_dc_0"], el["f_dc_1"], el["f_dc_2"]], axis=1).astype(np.float64)
    opacity = 1.0 / (1.0 + np.exp(-np.array(el["opacity"], dtype=np.float64).reshape(-1)))
    colors = np.clip(0.5 + _SH0 * sh, 0.0, 1.0)
    return xyz, colors, opacity


def load_gaussian_ellipsoids(ply_path):
    """Load full Gaussian splats (xyz, colors, opacity, scale, rot_zup) in Z-up world space."""
    ply = plyfile.PlyData.read(ply_path)
    el = ply["vertex"]
    xyz = np.stack([el["x"], el["z"], -el["y"]], axis=1).astype(np.float64)
    sh = np.stack([el["f_dc_0"], el["f_dc_1"], el["f_dc_2"]], axis=1).astype(np.float64)
    opacity = 1.0 / (1.0 + np.exp(-np.array(el["opacity"], dtype=np.float64).reshape(-1)))
    colors = np.clip(0.5 + _SH0 * sh, 0.0, 1.0)
    scale = np.exp(np.stack([el["scale_0"], el["scale_1"], el["scale_2"]], axis=1).astype(np.float32))
    raw_rot = np.stack([el["rot_0"], el["rot_1"], el["rot_2"], el["rot_3"]], axis=1).astype(np.float32)
    c = 0.5 ** 0.5
    sw, sx, sy, sz = raw_rot[:, 0], raw_rot[:, 1], raw_rot[:, 2], raw_rot[:, 3]
    rot = np.stack([c * (sw + sx), c * (sx - sw), c * (sy + sz), c * (sz - sy)], axis=1)
    rot /= np.maximum(np.linalg.norm(rot, axis=1, keepdims=True), 1e-8)
    return xyz, colors, opacity, scale, rot


def _remove_point_outliers(xyz, colors, nb_neighbors=20, std_ratio=2.2, extras=None):
    """Drop isolated floating splats before surface reconstruction."""
    if len(xyz) < 500:
        return (xyz, colors, extras) if extras is not None else (xyz, colors)
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    _, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    if len(ind) < max(500, int(len(xyz) * 0.25)):
        return (xyz, colors, extras) if extras is not None else (xyz, colors)
    mask = np.asarray(ind, dtype=np.int64)
    if extras is not None:
        return xyz[mask], colors[mask], [e[mask] for e in extras]
    return xyz[mask], colors[mask]


def _voxel_downsample(xyz, colors, max_points=35000):
    """Hash-grid voxel downsampling with kNN-estimated voxel size."""
    n = len(xyz)
    if n <= max_points:
        return xyz, colors, n

    from scipy.spatial import cKDTree

    m = min(20000, n)
    sub = xyz[np.random.default_rng(0).choice(n, size=m, replace=False)]
    nn = cKDTree(sub).query(sub, k=2, workers=-1)[0][:, 1]
    base = float(np.median(nn))
    if not np.isfinite(base) or base <= 1e-6:
        base = float(np.max(xyz.max(axis=0) - xyz.min(axis=0))) / 200.0

    vsize = base
    inverse = None
    nv = 0
    for _ in range(8):
        vmin = xyz.min(axis=0)
        cell = np.floor((xyz - vmin) / vsize).astype(np.int64)
        c0 = cell[:, 0] - cell.min(axis=0)[0]
        c1 = cell[:, 1] - cell.min(axis=0)[1]
        c2 = cell[:, 2] - cell.min(axis=0)[2]
        keys = (c0 * 100003) ^ (c1 * 1009) ^ c2
        _, inverse = np.unique(keys, return_inverse=True)
        nv = int(inverse.max()) + 1
        if nv <= max_points * 1.6:
            break
        vsize *= (nv / max_points) ** (1.0 / 3.0) * 1.1

    cnt = np.bincount(inverse, minlength=nv)
    pos = np.zeros((nv, 3), dtype=np.float64)
    col = np.zeros((nv, 3), dtype=np.float64)
    np.add.at(pos, inverse, xyz)
    np.add.at(col, inverse, colors)
    valid = cnt > 0
    pos[valid] /= cnt[valid, None]
    col[valid] /= cnt[valid, None]
    return pos[valid], col[valid], int(valid.sum())


def _parse_pyvista_faces(surf):
    """Extract (M, 3) triangle indices from a PyVista PolyData surface."""
    raw = np.asarray(surf.faces, dtype=np.int64).reshape(-1)
    if raw.size and raw.size % 4 == 0 and np.all(raw[0::4] == 3):
        return raw.reshape(-1, 4)[:, 1:].astype(np.int64)
    tris = []
    i = 0
    while i < raw.size:
        cnt = int(raw[i])
        if cnt <= 0 or i + 1 + cnt > raw.size:
            break
        idx = raw[i + 1 : i + 1 + cnt]
        if cnt == 3:
            tris.append(idx)
        elif cnt > 3:
            for k in range(1, cnt - 1):
                tris.append(np.array([idx[0], idx[k], idx[k + 1]]))
        i += 1 + cnt
    return np.array(tris, dtype=np.int64).reshape(-1, 3) if tris else np.zeros((0, 3), dtype=np.int64)


def _clean_and_smooth_mesh(verts, faces, colors, clean=True, smooth=True, fill_holes=True):
    """Stabilize mesh topology: remove degenerates/floaters, seal holes, and optionally smooth."""
    if len(verts) == 0 or len(faces) == 0:
        return verts, faces, colors
    import open3d as o3d

    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(verts)
    m.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    m.vertex_colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))

    m.remove_degenerate_triangles()
    m.remove_duplicated_triangles()
    m.remove_duplicated_vertices()
    m.remove_non_manifold_edges()

    if clean and len(m.triangles) > 200:
        tri_clusters, cluster_n_tris, _ = m.cluster_connected_triangles()
        tri_clusters = np.asarray(tri_clusters)
        cluster_n_tris = np.asarray(cluster_n_tris)
        if len(cluster_n_tris) > 1:
            max_cnt = int(cluster_n_tris.max())
            # Keep main body and substantial pieces (>= 5% of main component)
            min_keep = max(250, int(max_cnt * 0.05))
            drop_mask = cluster_n_tris[tri_clusters] < min_keep
            if not np.all(drop_mask):
                m.remove_triangles_by_mask(drop_mask)

    m.remove_unreferenced_vertices()

    if fill_holes and len(m.triangles) <= 60000 and not m.is_watertight():
        v_tmp, f_tmp = _repair_mesh_holes(np.asarray(m.vertices), np.asarray(m.triangles))
        if len(v_tmp) > 0 and len(f_tmp) > 0:
            m.vertices = o3d.utility.Vector3dVector(v_tmp)
            m.triangles = o3d.utility.Vector3iVector(f_tmp.astype(np.int32))
            if len(colors) > 0:
                from scipy.spatial import cKDTree
                _, idx = cKDTree(verts).query(v_tmp)
                m.vertex_colors = o3d.utility.Vector3dVector(np.clip(colors[idx], 0.0, 1.0))

    if smooth and len(m.triangles) > 0:
        m = m.filter_smooth_taubin(number_of_iterations=2)

    out_v = np.asarray(m.vertices, dtype=np.float64)
    out_f = np.asarray(m.triangles, dtype=np.int64)
    out_c = (
        np.asarray(m.vertex_colors, dtype=np.float64)
        if m.has_vertex_colors() and len(m.vertex_colors) == len(out_v)
        else np.full((len(out_v), 3), 0.75, dtype=np.float64)
    )
    return out_v, out_f, np.clip(out_c, 0.0, 1.0)


def _repair_mesh_holes(verts, faces):
    """Seal open boundary loops and ensure outward winding using trimesh (best-effort)."""
    try:
        import trimesh

        tm = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
        if not tm.is_watertight:
            trimesh.repair.fill_holes(tm)
            trimesh.repair.fix_winding(tm)
        if tm.is_watertight and tm.volume < 0:
            tm.invert()
        return np.asarray(tm.vertices, dtype=np.float64), np.asarray(tm.faces, dtype=np.int64)
    except Exception:
        return verts, faces


def _ensure_outward_winding(verts, faces):
    """Ensure closed mesh triangles are wound outward (positive signed volume)."""
    if len(verts) == 0 or len(faces) == 0:
        return verts, faces
    try:
        import trimesh

        tm = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
        trimesh.repair.fix_winding(tm)
        if tm.volume < 0:
            tm.invert()
        return np.asarray(tm.vertices, dtype=np.float64), np.asarray(tm.faces, dtype=np.int64)
    except Exception:
        return verts, faces


def reconstruct(xyz, colors, sample_spacing=None, max_points=35000,
                grid_res=160, method="marching_cubes", clean=True, smooth=True,
                fill_holes=True, scale=None, opacity=None):
    """Reconstruct a triangle surface from kept splats. Returns (verts, faces, colors)."""
    from scipy.spatial import cKDTree

    if clean:
        extras = [arr for arr in (scale, opacity) if arr is not None]
        if extras:
            xyz, colors, kept_ex = _remove_point_outliers(xyz, colors, extras=extras)
            it = iter(kept_ex)
            if scale is not None:
                scale = next(it)
            if opacity is not None:
                opacity = next(it)
        else:
            xyz, colors = _remove_point_outliers(xyz, colors)

    if method in ("marching_cubes", "poisson"):
        import pyvista as pv
        from scipy.ndimage import (
            binary_closing,
            binary_dilation,
            binary_erosion,
            binary_fill_holes,
            distance_transform_edt,
            gaussian_filter,
        )

        lo, hi = xyz.min(axis=0), xyz.max(axis=0)
        span = float(np.max(hi - lo)) or 1.0
        pad = span * 0.09
        origin = lo - pad
        spacing = float(sample_spacing) if (sample_spacing and sample_spacing > 0) else (span + 2.0 * pad) / (grid_res - 1)
        dims = np.ceil((hi - lo + 2.0 * pad) / spacing).astype(int) + 1

        if scale is not None and opacity is not None:
            # Continuous scale- & opacity-weighted 3D Gaussian density field
            r_vox = np.clip(np.mean(scale, axis=1) / spacing * 1.5, 0.6, 3.0)
            gx = (xyz[:, 0] - origin[0]) / spacing
            gy = (xyz[:, 1] - origin[1]) / spacing
            gz = (xyz[:, 2] - origin[2]) / spacing
            ix = np.floor(gx).astype(np.int32)
            iy = np.floor(gy).astype(np.int32)
            iz = np.floor(gz).astype(np.int32)
            fx, fy, fz = gx - ix, gy - iy, gz - iz
            valid = (
                (ix >= 0) & (ix < dims[0] - 1)
                & (iy >= 0) & (iy < dims[1] - 1)
                & (iz >= 0) & (iz < dims[2] - 1)
            )
            ix, iy, iz = ix[valid], iy[valid], iz[valid]
            fx, fy, fz = fx[valid].astype(np.float32), fy[valid].astype(np.float32), fz[valid].astype(np.float32)
            w_pts = (opacity[valid] * np.clip(r_vox[valid], 0.5, 2.5)).astype(np.float32)

            vol = np.zeros(dims, dtype=np.float32)
            for dx, wx in ((0, 1.0 - fx), (1, fx)):
                for dy, wy in ((0, 1.0 - fy), (1, fy)):
                    for dz, wz in ((0, 1.0 - fz), (1, fz)):
                        np.add.at(vol, (ix + dx, iy + dy, iz + dz), w_pts * wx * wy * wz)

            med_r = float(np.median(r_vox))
            base_sig = max(0.50, min(0.90, med_r * 0.65))
            sigma = base_sig * (1.40 if method == "poisson" else 1.0) if smooth else base_sig * 0.75
            vol_s = gaussian_filter(vol, sigma=sigma)
            nz = vol_s[vol_s > 0.01]
            thr = float(np.percentile(nz, 22)) if nz.size else 0.1

            if fill_holes:
                # Recessed solid core fill seals hollow interiors WITHOUT bloating thin spikes/claws
                occ_tight = vol_s >= thr
                r_dil = max(2, int(grid_res) // 48)
                st = np.ones((3, 3, 3), dtype=bool)
                occ_core = binary_erosion(
                    binary_fill_holes(binary_dilation(occ_tight, structure=st, iterations=r_dil)),
                    structure=st,
                    iterations=r_dil + 1,
                )
                vol_s[occ_core] = np.maximum(vol_s[occ_core], thr * 1.6)

            grid = pv.ImageData(dimensions=dims, spacing=(spacing, spacing, spacing), origin=origin)
            grid.point_data["d"] = vol_s.ravel(order="F")
            surf = grid.contour([thr], scalars="d").triangulate()
        else:
            idx = np.clip(np.round((xyz - origin) / spacing).astype(int), 0, dims - 1)
            occ = np.zeros(dims, dtype=bool)
            occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
            if fill_holes:
                r = max(2, int(grid_res) // 48)
                st = np.ones((3, 3, 3), dtype=bool)
                occ = binary_erosion(
                    binary_fill_holes(binary_dilation(occ, structure=st, iterations=r)),
                    structure=st,
                    iterations=r,
                )
            else:
                occ = binary_closing(occ, iterations=1)
            dist = distance_transform_edt(~occ).astype(np.float32) * spacing
            if smooth:
                dist = gaussian_filter(dist, sigma=0.85 if method == "poisson" else 0.55)
            grid = pv.ImageData(dimensions=dims, spacing=(spacing, spacing, spacing), origin=origin)
            grid.point_data["d"] = dist.ravel(order="F")
            surf = grid.contour([spacing * 0.72], scalars="d").triangulate()

        verts = np.asarray(surf.points, dtype=np.float64)
        faces = _parse_pyvista_faces(surf)
        verts, faces = _ensure_outward_winding(verts, faces)
        color_src_xyz, color_src_col = xyz, colors
    else:
        import pyvista as pv

        ds_xyz, ds_col, _ = _voxel_downsample(xyz, colors, max_points=max_points)
        pts = pv.PolyData(ds_xyz)
        kw = {}
        if sample_spacing is not None and sample_spacing > 0:
            kw["sample_spacing"] = sample_spacing
        surf = pts.reconstruct_surface(**kw)
        if surf is None or surf.n_points == 0:
            surf = pts.reconstruct_surface(sample_spacing=0.02)
        verts = np.asarray(surf.points, dtype=np.float64)
        faces = _parse_pyvista_faces(surf)
        color_src_xyz, color_src_col = ds_xyz, ds_col

    if len(verts) == 0 or len(faces) == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int64), np.zeros((0, 3), dtype=np.float64)

    tree = cKDTree(color_src_xyz)
    _, idx = tree.query(verts, workers=-1)
    out_colors = color_src_col[idx]

    verts, faces, out_colors = _clean_and_smooth_mesh(
        verts, faces, out_colors, clean=clean, smooth=smooth, fill_holes=fill_holes
    )
    return verts, faces.astype(np.int64), out_colors


def _transfer_splat_colors(verts, src_xyz, src_colors, src_opacity=None, k=6):
    """Transfer 3DGS colors onto mesh vertices using opacity- & distance-weighted kNN."""
    if len(verts) == 0 or len(src_xyz) == 0:
        return np.zeros((len(verts), 3), dtype=np.float64)
    from scipy.spatial import cKDTree

    k_eff = min(int(k), len(src_xyz))
    tree = cKDTree(src_xyz)
    dists, idx = tree.query(verts, k=k_eff, workers=-1)
    if k_eff == 1:
        return np.clip(src_colors[idx], 0.0, 1.0)

    # Soft distance scale avoids single interior splats dominating a vertex color
    d0 = np.median(dists[:, 0])
    sigma2 = max(float(d0) * 1.5, 1e-4) ** 2
    w = 1.0 / (dists * dists + sigma2)
    if src_opacity is not None:
        w *= np.square(np.clip(src_opacity[idx], 0.05, 1.0))
    w_sum = np.sum(w, axis=1, keepdims=True)
    w /= np.maximum(w_sum, 1e-12)
    blended = np.sum(src_colors[idx] * w[:, :, None], axis=1)
    return np.clip(blended, 0.0, 1.0)


def decimate(verts, faces, colors, target):
    """Decimate mesh to `target` triangles (open3d quadric). No-op if target<=0 or already below."""
    if target is None or target <= 0 or len(faces) <= target:
        return verts, faces, colors, len(faces)
    import open3d as o3d

    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(verts)
    m.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    m.vertex_colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))

    m2 = m.simplify_quadric_decimation(int(target))
    if m2 is None or len(m2.triangles) == 0 or len(m2.triangles) >= len(m.triangles):
        m2 = m
    else:
        m2.remove_degenerate_triangles()
        m2.remove_duplicated_triangles()
        m2.remove_unreferenced_vertices()

    verts = np.asarray(m2.vertices, dtype=np.float64)
    faces = np.asarray(m2.triangles, dtype=np.int64)
    colors = (
        np.asarray(m2.vertex_colors, dtype=np.float64)
        if m2.has_vertex_colors() and len(m2.vertex_colors) == len(verts)
        else np.full((len(verts), 3), 0.75, dtype=np.float64)
    )
    return verts, faces, np.clip(colors, 0.0, 1.0), len(faces)


def save_mesh_ply(path, verts, colors, faces=None):
    """Save a standard binary PLY (x,y,z float32 + rgb uint8 + list vertex_indices)."""
    v = np.asarray(verts, dtype=np.float32).reshape(-1, 3)
    c = np.clip(np.asarray(colors, dtype=np.float64).reshape(-1, 3), 0.0, 1.0) * 255.0
    rgb = np.round(c).astype(np.uint8)
    f = np.asarray(faces, dtype=np.int32).reshape(-1, 3) if faces is not None and len(faces) else np.zeros((0, 3), dtype=np.int32)

    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(v)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        f"element face {len(f)}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    ).encode("ascii")

    v_dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    v_buf = np.empty(len(v), dtype=v_dtype)
    v_buf["x"], v_buf["y"], v_buf["z"] = v[:, 0], v[:, 1], v[:, 2]
    v_buf["red"], v_buf["green"], v_buf["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    f_dtype = np.dtype([("cnt", "u1"), ("i0", "<i4"), ("i1", "<i4"), ("i2", "<i4")])
    f_buf = np.empty(len(f), dtype=f_dtype)
    if len(f):
        f_buf["cnt"] = 3
        f_buf["i0"], f_buf["i1"], f_buf["i2"] = f[:, 0], f[:, 1], f[:, 2]

    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(v_buf.tobytes())
        if len(f):
            fh.write(f_buf.tobytes())


def _resolve_max_points(quality, max_points):
    if max_points and max_points > 0:
        return int(max_points)
    return QUALITY_MAX_POINTS.get(quality, QUALITY_MAX_POINTS["standard"])


def _compute_vertex_normals(verts, faces):
    """Compute area-weighted unit vertex normals in Z-up world space."""
    v = np.asarray(verts, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    n = np.zeros_like(v)
    if len(f) > 0:
        fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
        for col in range(3):
            np.add.at(n[:, col], f[:, 0], fn[:, col])
            np.add.at(n[:, col], f[:, 1], fn[:, col])
            np.add.at(n[:, col], f[:, 2], fn[:, col])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ok = ln[:, 0] > 1e-12
    n[ok] /= ln[ok]
    n[~ok] = [0.0, 0.0, 1.0]
    return n


def _quats_to_inv_rot_mats(q):
    """Convert (N, 4) unit quaternions [w,x,y,z] to inverse rotation matrices R^T."""
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3), dtype=np.float32)
    R[:, 0, 0] = 1.0 - 2.0 * (qy * qy + qz * qz)
    R[:, 0, 1] = 2.0 * (qx * qy - qz * qw)
    R[:, 0, 2] = 2.0 * (qx * qz + qy * qw)
    R[:, 1, 0] = 2.0 * (qx * qy + qz * qw)
    R[:, 1, 1] = 1.0 - 2.0 * (qx * qx + qz * qz)
    R[:, 1, 2] = 2.0 * (qy * qz - qx * qw)
    R[:, 2, 0] = 2.0 * (qx * qz - qy * qw)
    R[:, 2, 1] = 2.0 * (qy * qz + qx * qw)
    R[:, 2, 2] = 1.0 - 2.0 * (qx * qx + qy * qy)
    return R.transpose(0, 2, 1)


def _raycast_splat_colors(pts, normals, src_xyz, src_colors, src_opacity, src_scale, src_rot, k=16):
    """Analytic 3DGS ellipsoid ray-compositing along outward surface normals.

    Evaluates front-to-back alpha transmittance through each sample's K nearest
    anisotropic Gaussian ellipsoids so outer surface splats naturally occlude
    any dark interior splats beneath the shell.
    """
    if len(pts) == 0 or len(src_xyz) == 0:
        return np.zeros((len(pts), 3), dtype=np.float64)
    from scipy.spatial import cKDTree

    X32 = np.asarray(src_xyz, dtype=np.float32)
    C32 = np.asarray(src_colors, dtype=np.float32)
    Op32 = np.asarray(src_opacity, dtype=np.float32)
    invS = 1.0 / np.maximum(np.asarray(src_scale, dtype=np.float32), 1e-4)
    RT = _quats_to_inv_rot_mats(np.asarray(src_rot, dtype=np.float32))

    P32 = np.asarray(pts, dtype=np.float32)
    N32 = np.asarray(normals, dtype=np.float32)
    n_len = np.linalg.norm(N32, axis=1, keepdims=True)
    N32 = np.where(n_len > 1e-6, N32 / np.maximum(n_len, 1e-6), np.array([0.0, 0.0, 1.0], dtype=np.float32))

    k_eff = min(int(k), len(X32))
    tree = cKDTree(X32)
    # Query slightly outward along normal so front-surface splats are prioritized
    _, idx = tree.query(P32 + 0.002 * N32, k=k_eff, workers=-1)
    if k_eff == 1:
        return np.clip(C32[idx], 0.0, 1.0).astype(np.float64)

    out = np.empty((len(P32), 3), dtype=np.float32)
    chunk = 50000
    for s0 in range(0, len(P32), chunk):
        s1 = min(len(P32), s0 + chunk)
        ii = idx[s0:s1]
        p = P32[s0:s1, None, :]
        n = N32[s0:s1, None, :]
        mu = X32[ii]
        rt = RT[ii]
        iv = invS[ii]
        diff = p - mu
        t_hit = np.einsum("bki,bki->bk", mu - p, n)

        p_loc = np.einsum("bkij,bkj->bki", rt, diff) * iv
        n_loc = np.einsum("bkij,bkj->bki", rt, n) * iv
        nn = np.einsum("bki,bki->bk", n_loc, n_loc) + 1e-8
        t_star = -np.einsum("bki,bki->bk", p_loc, n_loc) / nn
        closest = p_loc + t_star[:, :, None] * n_loc
        maha2 = np.einsum("bki,bki->bk", closest, closest)

        alpha = Op32[ii] * np.exp(-0.5 * np.minimum(maha2, 30.0))
        alpha *= (t_hit > -0.015)

        order = np.argsort(-t_hit, axis=1)
        a_s = np.take_along_axis(alpha, order, axis=1)
        c_s = np.take_along_axis(C32[ii], order[:, :, None], axis=1)

        T = np.ones_like(a_s)
        T[:, 1:] = np.cumprod(1.0 - a_s[:, :-1], axis=1)
        w = T * a_s
        w_sum = np.sum(w, axis=1, keepdims=True)
        comp = np.sum(w[:, :, None] * c_s, axis=1) / np.maximum(w_sum, 1e-6)

        miss = w_sum[:, 0] < 0.05
        if np.any(miss):
            comp[miss] = C32[ii[miss, 0]]
        out[s0:s1] = comp
    return np.clip(out, 0.0, 1.0).astype(np.float64)


def bake_uv_texture(tri_uv_bin, splat_xyz, splat_colors, splat_opacity,
                    out_tex_path, tex_size=1024, splat_scale=None, splat_rot=None):
    """Rasterize unwrapped UV triangles and bake crisp 3DGS colors into a PNG texture."""
    from PIL import Image
    from scipy.ndimage import distance_transform_edt

    raw = np.fromfile(tri_uv_bin, dtype="<f4")
    if raw.size == 0:
        return False
    # Supports both 8-float [x,y,z,nx,ny,nz,u,v] and legacy 5-float [x,y,z,u,v]
    stride = 8 if raw.size % 24 == 0 else 5
    arr = raw.reshape(-1, 3, stride)
    tri_pos = arr[:, :, :3]
    tri_nrm = arr[:, :, 3:6] if stride == 8 else None
    tri_uv = arr[:, :, 6:8] if stride == 8 else arr[:, :, 3:5]

    W = H = int(tex_size)
    px = tri_uv[:, :, 0] * (W - 1)
    py = (1.0 - tri_uv[:, :, 1]) * (H - 1)

    x0 = np.clip(np.floor(px.min(axis=1)).astype(np.int32), 0, W - 1)
    x1 = np.clip(np.ceil(px.max(axis=1)).astype(np.int32), 0, W - 1)
    y0 = np.clip(np.floor(py.min(axis=1)).astype(np.int32), 0, H - 1)
    y1 = np.clip(np.ceil(py.max(axis=1)).astype(np.int32), 0, H - 1)

    denom = (py[:, 1] - py[:, 2]) * (px[:, 0] - px[:, 2]) + (px[:, 2] - px[:, 1]) * (py[:, 0] - py[:, 2])
    valid_tri = np.abs(denom) > 1e-8

    pos_map = np.zeros((H, W, 3), dtype=np.float32)
    nrm_map = np.zeros((H, W, 3), dtype=np.float32) if tri_nrm is not None else None
    mask = np.zeros((H, W), dtype=bool)

    for i in np.nonzero(valid_tri)[0]:
        xa, xb, ya, yb = x0[i], x1[i], y0[i], y1[i]
        xs = np.arange(xa, xb + 1, dtype=np.float32)
        ys = np.arange(ya, yb + 1, dtype=np.float32)
        gx, gy = np.meshgrid(xs, ys)
        inv_d = 1.0 / denom[i]
        w0 = ((py[i, 1] - py[i, 2]) * (gx - px[i, 2]) + (px[i, 2] - px[i, 1]) * (gy - py[i, 2])) * inv_d
        w1 = ((py[i, 2] - py[i, 0]) * (gx - px[i, 2]) + (px[i, 0] - px[i, 2]) * (gy - py[i, 2])) * inv_d
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -0.02) & (w1 >= -0.02) & (w2 >= -0.02)
        if np.any(inside):
            yy, xx = np.nonzero(inside)
            yy += ya
            xx += xa
            cw0 = np.clip(w0[inside], 0.0, 1.0)
            cw1 = np.clip(w1[inside], 0.0, 1.0)
            cw2 = np.clip(w2[inside], 0.0, 1.0)
            cs = np.maximum(cw0 + cw1 + cw2, 1e-6)
            cw0 /= cs; cw1 /= cs; cw2 /= cs
            pos_map[yy, xx] = (
                cw0[:, None] * tri_pos[i, 0][None, :]
                + cw1[:, None] * tri_pos[i, 1][None, :]
                + cw2[:, None] * tri_pos[i, 2][None, :]
            )
            if nrm_map is not None:
                nrm_map[yy, xx] = (
                    cw0[:, None] * tri_nrm[i, 0][None, :]
                    + cw1[:, None] * tri_nrm[i, 1][None, :]
                    + cw2[:, None] * tri_nrm[i, 2][None, :]
                )
            mask[yy, xx] = True
        else:
            cx = int(np.clip(round(float(px[i].mean())), 0, W - 1))
            cy = int(np.clip(round(float(py[i].mean())), 0, H - 1))
            if not mask[cy, cx]:
                pos_map[cy, cx] = tri_pos[i].mean(axis=0)
                if nrm_map is not None:
                    nrm_map[cy, cx] = tri_nrm[i].mean(axis=0)
                mask[cy, cx] = True

    if not np.any(mask):
        return False

    if nrm_map is not None and splat_scale is not None and splat_rot is not None:
        tex_cols = _raycast_splat_colors(
            pos_map[mask], nrm_map[mask],
            splat_xyz, splat_colors, splat_opacity, splat_scale, splat_rot, k=16,
        )
    else:
        tex_cols = _transfer_splat_colors(pos_map[mask], splat_xyz, splat_colors, src_opacity=splat_opacity, k=6)

    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[mask] = np.round(np.clip(tex_cols, 0.0, 1.0) * 255.0).astype(np.uint8)

    _, nearest = distance_transform_edt(~mask, return_indices=True)
    img = img[nearest[0], nearest[1]]
    Image.fromarray(img).save(out_tex_path)
    return True


def convert(ply_path, out_stem, target_faces=30000, opacity_thr=0.3,
            sample_spacing=None, quality="standard", max_points=None,
            method="marching_cubes", clean=True, smooth=True, fill_holes=True,
            tex_size=1024, do_fbx=True, rig_path=None):
    """Run the full PLY->mesh->Binary FBX pipeline and persist _mesh.json metadata."""
    t0 = time.time()
    xyz, colors, opacity, scale, rot = load_gaussian_ellipsoids(ply_path)
    keep = opacity > opacity_thr
    if keep.sum() < 1000:
        keep = opacity >= np.quantile(opacity, 0.8)
    X, C, W, S = xyz[keep], colors[keep], opacity[keep], scale[keep]

    # Color baking keeps semi-transparent surface splats (e.g. fur/hair tips) down to 0.12
    c_keep = opacity > min(0.12, float(opacity_thr))
    if c_keep.sum() < keep.sum():
        c_keep = keep
    Xc, Cc, Wc, Sc, Qc = xyz[c_keep], colors[c_keep], opacity[c_keep], scale[c_keep], rot[c_keep]

    mp = _resolve_max_points(quality, max_points)
    gres = QUALITY_GRID_RES.get(quality, QUALITY_GRID_RES["standard"])
    t1 = time.time()
    verts, faces, vcolors = reconstruct(
        X, C,
        sample_spacing=sample_spacing,
        max_points=mp,
        grid_res=gres,
        method=method,
        clean=clean,
        smooth=smooth,
        fill_holes=fill_holes,
        scale=S,
        opacity=W,
    )
    base_faces = len(faces)
    t_recon = time.time() - t1

    verts, faces, vcolors, final_faces = decimate(verts, faces, vcolors, target_faces)
    if fill_holes and len(verts) > 0 and len(faces) > 0:
        verts, faces = _repair_mesh_holes(verts, faces)
        final_faces = len(faces)
    # Re-project colors via analytic 3DGS ellipsoid ray-compositing onto final mesh vertices
    if len(verts) > 0:
        vn = _compute_vertex_normals(verts, faces)
        vcolors = _raycast_splat_colors(verts, vn, Xc, Cc, Wc, Sc, Qc, k=16)

    mesh_ply = out_stem + "_mesh.ply"
    save_mesh_ply(mesh_ply, verts, vcolors, faces)

    bounds = None
    if len(verts):
        bounds = {
            "min": [round(float(x), 5) for x in verts.min(axis=0)],
            "max": [round(float(x), 5) for x in verts.max(axis=0)],
        }

    info = {
        "ply": ply_path,
        "mesh_ply": mesh_ply,
        "opacity": float(opacity_thr),
        "quality": quality,
        "method": method,
        "clean": bool(clean),
        "smooth": bool(smooth),
        "fill_holes": bool(fill_holes),
        "target_faces": int(target_faces or 0),
        "kept_points": int(keep.sum()),
        "base_faces": int(base_faces),
        "faces": int(final_faces),
        "verts": int(len(verts)),
        "bounds": bounds,
        "recon_seconds": round(t_recon, 1),
    }

    if do_fbx:
        import tempfile
        from .fbx_writer import find_blender, unwrap_uv_blender, write_fbx

        fbx_path = out_stem + ".fbx"
        tex_path = out_stem + "_texture.png"
        uv_path = out_stem + "_mesh_uv.bin"
        mesh_name = os.path.splitext(os.path.basename(out_stem))[0]

        has_tex = False
        if tex_size and tex_size > 0 and find_blender():
            with tempfile.TemporaryDirectory() as tmpd:
                blend_tmp = os.path.join(tmpd, "uv.blend")
                tri_uv_tmp = os.path.join(tmpd, "tri_uv.bin")
                if unwrap_uv_blender(mesh_ply, blend_tmp, tri_uv_tmp):
                    has_tex = bake_uv_texture(
                        tri_uv_tmp, Xc, Cc, Wc, tex_path,
                        tex_size=tex_size, splat_scale=Sc, splat_rot=Qc,
                    )
                res = write_fbx(
                    fbx_path,
                    verts.astype(np.float32),
                    faces.astype(np.int32),
                    vertex_colors=vcolors.astype(np.float32),
                    name=mesh_name,
                    mesh_ply_path=mesh_ply,
                    verified_ply_path=mesh_ply,
                    blend_path=blend_tmp if has_tex else None,
                    texture_path=tex_path if has_tex else None,
                    verified_uv_path=uv_path if has_tex else None,
                )
        else:
            res = write_fbx(
                fbx_path,
                verts.astype(np.float32),
                faces.astype(np.int32),
                vertex_colors=vcolors.astype(np.float32),
                name=mesh_name,
                mesh_ply_path=mesh_ply,
                verified_ply_path=mesh_ply,
            )

        info["fbx"] = fbx_path
        info["fbx_bytes"] = int(res.get("bytes", os.path.getsize(fbx_path)))
        info["verts"] = int(res.get("verts", info["verts"]))
        info["faces"] = int(res.get("faces", info["faces"]))
        info["verified_by"] = res.get("verified_by", "python-binary")
        info["has_texture"] = bool(res.get("has_texture", has_tex))
        if has_tex and os.path.exists(tex_path):
            info["texture"] = tex_path
            info["tex_size"] = int(tex_size)
        if has_tex and os.path.exists(uv_path):
            info["mesh_uv"] = uv_path
        if res.get("bounds"):
            info["bounds"] = res["bounds"]

    if rig_path:
        from .rig_bind import bind_rig_to_mesh
        sk_glb = out_stem + "_skinned.glb"
        sk_fbx = out_stem + "_skinned.fbx"
        mesh_name = os.path.splitext(os.path.basename(out_stem))[0]
        meta_yaw = None
        src_meta = os.path.splitext(ply_path)[0] + ".json"
        if os.path.exists(src_meta):
            try:
                with open(src_meta, "r", encoding="utf-8") as mf:
                    cam = json.load(mf).get("camera") or {}
                    if "yaw" in cam:
                        meta_yaw = float(cam["yaw"])
            except Exception:
                pass
        rig_info = bind_rig_to_mesh(
            rig_path, verts, faces, vcolors,
            bounds["min"], bounds["max"],
            out_glb=sk_glb, out_fbx=sk_fbx,
            mesh_name=mesh_name,
            meta_yaw=meta_yaw,
        )
        info["rig"] = rig_info

    info["seconds"] = round(time.time() - t0, 1)
    info["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    meta_path = out_stem + "_mesh.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(info, fh, ensure_ascii=False, indent=2)
    info["mesh_meta"] = meta_path
    return info


def probe(ply_path, opacity_thr=0.3, sample_spacing=None, quality="standard",
          max_points=None, method="marching_cubes", clean=True, smooth=True,
          fill_holes=True):
    """Reconstruction only (no decimation, no file writes) to report base face count."""
    t0 = time.time()
    xyz, colors, opacity, scale, _rot = load_gaussian_ellipsoids(ply_path)
    keep = opacity > opacity_thr
    if keep.sum() < 1000:
        keep = opacity >= np.quantile(opacity, 0.8)
    X, C, W, S = xyz[keep], colors[keep], opacity[keep], scale[keep]
    mp = _resolve_max_points(quality, max_points)
    gres = QUALITY_GRID_RES.get(quality, QUALITY_GRID_RES["standard"])
    verts, faces, _vcolors = reconstruct(
        X, C,
        sample_spacing=sample_spacing,
        max_points=mp,
        grid_res=gres,
        method=method,
        clean=clean,
        smooth=smooth,
        fill_holes=fill_holes,
        scale=S,
        opacity=W,
    )
    return {
        "kept_points": int(keep.sum()),
        "base_faces": int(len(faces)),
        "base_verts": int(len(verts)),
        "quality": quality,
        "method": method,
        "clean": bool(clean),
        "smooth": bool(smooth),
        "fill_holes": bool(fill_holes),
        "seconds": round(time.time() - t0, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True, help="input Gaussian splat PLY")
    ap.add_argument("--out", default=None, help="output stem (no extension)")
    ap.add_argument("--target_faces", type=int, default=30000,
                    help="target triangle count after decimation (0 = keep all)")
    ap.add_argument("--opacity", type=float, default=0.3,
                    help="minimum splat opacity kept for reconstruction")
    ap.add_argument("--quality", choices=list(QUALITY_MAX_POINTS.keys()), default="standard",
                    help="reconstruction point budget preset")
    ap.add_argument("--max_points", type=int, default=None,
                    help="explicit voxel-downsample point budget override")
    ap.add_argument("--method", choices=["marching_cubes", "poisson", "surface_fit"], default="marching_cubes",
                    help="surface reconstruction algorithm")
    ap.add_argument("--no_clean", action="store_true",
                    help="disable outlier & floater component cleanup")
    ap.add_argument("--no_smooth", action="store_true",
                    help="disable Taubin surface smoothing")
    ap.add_argument("--no_fill_holes", action="store_true",
                    help="disable morphological & mesh hole filling")
    ap.add_argument("--tex_size", type=int, default=1024,
                    help="UV baked texture resolution (0 = vertex colors only)")
    ap.add_argument("--sample_spacing", type=float, default=None,
                    help="reconstruction grid spacing (None = auto)")
    ap.add_argument("--no_fbx", action="store_true", help="skip FBX export")
    ap.add_argument("--rig", default=None,
                    help="rig GLB path (e.g. rigs/sweetie_rig.glb); binds the mesh "
                         "and writes skinned GLB + skinned FBX alongside the outputs")
    ap.add_argument("--probe", action="store_true",
                    help="only reconstruct to report base face count; write no files")
    args = ap.parse_args()

    clean = not args.no_clean
    smooth = not args.no_smooth
    fill_holes = not args.no_fill_holes

    if args.probe:
        print(json.dumps(probe(
            args.ply,
            opacity_thr=args.opacity,
            sample_spacing=args.sample_spacing,
            quality=args.quality,
            max_points=args.max_points,
            method=args.method,
            clean=clean,
            smooth=smooth,
            fill_holes=fill_holes,
        )), flush=True)
        return

    if not args.out:
        ap.error("--out is required unless --probe is used")

    target = args.target_faces if args.target_faces > 0 else 0
    info = convert(
        args.ply,
        args.out,
        target_faces=target,
        opacity_thr=args.opacity,
        sample_spacing=args.sample_spacing,
        quality=args.quality,
        max_points=args.max_points,
        method=args.method,
        clean=clean,
        smooth=smooth,
        fill_holes=fill_holes,
        tex_size=args.tex_size,
        do_fbx=not args.no_fbx,
        rig_path=args.rig,
    )
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
