"""Render a generated .ply Gaussian splat or triangle mesh with an auto-framed orbiting camera.

CPU-only (numpy + plyfile + Pillow, plus pyvista for --mesh).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile

import numpy as np
from PIL import Image
from plyfile import PlyData


def quaternion_to_rotation(qw, qx, qy, qz):
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz) or 1.0
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - (qx * qx + qy * qy)],
    ])


def quat_batch_to_rotations(q):
    n = q.shape[0]
    out = np.empty((n, 3, 3), dtype=np.float32)
    for i in range(n):
        out[i] = quaternion_to_rotation(*q[i])
    return out


def render_frame(xyz, colors, scale, rotQ, opacity, R, f, cx, cy, eye, W, H, bg):
    """Render the (subsampled) splats from camera `eye` looking along `R` rows."""
    cam_xyz = xyz @ R.T
    cam_xyz += -eye @ R.T
    z = cam_xyz[:, 2]
    keep = z > 0.02
    cam_xyz, z = cam_xyz[keep], z[keep]
    if len(cam_xyz) == 0:
        return None
    scale_k = scale[keep]
    opacity_k = opacity[keep]
    colors_k = colors[keep]

    u = cam_xyz[:, 0] / z * f + cx
    v = -cam_xyz[:, 1] / z * f + cy

    Q = quat_batch_to_rotations(rotQ[keep])
    M = np.einsum("ab,nbc->nac", R, Q) * scale_k[:, None, :]
    S = f / z
    R32 = M[:, :2, :]
    cov2d = (R32 @ R32.transpose(0, 2, 1)) * (S[:, None, None]) ** 2
    cov2d += 0.3 * np.eye(2)

    det = cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] * cov2d[:, 1, 0]
    ok = det > 1e-12
    cov2d, u, v, opacity_k, colors_k, z = (
        cov2d[ok], u[ok], v[ok], opacity_k[ok], colors_k[ok], z[ok]
    )
    det = det[ok]
    inv0 = cov2d[:, 1, 1] / det
    inv1 = -cov2d[:, 0, 1] / det
    inv3 = cov2d[:, 0, 0] / det

    rad = 3.0 * np.sqrt(np.maximum(inv0, inv3))
    x0 = np.maximum(0, (u - rad).astype(np.int32))
    x1 = np.minimum(W - 1, (u + rad).astype(np.int32))
    y0 = np.maximum(0, (v - rad).astype(np.int32))
    y1 = np.minimum(H - 1, (v + rad).astype(np.int32))
    valid = (x1 >= x0) & (y1 >= y0) & (opacity_k > 0.01)
    x0, x1, y0, y1 = x0[valid], x1[valid], y0[valid], y1[valid]
    u, v, inv0, inv1, inv3, opacity_k, colors_k, z = (
        u[valid], v[valid], inv0[valid], inv1[valid], inv3[valid],
        opacity_k[valid], colors_k[valid], z[valid],
    )
    n = len(x0)
    if n == 0:
        return None

    order = np.argsort(-z, kind="stable")
    x0s, x1s, y0s, y1s = x0[order], x1[order], y0[order], y1[order]
    us, vs, i0s, i1s, i3s = u[order], v[order], inv0[order], inv1[order], inv3[order]
    ops, cols = opacity_k[order], colors_k[order]

    alpha_acc = np.zeros((H, W), dtype=np.float32)
    color_acc = np.zeros((H, W, 3), dtype=np.float32)
    for i in range(n):
        if ops[i] < 0.01:
            continue
        X0, X1, Y0, Y1 = int(x0s[i]), int(x1s[i]), int(y0s[i]), int(y1s[i])
        gx = np.arange(X0, X1 + 1, dtype=np.float32) - us[i]
        gy = np.arange(Y0, Y1 + 1, dtype=np.float32) - vs[i]
        G = (
            gx[None, :] * gx[None, :] * i0s[i]
            + 2 * i1s[i] * gx[None, :] * gy[:, None]
            + gy[:, None] * gy[:, None] * i3s[i]
        )
        T = np.exp(-0.5 * G) * ops[i]
        if T.max() < 0.001:
            continue
        T = T * (1.0 - alpha_acc[Y0:Y1 + 1, X0:X1 + 1])
        color_acc[Y0:Y1 + 1, X0:X1 + 1] += T[:, :, None] * cols[i][None, None, :]
        alpha_acc[Y0:Y1 + 1, X0:X1 + 1] += T

    out = (1.0 - alpha_acc[:, :, None]) * bg + color_acc * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def load_mesh(ply_path):
    """Load a plain triangle-mesh PLY (x,y,z + rgb + face vertex_indices)."""
    ply = PlyData.read(ply_path)
    el = {e.name: e for e in ply.elements}
    v = el["vertex"]
    xyz = np.column_stack([
        np.asarray(v["x"], dtype=np.float64),
        np.asarray(v["y"], dtype=np.float64),
        np.asarray(v["z"], dtype=np.float64),
    ])
    _vnames = set(getattr(v, "names", None) or [p.name for p in v.properties])
    if "red" in _vnames:
        colors = np.column_stack([
            np.asarray(v["red"], dtype=np.float64),
            np.asarray(v["green"], dtype=np.float64),
            np.asarray(v["blue"], dtype=np.float64),
        ]) / 255.0
    else:
        colors = np.full((len(xyz), 3), 0.7, dtype=np.float64)
    if "face" in el:
        faces = np.asarray(el["face"]["vertex_indices"].tolist(), dtype=np.int64).reshape(-1, 3)
    else:
        faces = np.zeros((0, 3), dtype=np.int64)
    return xyz, colors, faces


def mesh_to_pyvista(xyz, colors, faces, mesh_ply_path=None):
    """Build a pyvista PolyData from mesh arrays, applying baked UV texture if available."""
    import pyvista as pv

    if mesh_ply_path and mesh_ply_path.lower().endswith("_mesh.ply"):
        stem = mesh_ply_path[:-9]
        uv_path = stem + "_mesh_uv.bin"
        tex_path = stem + "_texture.png"
        if os.path.exists(uv_path) and os.path.exists(tex_path):
            uvs = np.fromfile(uv_path, dtype="<f4").reshape(-1, 2)
            if len(uvs) == len(faces) * 3:
                exp_xyz = xyz[faces.reshape(-1)].astype(np.float64)
                fc = np.empty((len(faces), 4), dtype=np.int64)
                fc[:, 0] = 3
                fc[:, 1:] = np.arange(len(exp_xyz), dtype=np.int64).reshape(-1, 3)
                mesh = pv.PolyData(exp_xyz, fc)
                mesh.active_texture_coordinates = uvs.astype(np.float32)
                mesh._idol_texture = pv.read_texture(tex_path)
                return mesh

    fc = np.empty((len(faces), 4), dtype=np.int64)
    fc[:, 0] = 3
    fc[:, 1:] = faces
    mesh = pv.PolyData(xyz.astype(np.float64), fc)
    mesh["color"] = colors.astype(np.float32)
    return mesh


def render_mesh_frame(mesh, R, eye, f, cx, cy, W, H, bg, center, radius):
    """Render the mesh from the given orbit view using an offscreen VTK plotter."""
    import pyvista as pv

    up_world = np.asarray(R[1], dtype=np.float64)
    par_scale = (2.0 * radius) / 0.8
    plotter = pv.Plotter(off_screen=True, window_size=(W, H))
    plotter.set_background("#0e0f1a")
    tex = getattr(mesh, "_idol_texture", None)
    if tex is not None:
        plotter.add_mesh(
            mesh, texture=tex, smooth_shading=True,
            ambient=0.60, diffuse=0.45, specular=0.05,
        )
    else:
        plotter.add_mesh(
            mesh, scalars="color", rgb=True, smooth_shading=True,
            ambient=0.55, diffuse=0.50, specular=0.05,
        )
    cam = plotter.camera
    cam.position = tuple(eye)
    cam.focal_point = tuple(center)
    cam.view_up = tuple(up_world)
    cam.parallel_projection = True
    cam.parallel_scale = par_scale

    fd, tmp = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        plotter.screenshot(tmp)
        with Image.open(tmp) as im:
            frame = im.convert("RGB")
    finally:
        plotter.close()
        try:
            os.remove(tmp)
        except OSError:
            pass
    return np.asarray(frame)


def load_splats(ply_path, max_gaussians, rng):
    ply = PlyData.read(ply_path)
    el = ply["vertex"]
    xyz = np.stack([el["x"], el["z"], -el["y"]], axis=1).astype(np.float32)
    sh0 = np.array(el["f_dc_0"], dtype=np.float32)
    sh1 = np.array(el["f_dc_1"], dtype=np.float32)
    sh2 = np.array(el["f_dc_2"], dtype=np.float32)
    scale = np.exp(np.stack([el["scale_0"], el["scale_1"], el["scale_2"]], axis=1).astype(np.float32))
    raw_rot = np.stack([el["rot_0"], el["rot_1"], el["rot_2"], el["rot_3"]], axis=1).astype(np.float32)
    c = 0.5 ** 0.5
    sw, sx, sy, sz = raw_rot[:, 0], raw_rot[:, 1], raw_rot[:, 2], raw_rot[:, 3]
    rot = np.stack([c * (sw + sx), c * (sx - sw), c * (sy + sz), c * (sz - sy)], axis=1)
    raw_op = np.array(el["opacity"], dtype=np.float32).reshape(-1)
    opacity = 1.0 / (1.0 + np.exp(-raw_op))
    colors = 0.5 + 0.28209479177387814 * np.stack([sh0, sh1, sh2], axis=1)
    colors = np.clip(colors, 0.0, 1.0)

    n = len(xyz)
    if max_gaussians and n > max_gaussians:
        idx = rng.choice(n, size=max_gaussians, replace=False)
        xyz, scale, rot, opacity, colors = (
            xyz[idx], scale[idx], rot[idx], opacity[idx], colors[idx]
        )
    return xyz, colors, scale, rot, opacity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--mesh", action="store_true",
                    help="treat --ply as a triangle mesh (x,y,z+rgb+faces) and rasterize it")
    ap.add_argument("--meta", default=None)
    ap.add_argument("--out", default="preview.png")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--frames", type=int, default=1, help=">1 renders an orbit GIF")
    ap.add_argument("--fov", type=float, default=38.0, help="vertical FOV in degrees")
    ap.add_argument("--fit", type=float, default=0.62, help="fraction of frame height the model fills")
    ap.add_argument("--pitch", type=float, default=None, help="orbit pitch (rad); default 0.32")
    ap.add_argument("--max_gaussians", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bg", default="#0e0f1a")
    ap.add_argument("--fps", type=int, default=12, help="GIF loop speed")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    mesh_obj = None
    if args.mesh:
        xyz, colors, faces = load_mesh(args.ply)
        mesh_obj = mesh_to_pyvista(xyz, colors, faces, mesh_ply_path=args.ply)
        scale = rotQ = opacity = None
    else:
        xyz, colors, scale, rotQ, opacity = load_splats(args.ply, args.max_gaussians, rng)

    lo, hi = xyz.min(0), xyz.max(0)
    center = ((lo + hi) / 2.0).astype(np.float64)
    r = 0.5 * float(np.max(hi - lo)) or 0.5
    fov_rad = math.radians(args.fov)
    dist = r / (args.fit * math.tan(fov_rad / 2.0)) * 1.06
    pitch = 0.32 if args.pitch is None else args.pitch

    meta_yaw0 = 0.0
    if args.meta and os.path.exists(args.meta):
        try:
            with open(args.meta, "r", encoding="utf-8") as mf:
                cam = json.load(mf).get("camera", {})
            meta_yaw0 = float(cam.get("yaw", 0.0))
        except Exception:
            pass

    f = (args.height / 2.0) / math.tan(fov_rad / 2.0)
    cx = args.width / 2.0
    cy = args.height / 2.0
    W, H = args.width, args.height
    bg = np.array(Image.new("RGB", (W, H), args.bg)).astype(np.float32)
    world_up = np.array([0.0, 0.0, 1.0])

    def camera_for(yaw):
        cyaw, syaw = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        eye = center + dist * np.array([cp * syaw, -cp * cyaw, sp])
        fwd = center - eye
        fwd = fwd / np.linalg.norm(fwd)
        right = np.cross(world_up, fwd)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1.0, 0.0, 0.0])
        right = right / np.linalg.norm(right)
        up = np.cross(fwd, right)
        R = np.vstack([right, up, fwd])
        return R, eye

    def render(yaw):
        R, eye = camera_for(yaw)
        if args.mesh:
            return render_mesh_frame(mesh_obj, R, eye, f, cx, cy, W, H, bg, center, r)
        return render_frame(xyz, colors, scale, rotQ, opacity, R, f, cx, cy,
                            eye.astype(np.float32), W, H, bg)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if args.frames <= 1:
        img = render(meta_yaw0)
        if img is None:
            raise SystemExit("no gaussians in front of camera")
        Image.fromarray(img).save(args.out)
        print(f"saved {args.out} ({W}x{H}, {len(xyz)} gaussians, fit dist={dist:.3f})")
        return

    n = args.frames
    frames = []
    for i in range(n):
        yaw = meta_yaw0 + (2.0 * math.pi) * i / n
        img = render(yaw)
        if img is None:
            continue
        frames.append(img)
    if not frames:
        raise SystemExit("no gaussians in front of camera")
    ims = [Image.fromarray(fr) for fr in frames]
    duration = max(40, int(1000.0 / args.fps))
    ims[0].save(
        args.out, save_all=True, append_images=ims[1:],
        duration=duration, loop=0, optimize=True,
    )
    print(f"saved {args.out} (GIF {W}x{H}, {len(ims)} frames, {len(xyz)} gaussians, fit dist={dist:.3f})")


if __name__ == "__main__":
    main()
