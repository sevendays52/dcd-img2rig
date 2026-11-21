"""Cross-platform smoke & integration tests for FBX writing, auto-rigging, and server config."""

from __future__ import annotations

import json
import os
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from img2rig import config, fbx_writer, rig_bind


def _make_humanoid_box_mesh():
    """Create a simple watertight Z-up box mesh spanning [-0.25, 0.25] x [-0.15, 0.15] x [-0.5, 0.5]."""
    verts = np.array([
        [-0.25, -0.15, -0.50], [0.25, -0.15, -0.50], [0.25, 0.15, -0.50], [-0.25, 0.15, -0.50],
        [-0.25, -0.15,  0.50], [0.25, -0.15,  0.50], [0.25, 0.15,  0.50], [-0.25, 0.15,  0.50],
    ], dtype=np.float32)
    faces = np.array([
        [0, 1, 2], [0, 2, 3],
        [4, 6, 5], [4, 7, 6],
        [0, 4, 5], [0, 5, 1],
        [3, 2, 6], [3, 6, 7],
        [1, 5, 6], [1, 6, 2],
        [4, 0, 3], [4, 3, 7],
    ], dtype=np.int32)
    colors = np.tile(np.array([[0.8, 0.5, 0.3]], dtype=np.float32), (8, 1))
    return verts, faces, colors


def _read_glb(path: Path):
    data = path.read_bytes()
    magic, ver, _total = struct.unpack_from("<4sII", data, 0)
    assert magic == b"glTF" and ver == 2
    clen, ctype = struct.unpack_from("<I4s", data, 12)
    assert ctype == b"JSON"
    j = json.loads(data[20:20 + clen].decode("utf-8"))
    off = 20 + clen
    bin_data = b""
    if off < len(data):
        blen, btype = struct.unpack_from("<I4s", data, off)
        assert btype[:3] == b"BIN"
        bin_data = data[off + 8:off + 8 + blen]
    return j, bin_data


def test_config_paths_exist():
    assert config.VIEWER_DIR.is_dir()
    assert (config.VIEWER_DIR / "index.html").is_file()
    assert config.DEFAULT_RIG_GLB.is_file()
    assert (config.PRESETS_DIR / "walk.fbx").is_file()


def test_binary_fbx_export():
    verts, faces, colors = _make_humanoid_box_mesh()
    with tempfile.TemporaryDirectory() as tmpd:
        out_fbx = Path(tmpd) / "box.fbx"
        res = fbx_writer.write_fbx_binary(str(out_fbx), verts, faces, vertex_colors=colors, name="test_box")
        assert out_fbx.is_file()
        assert res["verts"] == 8 and res["faces"] == 12
        assert out_fbx.read_bytes()[:23] == b"Kaydara FBX Binary  \x00\x1a\x00"


def _make_dense_humanoid_mesh():
    """Create a >100-vertex watertight cylinder-like mesh to exercise anatomical fitting."""
    n_rings = 12
    n_seg = 12
    zs = np.linspace(-0.5, 0.5, n_rings)
    verts = []
    for zi, z in enumerate(zs):
        r = 0.14 if 0.2 < z < 0.45 else 0.18
        for si in range(n_seg):
            ang = 2.0 * np.pi * si / n_seg
            verts.append([r * np.cos(ang), r * 0.7 * np.sin(ang), z])
    verts = np.array(verts, dtype=np.float32)
    faces = []
    for zi in range(n_rings - 1):
        for si in range(n_seg):
            sn = (si + 1) % n_seg
            a = zi * n_seg + si
            b = zi * n_seg + sn
            c = (zi + 1) * n_seg + si
            d = (zi + 1) * n_seg + sn
            faces.append([a, b, d])
            faces.append([a, d, c])
    faces = np.array(faces, dtype=np.int32)
    colors = np.tile(np.array([[0.7, 0.6, 0.5]], dtype=np.float32), (len(verts), 1))
    return verts, faces, colors


def test_rig_bind_produces_valid_glb_and_fbx():
    for builder, use_blender in [
        (_make_humanoid_box_mesh, False),
        (_make_dense_humanoid_mesh, True),
    ]:
        verts, faces, colors = builder()
        with tempfile.TemporaryDirectory() as tmpd:
            out_glb = Path(tmpd) / "skinned.glb"
            out_fbx = Path(tmpd) / "skinned.fbx"
            info = rig_bind.bind_rig_to_mesh(
                str(config.DEFAULT_RIG_GLB),
                verts, faces, colors,
                verts.min(axis=0).tolist(), verts.max(axis=0).tolist(),
                out_glb=str(out_glb),
                out_fbx=str(out_fbx),
                mesh_name="test_char",
                use_blender_weights=use_blender,
            )
            assert info["bones"] == 19
            assert out_glb.is_file()
            assert out_fbx.is_file()
            assert out_fbx.read_bytes()[:23] == b"Kaydara FBX Binary  \x00\x1a\x00"

            gltf, bin_data = _read_glb(out_glb)
            prim = gltf["meshes"][0]["primitives"][0]
            attrs = prim["attributes"]
            assert "JOINTS_0" in attrs and "WEIGHT_0" in attrs

            acc_j = gltf["accessors"][attrs["JOINTS_0"]]
            bv_j = gltf["bufferViews"][acc_j["bufferView"]]
            j_off = bv_j.get("byteOffset", 0) + acc_j.get("byteOffset", 0)
            joints = np.frombuffer(bin_data, dtype="<u2", count=acc_j["count"] * 4, offset=j_off).reshape(-1, 4)
            assert joints.min() >= 0 and joints.max() < 19

            acc_w = gltf["accessors"][attrs["WEIGHT_0"]]
            bv_w = gltf["bufferViews"][acc_w["bufferView"]]
            w_off = bv_w.get("byteOffset", 0) + acc_w.get("byteOffset", 0)
            weights = np.frombuffer(bin_data, dtype="<f4", count=acc_w["count"] * 4, offset=w_off).reshape(-1, 4)
            sums = weights.sum(axis=1)
            assert np.allclose(sums, 1.0, atol=1e-3), f"weight sums out of range: {sums}"


if __name__ == "__main__":
    test_config_paths_exist()
    test_binary_fbx_export()
    test_rig_bind_produces_valid_glb_and_fbx()
    print("All smoke tests passed.")
