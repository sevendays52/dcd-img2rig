"""Mixamo / FBX motion retargeting onto a rigged SweetieRig GLB via Blender headless.

How it works (zero Y-up/Z-up round-trip drift):
  1. Blender headless imports ONLY the motion file (.fbx / .glb / .gltf / .bvh),
     which Blender natively converts into upright Z-up world space, and dumps
     each source bone's rest + per-frame world matrices to JSON.
  2. In Python, we load the target Z-up skinned GLB's exact rest nodes and
     world bind matrices (via rig_bind.load_rig_glb), compute world-space
     rest-compensated delta rotations:
         delta_R_world(f) = R_src_world(f) @ R_src_rest_world.T
         R_dst_world(f)   = delta_R_world(f) @ R_dst_rest_world
     convert back to each target bone's parent-local quaternion (and scaled
     root hip translation), and embed the glTF 2.0 `animations` chunk directly
     into a copy of the target skinned GLB.

At rest pose (delta_R = I, delta_t = 0), every animated channel equals the
target GLB node's exact rest TRS to floating-point precision.

CLI:
    .venv/bin/python motion_retarget.py \
        --skinned outputs/X_skinned.glb \
        --motion rigs/presets/walk.fbx \
        --out outputs/X_anim_walk.glb
"""
import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np

from . import fbx_writer, rig_bind


BONE_ALIASES = {
    "root": ["hips", "pelvis", "root"],
    "spine": ["spine"],
    "chest": ["spine2", "spine1", "chest", "upperchest"],
    "neck": ["neck"],
    "head": ["head"],
    "lshoulder": ["leftshoulder", "shoulder_l", "l_shoulder"],
    "lupperarm": ["leftarm", "leftupperarm", "upperarm_l", "l_upperarm"],
    "llowerarm": ["leftforearm", "leftlowerarm", "lowerarm_l", "l_lowerarm"],
    "lhand": ["lefthand", "hand_l", "l_hand"],
    "rshoulder": ["rightshoulder", "shoulder_r", "r_shoulder"],
    "rupperarm": ["rightarm", "rightupperarm", "upperarm_r", "r_upperarm"],
    "rlowerarm": ["rightforearm", "rightlowerarm", "lowerarm_r", "r_lowerarm"],
    "rhand": ["righthand", "hand_r", "r_hand"],
    "lthigh": ["leftupleg", "leftthigh", "thigh_l", "l_thigh"],
    "lshin": ["leftleg", "leftshin", "shin_l", "l_shin"],
    "lfoot": ["leftfoot", "foot_l", "l_foot"],
    "rthigh": ["rightupleg", "rightthigh", "thigh_r", "r_thigh"],
    "rshin": ["rightleg", "rightshin", "shin_r", "r_shin"],
    "rfoot": ["rightfoot", "foot_r", "r_foot"],
}


def _norm_name(name):
    s = name.split(":")[-1].split("|")[-1]
    return s.strip().lower().replace(" ", "").replace(".", "")


_EXTRACT_MOTION_SCRIPT = r"""
import json, os, sys
import bpy

args = sys.argv[sys.argv.index("--") + 1:]
motion_path, out_json = args[0], args[1]

bpy.ops.wm.read_factory_settings(use_empty=True)

ext = os.path.splitext(motion_path)[1].lower()
if ext == ".fbx":
    bpy.ops.import_scene.fbx(filepath=motion_path, automatic_bone_orientation=False)
elif ext in (".glb", ".gltf"):
    bpy.ops.import_scene.gltf(filepath=motion_path)
elif ext == ".bvh":
    bpy.ops.import_anim.bvh(filepath=motion_path)
else:
    raise RuntimeError(f"Unsupported motion format: {ext}")

src_arm = next((o for o in bpy.data.objects if o.type == "ARMATURE"), None)
if src_arm is None:
    raise RuntimeError("Motion file contains no Armature")

scene = bpy.context.scene
act = src_arm.animation_data.action if src_arm.animation_data else None
if act:
    f_start = int(round(act.frame_range[0]))
    f_end = int(round(act.frame_range[1]))
else:
    f_start, f_end = 1, 60
if f_end <= f_start:
    f_end = f_start + 1

fps = float(scene.render.fps / max(1e-6, scene.render.fps_base))
arm_w = src_arm.matrix_world.copy()

def mat3_list(m3):
    m = m3.normalized()
    return [[float(m[r][c]) for c in range(3)] for r in range(3)]

rest_rot = {}
rest_pos = {}
zs = []
for b in src_arm.data.bones:
    mw = arm_w @ b.matrix_local
    rest_rot[b.name] = mat3_list(mw.to_3x3())
    p = mw.to_translation()
    rest_pos[b.name] = [float(p.x), float(p.y), float(p.z)]
    zs.append(float(p.z))

src_z_span = max(1e-4, (max(zs) - min(zs))) if zs else 1.0

frames = []
for f in range(f_start, f_end + 1):
    scene.frame_set(f)
    bpy.context.view_layer.update()
    f_rot = {}
    f_pos = {}
    for pb in src_arm.pose.bones:
        mw = src_arm.matrix_world @ pb.matrix
        f_rot[pb.name] = mat3_list(mw.to_3x3())
        p = mw.to_translation()
        f_pos[pb.name] = [float(p.x), float(p.y), float(p.z)]
    frames.append({"rot": f_rot, "pos": f_pos})

with open(out_json, "w", encoding="utf-8") as fh:
    json.dump({
        "fps": fps,
        "frame_start": f_start,
        "frame_end": f_end,
        "z_span": src_z_span,
        "rest_rot": rest_rot,
        "rest_pos": rest_pos,
        "frames": frames,
    }, fh)
"""


def _read_glb(path):
    with open(path, "rb") as fh:
        data = fh.read()
    magic, version, _ = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF" or version != 2:
        raise ValueError(f"Not a valid glTF 2.0 binary: {path}")
    off = 12
    json_chunk = bin_chunk = b""
    while off < len(data):
        clen, ctype = struct.unpack_from("<I4s", data, off)
        off += 8
        chunk = data[off:off + clen]
        if ctype == b"JSON":
            json_chunk = chunk
        elif ctype[:3] == b"BIN":
            bin_chunk = chunk
        off += (clen + 3) // 4 * 4
    return json.loads(json_chunk.decode("utf-8")), bytearray(bin_chunk)


def _write_glb(path, gltf, bin_bytes):
    bin_pad = (-len(bin_bytes)) % 4
    bin_full = bytes(bin_bytes) + b"\x00" * bin_pad
    json_raw = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    json_pad = (-len(json_raw)) % 4
    json_full = json_raw + b" " * json_pad
    total = 12 + 8 + len(json_full) + 8 + len(bin_full)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<4sII", b"glTF", 2, total))
        fh.write(struct.pack("<I4s", len(json_full), b"JSON"))
        fh.write(json_full)
        fh.write(struct.pack("<I4s", len(bin_full), b"BIN\x00"))
        fh.write(bin_full)


def _orthonormalize_3x3(M):
    """Extract pure rotation matrix from a 3x3 (handles uniform/non-uniform scale)."""
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def _damp_rot3(R, factor):
    """Scale rotation matrix R's angle by `factor` in [0, 1] (SLERP from I to R)."""
    if factor >= 0.999:
        return R
    if factor <= 1e-4:
        return np.identity(3, dtype=np.float64)
    tr = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(tr))
    if angle < 1e-6:
        return np.identity(3, dtype=np.float64)
    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ], dtype=np.float64)
    alen = float(np.linalg.norm(axis))
    if alen < 1e-8:
        return np.identity(3, dtype=np.float64)
    axis /= alen
    a = angle * factor
    c, s = float(np.cos(a)), float(np.sin(a))
    ux, uy, uz = axis
    K = np.array([[0.0, -uz, uy], [uz, 0.0, -ux], [-uy, ux, 0.0]], dtype=np.float64)
    return np.identity(3, dtype=np.float64) + s * K + (1.0 - c) * (K @ K)


def _quat_from_rot3(R):
    """3x3 proper rotation matrix -> unit quaternion [x, y, z, w] (glTF order)."""
    q = np.asarray(rig_bind._quat_from_mat(R), dtype=np.float64)
    n = float(np.linalg.norm(q))
    return (q / n) if n > 1e-12 else np.array([0.0, 0.0, 0.0, 1.0])


def retarget_motion(skinned_glb, motion_path, out_glb, timeout=180):
    """Retarget a Mixamo/FBX/GLB/BVH motion onto skinned_glb and write out_glb."""
    blender = fbx_writer.find_blender()
    if not blender:
        raise RuntimeError("Blender executable not found (required for motion retargeting)")
    if not os.path.exists(skinned_glb):
        raise FileNotFoundError(f"Skinned GLB not found: {skinned_glb}")
    if not os.path.exists(motion_path):
        raise FileNotFoundError(f"Motion file not found: {motion_path}")

    # 1. Extract source motion world transforms in Blender Z-up
    with tempfile.TemporaryDirectory(prefix="idol_retarget_") as tmpdir:
        script_py = os.path.join(tmpdir, "extract_motion.py")
        motion_json = os.path.join(tmpdir, "motion.json")
        with open(script_py, "w", encoding="utf-8") as fh:
            fh.write(_EXTRACT_MOTION_SCRIPT)

        proc = subprocess.run(
            [blender, "--background", "--factory-startup",
             "--python", script_py, "--",
             os.path.abspath(motion_path),
             motion_json],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
        )
        if proc.returncode != 0 or not os.path.exists(motion_json):
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-10:]
            raise RuntimeError("Blender motion extraction failed: " + "\n".join(tail))

        with open(motion_json, "r", encoding="utf-8") as fh:
            mdata = json.load(fh)

    # 2. Load target skinned GLB (exact Z-up space, untouched)
    dst_j, dst_bin = _read_glb(skinned_glb)
    rig = rig_bind.load_rig_glb(skinned_glb)
    skin_joints = dst_j["skins"][0]["joints"]
    nb = len(rig["bones"])

    # Map target bones -> source bones
    src_by_norm = {_norm_name(k): k for k in mdata["rest_rot"].keys()}
    mapping = {}  # target bone idx -> source bone name
    for bi, b in enumerate(rig["bones"]):
        dn = _norm_name(b["name"])
        for cand in BONE_ALIASES.get(dn, [dn]):
            if cand in src_by_norm:
                mapping[bi] = src_by_norm[cand]
                break

    if not mapping:
        raise RuntimeError("No matching bones found between character and motion rig")

    # Target rest world rotations (pure 3x3) and bone span
    dst_rest_rot = [_orthonormalize_3x3(np.asarray(bw, dtype=np.float64)[:3, :3])
                    for bw in rig["bind_world"]]
    dst_rest_pos = [np.asarray(bw, dtype=np.float64)[:3, 3]
                    for bw in rig["bind_world"]]
    dst_zs = [p[2] for p in dst_rest_pos]
    dst_z_span = max(1e-4, max(dst_zs) - min(dst_zs)) if dst_zs else 1.0
    height_scale = dst_z_span / max(1e-4, float(mdata["z_span"]))

    # Topological order (parent before child)
    order = []
    seen = set()
    def visit(i):
        if i in seen:
            return
        p = rig["bones"][i].get("parent")
        if p is not None and 0 <= p < nb:
            visit(p)
        seen.add(i)
        order.append(i)
    for i in range(nb):
        visit(i)

    # Source rest world rotations (inverse = transpose for pure rotation)
    src_rest_R_inv = {
        k: _orthonormalize_3x3(np.asarray(v, dtype=np.float64)).T
        for k, v in mdata["rest_rot"].items()
    }
    src_rest_pos = {
        k: np.asarray(v, dtype=np.float64)
        for k, v in mdata["rest_pos"].items()
    }

    frames = mdata["frames"]
    nf = len(frames)
    fps = float(mdata["fps"])
    times = (np.arange(nf, dtype=np.float32) / fps).astype("<f4")

    root_bi = next((bi for bi, b in enumerate(rig["bones"]) if b.get("parent") is None), 0)
    root_node = dst_j["nodes"][skin_joints[root_bi]]
    root_rest_t = np.asarray(root_node.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64)

    # Target facing yaw alignment & SD-mascot proportion damping
    extras = dst_j.get("asset", {}).get("extras") or {}
    facing_yaw = float(extras.get("facing_yaw", 0.0))
    is_sd = bool(extras.get("is_sd", False))
    cy, sy = float(np.cos(facing_yaw)), float(np.sin(facing_yaw))
    R_align = np.array([
        [cy, -sy, 0.0],
        [sy,  cy, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    R_align_T = R_align.T

    # On small/chubby SD characters, gently damp large human joint angles so
    # walking looks like a cute, crisp waddle without shearing the plump body.
    def bone_damp(bname):
        if not is_sd:
            return 1.0
        n = _norm_name(bname)
        if n in ("head", "neck"):
            return 0.45
        if n in ("root", "spine", "chest"):
            return 0.55
        if "shin" in n or "foot" in n:
            return 0.50
        if "thigh" in n:
            return 0.65
        return 0.60

    root_trans_scale = (0.65 if is_sd else 1.0) * height_scale

    # Preallocate per-bone quaternion tracks (nf, 4) and root translation (nf, 3)
    bone_quats = {bi: np.zeros((nf, 4), dtype=np.float32) for bi in range(nb)}
    root_trans = np.zeros((nf, 3), dtype=np.float32)

    for fi, fr in enumerate(frames):
        posed_R_world = [None] * nb
        for bi in order:
            p = rig["bones"][bi].get("parent")
            if bi in mapping:
                sb = mapping[bi]
                R_src = _orthonormalize_3x3(np.asarray(fr["rot"][sb], dtype=np.float64))
                delta_src = R_src @ src_rest_R_inv[sb]
                delta_src = _damp_rot3(delta_src, bone_damp(rig["bones"][bi]["name"]))
                delta_w = R_align @ delta_src @ R_align_T
                R_dst_w = delta_w @ dst_rest_rot[bi]
            else:
                if p is not None and posed_R_world[p] is not None:
                    # Keep rest local rotation relative to parent
                    R_local_rest = dst_rest_rot[p].T @ dst_rest_rot[bi]
                    R_dst_w = posed_R_world[p] @ R_local_rest
                else:
                    R_dst_w = dst_rest_rot[bi]
            posed_R_world[bi] = R_dst_w

            if p is not None and posed_R_world[p] is not None:
                R_local = posed_R_world[p].T @ R_dst_w
            else:
                R_local = R_dst_w

            q = _quat_from_rot3(R_local)
            # Enforce quaternion sign continuity across adjacent frames for smooth SLERP
            if fi > 0 and float(np.dot(bone_quats[bi][fi - 1], q)) < 0.0:
                q = -q
            bone_quats[bi][fi] = q.astype(np.float32)

        if root_bi in mapping:
            sb = mapping[root_bi]
            dp = np.asarray(fr["pos"][sb], dtype=np.float64) - src_rest_pos[sb]
            dp_char = R_align @ dp
            root_trans[fi] = (root_rest_t + dp_char * root_trans_scale).astype(np.float32)
        else:
            root_trans[fi] = root_rest_t.astype(np.float32)

    # 3. Append glTF 2.0 animation buffers/accessors directly into dst_j + dst_bin
    dst_bvs = dst_j.setdefault("bufferViews", [])
    dst_accs = dst_j.setdefault("accessors", [])

    def append_accessor(arr, comp_type, acc_type, with_minmax=False):
        raw = np.ascontiguousarray(arr, dtype="<f4").tobytes()
        pad = (-len(dst_bin)) % 4
        if pad:
            dst_bin.extend(b"\x00" * pad)
        off = len(dst_bin)
        dst_bin.extend(raw)
        bv_idx = len(dst_bvs)
        dst_bvs.append({"buffer": 0, "byteOffset": off, "byteLength": len(raw)})
        acc = {
            "bufferView": bv_idx,
            "componentType": comp_type,
            "count": int(arr.shape[0]),
            "type": acc_type,
        }
        if with_minmax:
            acc["min"] = [float(arr.min())]
            acc["max"] = [float(arr.max())]
        acc_idx = len(dst_accs)
        dst_accs.append(acc)
        return acc_idx

    time_acc = append_accessor(times, 5126, "SCALAR", with_minmax=True)

    samplers = []
    channels = []

    # Root translation channel
    rt_acc = append_accessor(root_trans, 5126, "VEC3")
    samplers.append({"input": time_acc, "interpolation": "LINEAR", "output": rt_acc})
    channels.append({
        "sampler": len(samplers) - 1,
        "target": {"node": skin_joints[root_bi], "path": "translation"},
    })

    # Per-bone rotation channels
    for bi in range(nb):
        rq_acc = append_accessor(bone_quats[bi], 5126, "VEC4")
        samplers.append({"input": time_acc, "interpolation": "LINEAR", "output": rq_acc})
        channels.append({
            "sampler": len(samplers) - 1,
            "target": {"node": skin_joints[bi], "path": "rotation"},
        })

    anim_name = os.path.splitext(os.path.basename(motion_path))[0]
    dst_j["animations"] = [{
        "name": anim_name,
        "channels": channels,
        "samplers": samplers,
    }]
    if dst_j.get("buffers"):
        dst_j["buffers"][0]["byteLength"] = len(dst_bin)

    os.makedirs(os.path.dirname(os.path.abspath(out_glb)), exist_ok=True)
    _write_glb(out_glb, dst_j, dst_bin)

    return {
        "mapped_bones": len(mapping),
        "frames": nf,
        "fps": fps,
        "duration": round(float(times[-1]) if nf > 0 else 0.0, 3),
        "out_glb": out_glb,
        "bytes": os.path.getsize(out_glb),
    }


def main():
    ap = argparse.ArgumentParser(description="Retarget Mixamo FBX motion onto skinned GLB")
    ap.add_argument("--skinned", required=True, help="input *_skinned.glb")
    ap.add_argument("--motion", required=True, help="input Mixamo .fbx / .glb / .bvh")
    ap.add_argument("--out", required=True, help="output animated .glb")
    args = ap.parse_args()
    info = retarget_motion(args.skinned, args.motion, args.out)
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
