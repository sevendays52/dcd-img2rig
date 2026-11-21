"""Skinning bind pipeline for the reconstructed mesh (Phase 2).

Responsibilities:
  * Read a rig GLB authored in Z-up metric space (rigs/sweetie_rig.glb) and
    return bones with WORLD bind matrices, in the SAME space as the
    reconstructed mesh (feet ~ z=-0.5, crown ~ +0.5, ~1.0 m tall).
  * Fit the authored rig to the reconstructed mesh's bounding box
    (uniform-scale + center align) so the bind pose matches the actual body.
  * Compute per-vertex skin weights (nearest-bone-by-bone-segment).
  * Write the skinned GLB (web-viewer canonical format) and the skinned FBX
    (Blender / game-engine format) via the fbx_writer module.

The whole module stays in the mesh's native Z-up metric space: no axis swaps,
matching convert_fbx.py output and the web viewer GLB parser.

Usage as library:
    from rig_bind import load_rig_glb, fit_rig_to_mesh, bind_rig_to_mesh
"""
import json
import struct

import numpy as np

from . import fbx_writer

# glTF component types (per glTF 2.0 spec)
T_BYTE, T_UBYTE = 5120, 5121
T_SHORT, T_USHORT = 5122, 5123
T_INT, T_UINT, T_FLOAT = 5124, 5125, 5126
_TYPE_N = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4,
           "MAT2": 4, "MAT3": 9, "MAT4": 16}
_SIZE = {T_BYTE: 1, T_UBYTE: 1, T_SHORT: 2, T_USHORT: 2, T_UINT: 4, T_FLOAT: 4}


# ---------------------------------------------------------------------------
# GLB reader (minimal, dependency-free): nodes with TRS + skins + inverses.
# ---------------------------------------------------------------------------

def _glb_read_json_bin(path):
    with open(path, "rb") as fh:
        data = fh.read()
    magic, version, length = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF":
        raise ValueError("not a glb")
    off = 12
    chunks = {}
    while off < len(data):
        clen, ctype = struct.unpack_from("<I4s", data, off)
        off += 8
        chunks[ctype] = data[off:off + clen]
        off += clen
    json_part = chunks.get(b"JSON")
    bin_part = chunks.get(b"BIN\x00") or chunks.get(b"BIN") or b""
    if not json_part:
        raise ValueError("missing JSON chunk")
    return json.loads(json_part.decode("utf-8")), bin_part


def _accessor_data(gltf, bin_part, acc):
    a = gltf["accessors"][acc]
    bv = gltf["bufferViews"][a["bufferView"]]
    byte_off = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
    n = _TYPE_N[a["type"]]
    comp = _SIZE[a["componentType"]]
    count = a["count"]
    raw = bin_part[byte_off:byte_off + count * n * comp]
    if a["componentType"] == T_USHORT:
        arr = np.frombuffer(raw, dtype="<u2").reshape(count, n)
    elif a["componentType"] == T_UINT:
        arr = np.frombuffer(raw, dtype="<u4").reshape(count, n)
    else:
        arr = np.frombuffer(raw, dtype="<f4").reshape(count, n)
    return arr


def _mat4_from_trs(t, r, s):
    """Compose 4x4 (row-major numpy, column-vector convention) from TRS."""
    tx, ty, tz = (t or [0.0, 0.0, 0.0])
    sx, sy, sz = (s or [1.0, 1.0, 1.0])
    rx, ry, rz, rw = (r or [0.0, 0.0, 0.0, 1.0])
    x, y, z, w = rx, ry, rz, rw
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    R = np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz),     2 * (xz + wy)],
        [2 * (xy + wz),     1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy),     2 * (yz + wx),     1 - 2 * (xx + yy)],
    ])
    M = np.identity(4)
    # Scale is applied per rotation COLUMN (R @ S), not element-wise,
    # otherwise off-diagonal rotation entries are zeroed.
    M[:3, :3] = R @ np.array([[sx, 0, 0], [0, sy, 0], [0, 0, sz]])
    M[:3, 3] = [tx, ty, tz]
    return M


def _quat_from_mat(R):
    """Rotation matrix -> quaternion [x,y,z,w]."""
    R = np.asarray(R, dtype=np.float64)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w])
    q /= (np.linalg.norm(q) or 1.0)
    return q.tolist()


def _decompose(mat):
    """4x4 (R*S | t) -> (translation[3], quat[4], scale[3])."""
    mat = np.asarray(mat, dtype=np.float64)
    t = mat[:3, 3].tolist()
    m3 = mat[:3, :3]
    s = np.array([np.linalg.norm(m3[:, i]) for i in range(3)])
    s[s == 0] = 1.0
    R = m3 / s
    return t, _quat_from_mat(R), s.tolist()


def load_rig_glb(path):
    """Parse a Z-up rig GLB (Blender-authored: armature node + bone nodes).

    Returns dict:
      bones:       [{"name", "parent"(bone idx|None)} in skin.joints order]
      bind_world:  [4x4 per bone, world matrices in rig GLB native space]
      armature_world: 4x4 (identity for Blender exports)
      joint_inv:   [4x4 inverseBindingMatrix per bone (if present)]
    """
    gltf, bin_part = _glb_read_json_bin(path)

    parent_of = {}
    for i, nd in enumerate(gltf.get("nodes", [])):
        for c in nd.get("children", []):
            parent_of[c] = i

    def node_local_mat(i):
        nd = gltf["nodes"][i]
        if "matrix" in nd:
            return np.array(nd["matrix"], dtype=np.float64).reshape(4, 4,
                                                                    order="F")
        return _mat4_from_trs(nd.get("translation"), nd.get("rotation"),
                              nd.get("scale"))

    memo = {}

    def node_world(i):
        if i in memo:
            return memo[i]
        m = node_local_mat(i)
        p = parent_of.get(i)
        memo[i] = node_world(p) @ m if p is not None else m
        return memo[i]

    skins = gltf.get("skins", [])
    if not skins:
        raise ValueError("rig GLB has no skin")
    skin = skins[0]
    joints = list(skin["joints"])
    inv = skin.get("inverseBindingMatrices")

    bones = []
    bind_world = []
    for bi, jnd in enumerate(joints):
        bones.append({"name": gltf["nodes"][jnd].get("name", f"bone{bi}"),
                      "parent": None})
        bind_world.append(node_world(jnd))

    joint_to_bi = {j: i for i, j in enumerate(joints)}
    for bi, jnd in enumerate(joints):
        p = parent_of.get(jnd)
        if p is not None and p in joint_to_bi:
            bones[bi]["parent"] = joint_to_bi[p]

    joint_inv = []
    if inv is not None:
        if isinstance(inv, int):
            arr = _accessor_data(gltf, bin_part, inv)
            for k in range(len(arr)):
                joint_inv.append(arr[k].reshape(4, 4, order="F"))
        else:
            for ai in inv:
                joint_inv.append(_accessor_data(gltf, bin_part, ai).reshape(4, 4, order="F"))

    # Armature root = topmost ancestor of the first joint.
    i = joints[0]
    while i in parent_of:
        i = parent_of[i]
    arm_world = node_world(i)

    return {
        "bones": bones,
        "bind_world": [np.array(m, dtype=np.float64) for m in bind_world],
        "armature_world": np.array(arm_world, dtype=np.float64),
        "joint_inv": joint_inv,
    }


# ---------------------------------------------------------------------------
# Fit authored rig (metric Z-up) to reconstructed mesh bounding box.
# ---------------------------------------------------------------------------

def _compute_bone_tails(bones, bind_world):
    """Compute world-space tail position for each bone.

    Non-leaf bones end at the centroid of their direct children's heads.
    Leaf bones extend along their local +Y axis (Blender glTF bone axis)
    by half their parent bone's length (clamped to [0.06, 0.25] of rig span).
    """
    nb = len(bind_world)
    heads = np.array([np.asarray(bw, dtype=np.float64)[:3, 3] for bw in bind_world])
    children = [[] for _ in range(nb)]
    if bones is not None:
        for i, b in enumerate(bones):
            p = b.get("parent")
            if p is not None and 0 <= p < nb:
                children[p].append(i)

    span = float(np.linalg.norm(heads.max(axis=0) - heads.min(axis=0))) if nb > 1 else 1.0
    default_leaf = max(0.06, min(0.25, span * 0.10))

    tails = np.empty_like(heads)
    for i in range(nb):
        if children[i]:
            tails[i] = heads[children[i]].mean(axis=0)
        else:
            bw = np.asarray(bind_world[i], dtype=np.float64)
            y_dir = bw[:3, 1]
            nrm = float(np.linalg.norm(y_dir))
            y_dir = (y_dir / nrm) if nrm > 1e-8 else np.array([0.0, 0.0, 1.0])
            leaf_len = default_leaf
            if bones is not None:
                p = bones[i].get("parent")
                if p is not None and 0 <= p < nb:
                    plen = float(np.linalg.norm(heads[i] - heads[p]))
                    if plen > 1e-4:
                        leaf_len = max(0.05, min(0.25, plen * 0.7))
            tails[i] = heads[i] + y_dir * leaf_len
    return heads, tails


def _make_bone_matrix(head, tail, ref_z):
    """Build a right-handed orthonormal 4x4 bone bind_world matrix.

    Blender glTF bone convention: +Y is along head -> tail, +Z faces ref_z.
    """
    head = np.asarray(head, dtype=np.float64)
    tail = np.asarray(tail, dtype=np.float64)
    y_ax = tail - head
    ylen = float(np.linalg.norm(y_ax))
    y_ax = (y_ax / ylen) if ylen > 1e-8 else np.array([0.0, 0.0, 1.0])
    ref = np.asarray(ref_z, dtype=np.float64)
    x_ax = np.cross(y_ax, ref)
    xlen = float(np.linalg.norm(x_ax))
    if xlen < 1e-5:
        alt = np.array([1.0, 0.0, 0.0]) if abs(y_ax[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
        x_ax = np.cross(y_ax, alt)
        xlen = float(np.linalg.norm(x_ax))
    x_ax /= max(xlen, 1e-8)
    z_ax = np.cross(x_ax, y_ax)
    z_ax /= max(float(np.linalg.norm(z_ax)), 1e-8)
    M = np.identity(4, dtype=np.float64)
    M[:3, 0] = x_ax
    M[:3, 1] = y_ax
    M[:3, 2] = z_ax
    M[:3, 3] = head
    return M


def _estimate_facing_yaw(verts, meta_yaw=None):
    """Return character facing yaw (radians; 0 = -Y front, +pi/2 = +X front)."""
    if meta_yaw is not None:
        return float(meta_yaw)
    v = np.asarray(verts, dtype=np.float64)
    z = v[:, 2]
    zmin, zmax = float(z.min()), float(z.max())
    H = max(zmax - zmin, 1e-6)
    zn = (z - zmin) / H
    # Compare lateral spread in upper/lower bands where ears/legs separate bilaterally
    band = v[(zn < 0.35) | (zn > 0.72)]
    if len(band) < 20:
        band = v
    x_span = float(np.quantile(band[:, 0], 0.96) - np.quantile(band[:, 0], 0.04))
    y_span = float(np.quantile(band[:, 1], 0.96) - np.quantile(band[:, 1], 0.04))
    return float(np.pi * 0.5) if y_span > x_span * 1.08 else 0.0


def fit_rig_anatomical(rig, mesh_verts, meta_yaw=None):
    """Shape-adaptive anatomical fitting of the 19-bone humanoid rig to mesh_verts.

    Aligns the rig to the character's true facing direction (meta_yaw or auto),
    isolates the body core (ignoring back hair / fluffy tails), and detects
    head/neck, crotch/legs, and shoulder/elbow/hand joints directly from the
    mesh cross-sections.

    Returns (scale, fitted_bind_world, fitted_armature_world, fitted_tails, facing_yaw).
    """
    v = np.asarray(mesh_verts, dtype=np.float64)
    if len(v) < 50:
        _, scale, fitted, arm = fit_rig_to_mesh(rig, v.min(0), v.max(0))
        _, tails = _compute_bone_tails(rig.get("bones"), fitted)
        return scale, fitted, arm, tails, {"yaw": 0.0, "is_sd": False}

    yaw = _estimate_facing_yaw(v, meta_yaw=meta_yaw)
    cy, sy = float(np.cos(yaw)), float(np.sin(yaw))
    L_ax = np.array([cy, sy, 0.0], dtype=np.float64)     # character Left
    F_ax = np.array([sy, -cy, 0.0], dtype=np.float64)    # character Front
    U_ax = np.array([0.0, 0.0, 1.0], dtype=np.float64)   # Up

    l_all = v @ L_ax
    f_all = v @ F_ax
    z_all = v[:, 2]
    zmin, zmax = float(z_all.min()), float(z_all.max())
    H = max(zmax - zmin, 1e-4)
    zn = (z_all - zmin) / H

    def to_world(lc, fc, zc):
        return lc * L_ax + fc * F_ax + zc * U_ax

    # Bilateral center (robust median along Left axis)
    l_c = float(np.median(l_all))

    # Build 32 horizontal Z-slices
    n_sl = 32
    sl_zn = np.linspace(0.03, 0.97, n_sl)
    sl_w = np.zeros(n_sl)
    sl_d = np.zeros(n_sl)
    sl_f_med = np.zeros(n_sl)
    sl_f_hi = np.zeros(n_sl)
    sl_f_lo = np.zeros(n_sl)
    sl_cnt = np.zeros(n_sl, dtype=int)
    for k, zf in enumerate(sl_zn):
        m = (zn >= zf - 0.035) & (zn <= zf + 0.035)
        cnt = int(m.sum())
        sl_cnt[k] = cnt
        if cnt >= 6:
            lk = l_all[m]
            fk = f_all[m]
            sl_w[k] = float(np.quantile(lk, 0.97) - np.quantile(lk, 0.03))
            sl_f_hi[k] = float(np.quantile(fk, 0.96))
            sl_f_lo[k] = float(np.quantile(fk, 0.04))
            sl_d[k] = sl_f_hi[k] - sl_f_lo[k]
            sl_f_med[k] = float(np.median(fk))
        elif k > 0:
            sl_w[k], sl_d[k] = sl_w[k - 1], sl_d[k - 1]
            sl_f_med[k], sl_f_hi[k], sl_f_lo[k] = sl_f_med[k - 1], sl_f_hi[k - 1], sl_f_lo[k - 1]

    mid_mask = (sl_zn >= 0.30) & (sl_zn <= 0.65)
    mid_w = float(np.median(sl_w[mid_mask])) if np.any(mid_mask) else 0.4 * H
    mid_d = float(np.median(sl_d[mid_mask])) if np.any(mid_mask) else 0.3 * H
    is_sd = (mid_w / H) > 0.58  # chubby/mascot vs tall humanoid

    # Anchor body front-back core from feet/lower-legs and upper-head (avoids back hair & tails)
    leg_band = (zn >= 0.05) & (zn <= 0.25)
    f_legs = float(np.median(f_all[leg_band])) if leg_band.sum() >= 10 else float(np.median(f_all))

    # Detect head top (on mascots with tall thin ears, skull ends where depth drops sharply)
    zn_skull_top = 0.96
    if is_sd:
        for k in range(n_sl - 1, int(n_sl * 0.55), -1):
            if sl_d[k] >= 0.45 * mid_d and sl_w[k] >= 0.40 * mid_w:
                zn_skull_top = float(sl_zn[k])
                break
        zn_skull_top = float(np.clip(zn_skull_top, 0.68, 0.94))

    head_band = (zn >= max(0.60, zn_skull_top - 0.18)) & (zn <= zn_skull_top)
    if head_band.sum() >= 10:
        fh = f_all[head_band]
        f_face = float(np.quantile(fh, 0.95))
        f_back = float(np.quantile(fh, 0.05))
        if (f_face - f_back) > 0.32 * H:
            # Back hair or tail behind head: anchor inside skull behind face
            f_head = f_face - min(0.14 * H, 0.36 * (f_face - f_back))
        else:
            f_head = 0.5 * (f_face + f_back)
    else:
        f_head = f_legs

    def f_core_at(z_frac):
        """Front-back body centerline at normalized height z_frac."""
        t = float(np.clip(z_frac / max(zn_skull_top, 0.5), 0.0, 1.0))
        f_lin = (1.0 - t) * f_legs + t * f_head
        m = (zn >= z_frac - 0.04) & (zn <= z_frac + 0.04) & (np.abs(l_all - l_c) <= max(0.18 * H, 0.35 * mid_w))
        if m.sum() >= 8:
            fk = f_all[m]
            flo, fhi = float(np.quantile(fk, 0.05)), float(np.quantile(fk, 0.95))
            # Prefer anchor if slice extends far backward (tail/hair), but clamp inside slice
            f_med = float(np.quantile(fk, 0.58))
            f_est = 0.65 * f_lin + 0.35 * f_med
            margin = 0.03 * H
            return float(np.clip(f_est, flo + margin, max(flo + margin, fhi - margin)))
        return f_lin

    # --- 1. Neck & Head heights ---
    if not is_sd:
        # Look for neck width pinch in [0.70, 0.86]
        neck_idxs = [k for k in range(n_sl) if 0.70 <= sl_zn[k] <= 0.86]
        if neck_idxs:
            best_k = min(neck_idxs, key=lambda k: sl_w[k])
            zn_neck = float(np.clip(sl_zn[best_k], 0.74, 0.84))
        else:
            zn_neck = 0.80
        zn_head = zn_neck + 0.28 * (0.98 - zn_neck)
        zn_crown = 0.96
    else:
        # Chubby / 2-head SD mascot: large head/face occupies upper ~55% of height,
        # stubby torso/arms sit in [0.20..0.40], stubby legs in [0.00..0.20].
        zn_neck = float(np.clip(zn_skull_top - 0.34, 0.38, 0.46))
        zn_head = zn_neck + 0.08
        zn_crown = float(np.clip(zn_skull_top + 0.06, zn_head + 0.14, 0.90))

    # --- 2. Crotch & Legs ---
    zn_crotch = None
    for k in range(n_sl):
        zf = float(sl_zn[k])
        if not (0.08 <= zf <= 0.52):
            continue
        m = (zn >= zf - 0.025) & (zn <= zf + 0.025)
        if m.sum() < 20:
            continue
        lk = l_all[m] - l_c
        left_n = int((lk > 0.025 * H).sum())
        right_n = int((lk < -0.025 * H).sum())
        mid_n = int((np.abs(lk) <= 0.022 * H).sum())
        if left_n >= 8 and right_n >= 8 and mid_n < 0.14 * len(lk):
            zn_crotch = zf

    if zn_crotch is None:
        zn_crotch = 0.16 if is_sd else 0.43
    else:
        zn_crotch = float(np.clip(zn_crotch, 0.12 if is_sd else 0.34, 0.24 if is_sd else 0.50))

    zn_hip = float(np.clip(zn_crotch + 0.04, 0.16, zn_neck - 0.16))
    zn_ankle = 0.05
    zn_foot = 0.02
    zn_knee = 0.5 * (zn_hip + zn_ankle)

    # Measure left/right leg lateral spread inside [0.05, zn_crotch]
    leg_zone = (zn >= 0.05) & (zn <= max(0.14, zn_crotch))
    l_leg_vals = l_all[leg_zone] - l_c
    l_pos = l_leg_vals[l_leg_vals > 0.015 * H]
    l_neg = -l_leg_vals[l_leg_vals < -0.015 * H]
    if len(l_pos) >= 8 and len(l_neg) >= 8:
        leg_half = 0.5 * (float(np.median(l_pos)) + float(np.median(l_neg)))
    else:
        leg_half = 0.14 * H if is_sd else 0.08 * H
    leg_half = float(np.clip(leg_half, 0.045 * H, 0.24 * H))

    def leg_f_at(z_frac, side_sign):
        m = (zn >= z_frac - 0.05) & (zn <= z_frac + 0.05) & ((l_all - l_c) * side_sign > 0.01 * H)
        if m.sum() >= 6:
            fk = f_all[m]
            f_est = 0.5 * f_core_at(z_frac) + 0.5 * float(np.median(fk))
            return float(np.clip(f_est, float(np.quantile(fk, 0.08)), float(np.quantile(fk, 0.92))))
        return f_core_at(z_frac)

    # --- 3. Torso chain ---
    zn_shoulder = float(np.clip(zn_neck - (0.06 if is_sd else 0.04), zn_hip + 0.12, zn_neck - 0.02))
    zn_spine = zn_hip + 0.15 * (zn_shoulder - zn_hip)
    zn_chest = zn_hip + 0.58 * (zn_shoulder - zn_hip)

    p_root_h = to_world(l_c, f_core_at(zn_hip), zmin + zn_hip * H)
    p_spine_h = to_world(l_c, f_core_at(zn_spine), zmin + zn_spine * H)
    p_chest_h = to_world(l_c, f_core_at(zn_chest), zmin + zn_chest * H)
    p_neck_h = to_world(l_c, f_core_at(zn_neck), zmin + zn_neck * H)
    p_head_h = to_world(l_c, f_core_at(zn_head), zmin + zn_head * H)
    p_head_t = to_world(l_c, f_core_at(zn_crown), zmin + zn_crown * H)

    # --- 4. Shoulders & Arms (trace outer left/right silhouette vertices) ---
    sh_band = (zn >= zn_shoulder - 0.05) & (zn <= zn_shoulder + 0.05)
    if sh_band.sum() >= 10:
        sh_half_raw = 0.5 * float(np.quantile(l_all[sh_band], 0.97) - np.quantile(l_all[sh_band], 0.03))
    else:
        sh_half_raw = 0.5 * mid_w

    sh_half = float(np.clip(sh_half_raw * (0.50 if is_sd else 0.58), 0.08 * H, 0.28 * H))
    f_sh = f_core_at(zn_shoulder)
    z_sh = zmin + zn_shoulder * H

    def fit_arm(side_sign):
        """Fit (shoulder_head, upper_head, lower_head, hand_head, hand_tail) for left (+1) / right (-1)."""
        p_sh_h = to_world(l_c + side_sign * (0.22 * sh_half), f_sh, z_sh)
        p_up_h = to_world(l_c + side_sign * sh_half, f_sh, z_sh)

        # Look for arm vertices outside the shoulder joint in the upper-mid vertical band,
        # restricted to the torso's front/side corridor so back-hair or tails aren't grabbed.
        arm_z_lo = max(0.18, zn_hip - (0.04 if is_sd else 0.08))
        arm_z_hi = min(0.92, zn_shoulder + 0.06)
        arm_m = (
            (zn >= arm_z_lo) & (zn <= arm_z_hi)
            & ((l_all - l_c) * side_sign > sh_half * 0.82)
            & (f_all >= f_sh - (0.14 * H if is_sd else 0.18 * H))
        )
        if arm_m.sum() >= 15:
            av = v[arm_m]
            dists = np.linalg.norm(av - p_up_h[None, :], axis=1)
            max_d = float(np.quantile(dists, 0.97))
            if max_d >= 0.11 * H:
                q45, q60 = float(np.quantile(dists, 0.42)), float(np.quantile(dists, 0.58))
                q78, q92 = float(np.quantile(dists, 0.78)), float(np.quantile(dists, 0.92))
                q94 = float(np.quantile(dists, 0.94))
                elb_pts = av[(dists >= q45) & (dists <= q60)]
                wri_pts = av[(dists >= q78) & (dists <= q92)]
                tip_pts = av[dists >= q94]
                p_low_h = elb_pts.mean(axis=0) if len(elb_pts) else (p_up_h + to_world(l_c + side_sign * (sh_half * 1.2), f_sh, z_sh - 0.14 * H)) * 0.5
                p_hnd_h = wri_pts.mean(axis=0) if len(wri_pts) else to_world(l_c + side_sign * (sh_half * 1.25), f_sh, z_sh - 0.28 * H)
                p_hnd_t = tip_pts.mean(axis=0) if len(tip_pts) else p_hnd_h + (p_hnd_h - p_low_h) * 0.35
                # Guarantee minimum segment lengths along chain
                if np.linalg.norm(p_low_h - p_up_h) < 0.04 * H:
                    p_low_h = p_up_h + (p_hnd_h - p_up_h) * 0.5
                if np.linalg.norm(p_hnd_t - p_hnd_h) < 0.03 * H:
                    dir_h = p_hnd_h - p_low_h
                    dn = float(np.linalg.norm(dir_h))
                    p_hnd_t = p_hnd_h + ((dir_h / dn) if dn > 1e-6 else -U_ax) * (0.05 * H)
                return p_sh_h, p_up_h, p_low_h, p_hnd_h, p_hnd_t

        # Fallback: natural relaxed A-pose hugging the outer torso
        arm_drop = 0.24 * H if is_sd else 0.36 * H
        arm_out = 0.08 * H if is_sd else 0.06 * H
        p_low_h = to_world(l_c + side_sign * (sh_half + 0.5 * arm_out), f_sh, z_sh - 0.45 * arm_drop)
        p_hnd_h = to_world(l_c + side_sign * (sh_half + arm_out), f_sh + 0.02 * H, z_sh - 0.85 * arm_drop)
        p_hnd_t = to_world(l_c + side_sign * (sh_half + 1.1 * arm_out), f_sh + 0.04 * H, z_sh - arm_drop)
        return p_sh_h, p_up_h, p_low_h, p_hnd_h, p_hnd_t

    l_sh_h, l_up_h, l_low_h, l_hnd_h, l_hnd_t = fit_arm(+1.0)
    r_sh_h, r_up_h, r_low_h, r_hnd_h, r_hnd_t = fit_arm(-1.0)

    # Legs (head & tail)
    l_th_h = to_world(l_c + leg_half, leg_f_at(zn_hip, +1.0), zmin + zn_hip * H)
    l_shn_h = to_world(l_c + leg_half, leg_f_at(zn_knee, +1.0), zmin + zn_knee * H)
    l_ft_h = to_world(l_c + leg_half, leg_f_at(zn_ankle, +1.0), zmin + zn_ankle * H)
    l_ft_t = l_ft_h + F_ax * (0.09 * H) - U_ax * (0.025 * H)

    r_th_h = to_world(l_c - leg_half, leg_f_at(zn_hip, -1.0), zmin + zn_hip * H)
    r_shn_h = to_world(l_c - leg_half, leg_f_at(zn_knee, -1.0), zmin + zn_knee * H)
    r_ft_h = to_world(l_c - leg_half, leg_f_at(zn_ankle, -1.0), zmin + zn_ankle * H)
    r_ft_t = r_ft_h + F_ax * (0.09 * H) - U_ax * (0.025 * H)

    # Bone geometry table by canonical SweetieRig bone name
    specs = {
        "Root":      (p_root_h,  p_spine_h, F_ax),
        "Spine":     (p_spine_h, p_chest_h, F_ax),
        "Chest":     (p_chest_h, p_neck_h,  F_ax),
        "Neck":      (p_neck_h,  p_head_h,  F_ax),
        "Head":      (p_head_h,  p_head_t,  F_ax),
        "LShoulder": (l_sh_h,    l_up_h,    U_ax),
        "LUpperArm": (l_up_h,    l_low_h,  -F_ax),
        "LLowerArm": (l_low_h,   l_hnd_h,  -F_ax),
        "LHand":     (l_hnd_h,   l_hnd_t,  -F_ax),
        "RShoulder": (r_sh_h,    r_up_h,    U_ax),
        "RUpperArm": (r_up_h,    r_low_h,  -F_ax),
        "RLowerArm": (r_low_h,   r_hnd_h,  -F_ax),
        "RHand":     (r_hnd_h,   r_hnd_t,  -F_ax),
        "LThigh":    (l_th_h,    l_shn_h,  -F_ax),
        "LShin":     (l_shn_h,   l_ft_h,   -F_ax),
        "LFoot":     (l_ft_h,    l_ft_t,   -U_ax),
        "RThigh":    (r_th_h,    r_shn_h,  -F_ax),
        "RShin":     (r_shn_h,   r_ft_h,   -F_ax),
        "RFoot":     (r_ft_h,    r_ft_t,   -U_ax),
    }

    # Fallback bbox fit for any non-standard bone names
    _, fallback_scale, fallback_bw, _ = fit_rig_to_mesh(rig, v.min(0), v.max(0))
    _, fallback_tails = _compute_bone_tails(rig.get("bones"), fallback_bw)

    fitted = []
    fitted_tails = []
    for i, b in enumerate(rig["bones"]):
        bname = b["name"]
        if bname in specs:
            bh, bt, ref = specs[bname]
            fitted.append(_make_bone_matrix(bh, bt, ref))
            fitted_tails.append(bt)
        else:
            fitted.append(fallback_bw[i])
            fitted_tails.append(fallback_tails[i])

    fit_meta = {
        "yaw": float(yaw),
        "is_sd": bool(is_sd),
        "L_ax": L_ax,
        "F_ax": F_ax,
        "l_c": float(l_c),
        "zmin": float(zmin),
        "H": float(H),
        "zn_neck": float(zn_neck),
        "zn_head": float(zn_head),
        "zn_shoulder": float(zn_shoulder),
        "zn_hip": float(zn_hip),
        "sh_half": float(sh_half),
        "f_sh": float(f_sh),
    }
    return float(H), fitted, np.identity(4, dtype=np.float64), np.array(fitted_tails), fit_meta


def refine_weights_anatomical(weights, mesh_verts, bones, fit_meta):
    """Anatomical skin-weight guard pass preventing face squashing and limb bleed.

    Enforces:
      1. Rigid Head/Face zone: vertices above zn_neck cannot receive arm,
         shoulder, or leg weights; above zn_head they blend cleanly to Head
         so eyes, cheeks, mouth, and ears never squash or shear during motion.
      2. Arm isolation: arm bones only influence outer-side vertices below
         zn_neck (never the central face/chest or back tail).
      3. Leg isolation: leg bones only influence lower-body vertices on their
         own left/right side.
    """
    if not isinstance(fit_meta, dict) or "L_ax" not in fit_meta:
        return weights
    v = np.asarray(mesh_verts, dtype=np.float64)
    name_to_idx = {b["name"]: i for i, b in enumerate(bones)}

    head_i = name_to_idx.get("Head")
    neck_i = name_to_idx.get("Neck")
    chest_i = name_to_idx.get("Chest")
    spine_i = name_to_idx.get("Spine")
    root_i = name_to_idx.get("Root")
    if head_i is None or chest_i is None or root_i is None:
        return weights

    l_arm_set = {name_to_idx[n] for n in ("LShoulder", "LUpperArm", "LLowerArm", "LHand") if n in name_to_idx}
    r_arm_set = {name_to_idx[n] for n in ("RShoulder", "RUpperArm", "RLowerArm", "RHand") if n in name_to_idx}
    arm_set = l_arm_set | r_arm_set
    l_leg_set = {name_to_idx[n] for n in ("LThigh", "LShin", "LFoot") if n in name_to_idx}
    r_leg_set = {name_to_idx[n] for n in ("RThigh", "RShin", "RFoot") if n in name_to_idx}
    leg_set = l_leg_set | r_leg_set

    L_ax = fit_meta["L_ax"]
    F_ax = fit_meta["F_ax"]
    l_c = fit_meta["l_c"]
    zmin = fit_meta["zmin"]
    H = fit_meta["H"]
    is_sd = fit_meta["is_sd"]
    zn_neck = fit_meta["zn_neck"]
    zn_head = fit_meta["zn_head"]
    zn_hip = fit_meta["zn_hip"]
    sh_half = fit_meta["sh_half"]
    f_sh = fit_meta["f_sh"]

    l_rel = (v @ L_ax) - l_c
    f_rel = (v @ F_ax) - f_sh
    zn = (v[:, 2] - zmin) / H

    def smoothstep(edge0, edge1, x):
        t = (x - edge0) / max(edge1 - edge0, 1e-6)
        t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
        return t * t * (3.0 - 2.0 * t)

    # Back/tail cutoff: vertices far behind shoulders belong to torso/tail, never arms
    back_cut = -0.14 * H if is_sd else -0.20 * H
    # Lateral cutoff for arm influence
    arm_l_cut = sh_half * (0.72 if is_sd else 0.52)

    refined = []
    for i, (bi_list, wi_list) in enumerate(weights):
        z_i = float(zn[i])
        l_i = float(l_rel[i])
        f_i = float(f_rel[i])

        # --- Region 1: Rigid Head / Face zone ---
        # Smooth transition across [zn_neck - 0.04, zn_head + 0.04] into 100% rigid Head.
        head_start = zn_neck - 0.04
        head_solid = zn_head + (0.04 if is_sd else 0.02)
        if z_i >= head_solid:
            # 100% rigid Head (or on SD characters, back tail at upper height stays Chest/Root)
            if is_sd and f_i < -0.22 * H and z_i < 0.72:
                refined.append(([chest_i, root_i], [0.65, 0.35]))
            else:
                refined.append(([head_i], [1.0]))
            continue

        # Build mutable weight map for vertex i
        wmap = {}
        for b_idx, w in zip(bi_list, wi_list):
            if w > 1e-5:
                wmap[int(b_idx)] = wmap.get(int(b_idx), 0.0) + float(w)

        # Suppress arm weights on head/neck, wrong side, central chest, or back tail
        for b_idx in list(wmap.keys()):
            if b_idx in arm_set:
                side = 1.0 if b_idx in l_arm_set else -1.0
                # Taper off above shoulder/neck
                z_ok = 1.0 - smoothstep(zn_neck - 0.05, zn_neck + 0.02, z_i)
                # Must be on own side and outside central torso core
                l_ok = smoothstep(arm_l_cut * 0.65, arm_l_cut * 1.15, l_i * side)
                # Not far behind back (tail/hair)
                f_ok = smoothstep(back_cut - 0.08 * H, back_cut + 0.04 * H, f_i)
                # Not below hip
                low_ok = smoothstep(zn_hip - 0.12, zn_hip - 0.02, z_i)
                keep = z_ok * l_ok * f_ok * low_ok
                if is_sd:
                    # On SD mascots, keep arm deformation gentle so stubby body stays plump
                    keep *= 0.75
                if keep < 0.02:
                    wmap.pop(b_idx, None)
                else:
                    wmap[b_idx] *= keep
            elif b_idx in leg_set:
                side = 1.0 if b_idx in l_leg_set else -1.0
                # Taper off above hip
                z_ok = 1.0 - smoothstep(zn_hip + 0.01, zn_hip + 0.10, z_i)
                # Own side
                l_ok = smoothstep(-0.04 * H, 0.03 * H, l_i * side)
                # Not far behind back tail
                f_ok = smoothstep(back_cut - 0.12 * H, back_cut + 0.02 * H, f_i)
                keep = z_ok * l_ok * f_ok
                if keep < 0.02:
                    wmap.pop(b_idx, None)
                else:
                    wmap[b_idx] *= keep

        # Blend smoothly into rigid Head across the collar transition band [head_start, head_solid]
        if z_i > head_start and not (is_sd and f_i < -0.22 * H):
            alpha = smoothstep(head_start, head_solid, z_i)
            for k in list(wmap.keys()):
                wmap[k] *= (1.0 - alpha)
            wmap[head_i] = wmap.get(head_i, 0.0) + alpha

        # Fallback if all limb weights were stripped: bind cleanly to nearest torso bone
        if not wmap or sum(wmap.values()) < 1e-4:
            if z_i >= zn_neck:
                wmap = {head_i: 1.0}
            elif z_i >= zn_hip + 0.06:
                wmap = {chest_i: 0.7, (spine_i if spine_i is not None else root_i): 0.3}
            else:
                wmap = {root_i: 1.0}

        # Keep top 4 normalized weights
        pairs = sorted(wmap.items(), key=lambda x: -x[1])[:4]
        s = sum(w for _, w in pairs)
        refined.append(([p[0] for p in pairs], [p[1] / s for p in pairs]))

    return refined


def fit_rig_to_mesh(rig, mesh_min, mesh_max):
    """Uniform-scale + translate the rig so its bone-space bbox matches mesh.

    Rig bbox = bbox of all bone head + tail positions.
    Returns (T, scale, fitted_bind_world, fitted_armature_world).
    """
    heads, tails = _compute_bone_tails(rig.get("bones"), rig["bind_world"])
    pts = np.vstack([heads, tails])
    rmin, rmax = pts.min(axis=0), pts.max(axis=0)
    mmin = np.array(mesh_min, dtype=np.float64)
    mmax = np.array(mesh_max, dtype=np.float64)

    rsize = np.maximum(rmax - rmin, 1e-6)
    msize = np.maximum(mmax - mmin, 1e-6)
    # Humanoid rigs are thin in Y (depth), so median(msize/rsize) overscales
    # height by ~1.5-2x on round/mascot meshes. Fit primarily to Z height
    # (capped by X width) with a 0.90 inset so bones stay inside the mesh.
    scale = float(min(msize[2] / rsize[2], (msize[0] / rsize[0]) * 1.15) * 0.90)

    rc = (rmin + rmax) * 0.5
    mc = (mmin + mmax) * 0.5
    T = np.identity(4)
    T[:3, :3] *= scale
    T[:3, 3] = mc - scale * rc

    fitted = [T @ np.array(bw) for bw in rig["bind_world"]]
    arm = T @ np.array(rig["armature_world"])
    return T, scale, fitted, arm


# ---------------------------------------------------------------------------
# Weight computation: nearest bone segment (head->tail) in fitted space.
# ---------------------------------------------------------------------------

def nearest_bone_weights(verts, bind_world, k=4, bones=None):
    """Assign each vertex to its k nearest bone segments, weight ~ 1/d^2."""
    heads, tails = _compute_bone_tails(bones, bind_world)
    verts = np.asarray(verts, dtype=np.float64)
    n, nb = len(verts), len(heads)
    k = min(k, nb)

    d = np.empty((n, nb), dtype=np.float64)
    for bi in range(nb):
        a, b = heads[bi], tails[bi]
        ab = b - a
        ap = verts - a
        t = np.clip((ap @ ab) / (ab @ ab + 1e-12), 0.0, 1.0)
        c = a + np.outer(t, ab)
        d[:, bi] = ((verts - c) ** 2).sum(axis=1)

    order = np.argsort(d, axis=1)[:, :k]
    w = 1.0 / (d[np.arange(n)[:, None], order] + 1e-9)
    w /= w.sum(axis=1, keepdims=True)

    out = []
    for i in range(n):
        bi = [int(order[i, j]) for j in range(k) if w[i, j] > 1e-4]
        wi = [float(w[i, j]) for j in range(k) if w[i, j] > 1e-4]
        if not bi:
            bi, wi = [int(order[i, 0])], [1.0]
        else:
            # Renormalize after dropping small weights so they still sum to 1.
            # Without this, filtered weights sum < 1 and the vertex is only
            # partially deformed by its selected bones (tearing in posed skin).
            s = sum(wi)
            wi = [x / s for x in wi]
        out.append((bi, wi))
    return out


_BLENDER_AUTO_WEIGHTS_SCRIPT = r"""
import json, os, sys
import bpy

args = sys.argv[sys.argv.index("--") + 1:]
in_json, out_json = args[0], args[1]

with open(in_json, "r", encoding="utf-8") as fh:
    data = json.load(fh)

verts = data["verts"]
faces = data["faces"]
bones = data["bones"]

bpy.ops.wm.read_factory_settings(use_empty=True)

mesh = bpy.data.meshes.new("RigMesh")
mesh.from_pydata(verts, [], faces)
mesh.update()
mesh_obj = bpy.data.objects.new("RigMesh", mesh)
bpy.context.collection.objects.link(mesh_obj)

arm = bpy.data.armatures.new("RigArm")
arm_obj = bpy.data.objects.new("RigArm", arm)
bpy.context.collection.objects.link(arm_obj)
bpy.context.view_layer.objects.active = arm_obj
arm_obj.select_set(True)

bpy.ops.object.mode_set(mode="EDIT")
eb = arm.edit_bones
created = []
for b in bones:
    e = eb.new(b["name"])
    e.head = tuple(b["head"])
    e.tail = tuple(b["tail"])
    if (e.tail - e.head).length < 1e-5:
        e.tail = (e.head[0], e.head[1], e.head[2] + 0.05)
    created.append(e)
for i, b in enumerate(bones):
    p = b.get("parent")
    if p is not None and 0 <= p < len(created):
        created[i].parent = created[p]
bpy.ops.object.mode_set(mode="OBJECT")

bpy.ops.object.select_all(action="DESELECT")
mesh_obj.select_set(True)
arm_obj.select_set(True)
bpy.context.view_layer.objects.active = arm_obj
bpy.ops.object.parent_set(type="ARMATURE_AUTO")

name_to_idx = {b["name"]: i for i, b in enumerate(bones)}
vg_to_bone = {vg.index: name_to_idx[vg.name]
              for vg in mesh_obj.vertex_groups if vg.name in name_to_idx}

out_weights = []
for v in mesh.vertices:
    pairs = []
    for g in v.groups:
        bi = vg_to_bone.get(g.group)
        if bi is not None and g.weight > 1e-4:
            pairs.append((bi, float(g.weight)))
    pairs.sort(key=lambda x: -x[1])
    pairs = pairs[:4]
    if pairs:
        s = sum(w for _, w in pairs)
        out_weights.append(([p[0] for p in pairs], [p[1] / s for p in pairs]))
    else:
        out_weights.append(([], []))

with open(out_json, "w", encoding="utf-8") as fh:
    json.dump(out_weights, fh)
"""


def blender_auto_weights(mesh_verts, mesh_faces, bones, fitted_bind_world,
                         fallback_k=4, timeout=120, explicit_tails=None):
    """Compute Heat-Diffusion automatic weights using Blender ARMATURE_AUTO.

    Returns [(bone_indices, weights), ...] or None if Blender is unavailable/fails.
    Any unweighted vertices (e.g. disconnected floaters) fall back to nearest_bone_weights.
    """
    import os
    import subprocess
    import tempfile

    blender = fbx_writer.find_blender()
    if not blender:
        return None

    heads, tails = _compute_bone_tails(bones, fitted_bind_world)
    if explicit_tails is not None and len(explicit_tails) == len(heads):
        tails = np.asarray(explicit_tails, dtype=np.float64)
    bone_spec = [
        {
            "name": b["name"],
            "parent": b.get("parent"),
            "head": [float(x) for x in heads[i]],
            "tail": [float(x) for x in tails[i]],
        }
        for i, b in enumerate(bones)
    ]
    payload = {
        "verts": np.asarray(mesh_verts, dtype=np.float64).tolist(),
        "faces": np.asarray(mesh_faces, dtype=np.int64).tolist(),
        "bones": bone_spec,
    }

    with tempfile.TemporaryDirectory(prefix="idol_autow_") as tmpdir:
        in_json = os.path.join(tmpdir, "in.json")
        out_json = os.path.join(tmpdir, "out.json")
        script_py = os.path.join(tmpdir, "auto_weights.py")
        with open(in_json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        with open(script_py, "w", encoding="utf-8") as fh:
            fh.write(_BLENDER_AUTO_WEIGHTS_SCRIPT)

        try:
            proc = subprocess.run(
                [blender, "--background", "--factory-startup",
                 "--python", script_py, "--", in_json, out_json],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
            )
            if proc.returncode != 0 or not os.path.exists(out_json):
                return None
            with open(out_json, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
        except Exception:
            return None

    if len(raw) != len(mesh_verts):
        return None

    empty_idx = [i for i, (bi, _) in enumerate(raw) if not bi]
    if len(empty_idx) == len(raw):
        return None
    if empty_idx:
        fb = nearest_bone_weights(
            np.asarray(mesh_verts)[empty_idx], fitted_bind_world,
            k=fallback_k, bones=bones,
        )
        for idx, val in zip(empty_idx, fb):
            raw[idx] = val
    return [(list(bi), list(wi)) for bi, wi in raw]


# ---------------------------------------------------------------------------
# Skinned GLB writer (web canonical format, Z-up, no axis swap).
# ---------------------------------------------------------------------------

def _compute_normals(verts, faces):
    normals = np.zeros((len(verts), 3), dtype=np.float32)
    for a, b, c in faces:
        nrm = np.cross(verts[b] - verts[a], verts[c] - verts[a])
        normals[a] += nrm
        normals[b] += nrm
        normals[c] += nrm
    nlen = np.linalg.norm(normals, axis=1, keepdims=True)
    nlen[nlen == 0] = 1.0
    return (normals / nlen).astype(np.float32)


def write_skinned_glb(path, verts, faces, vcolors, skin, name="mesh"):
    """Write a skinned GLB (glTF 2.0 binary) in Z-up metric space.

    skin: {"bones":[{name,parent}], "bind_world":[4x4], "armature_world":4x4,
           "weights":[(bone_idx_list, weight_list) per vertex]}
    """
    V, F = len(verts), len(faces)
    verts = np.asarray(verts, dtype=np.float32).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    vcolors = np.asarray(vcolors, dtype=np.float32).reshape(-1, 3)
    normals = _compute_normals(verts, faces)

    joints_arr = np.zeros((V, 4), dtype=np.uint16)
    weights_arr = np.zeros((V, 4), dtype=np.float32)
    for i, (bi, wi) in enumerate(skin["weights"]):
        for j in range(min(4, len(bi))):
            joints_arr[i, j] = bi[j]
            weights_arr[i, j] = wi[j]

    bind_world = [np.array(m, dtype=np.float64) for m in skin["bind_world"]]
    nb = len(bind_world)
    invs = [np.linalg.inv(m).astype(np.float32) for m in bind_world]

    blobs = {
        "pos": verts.tobytes(),
        "nrm": normals.tobytes(),
        "col": vcolors.tobytes(),
        "idx": faces.tobytes(),
        "jnt": joints_arr.tobytes(),
        "wgt": weights_arr.tobytes(),
    }
    body = b""
    offsets = {}
    for key, b in blobs.items():
        offsets[key] = len(body)
        body += b
        body += b"\0" * ((4 - len(b) % 4) % 4)  # 4-byte align each view
    offsets["inv"] = len(body)
    for m in invs:
        # glTF stores MAT4 in column-major bytes; numpy arrays are row-major.
        body += m.T.tobytes()

    buffer_views = []

    def add_view(off, ln):
        buffer_views.append({"buffer": 0, "byteOffset": off, "byteLength": ln})
        return len(buffer_views) - 1

    accessors = []

    def add_acc(bv, ctype, typ, count, target=None, extra=None):
        a = {"bufferView": bv, "componentType": ctype, "count": count,
             "type": typ}
        if target is not None:
            a["target"] = target
        if extra:
            a.update(extra)
        accessors.append(a)
        return len(accessors) - 1

    acc = {
        "POSITION": add_acc(add_view(offsets["pos"], len(blobs["pos"])),
                            T_FLOAT, "VEC3", V, 34962,
                            {"min": verts.min(0).tolist(),
                             "max": verts.max(0).tolist()}),
        "NORMAL": add_acc(add_view(offsets["nrm"], len(blobs["nrm"])),
                          T_FLOAT, "VEC3", V, 34962),
        "COLOR_0": add_acc(add_view(offsets["col"], len(blobs["col"])),
                           T_FLOAT, "VEC3", V, 34962),
        "JOINTS_0": add_acc(add_view(offsets["jnt"], len(blobs["jnt"])),
                            T_USHORT, "VEC4", V, 34962),
        "WEIGHT_0": add_acc(add_view(offsets["wgt"], len(blobs["wgt"])),
                            T_FLOAT, "VEC4", V, 34962),
        "indices": add_acc(add_view(offsets["idx"], len(blobs["idx"])),
                           T_UINT, "SCALAR", F * 3, 34963),
    }
    inv_accs = []
    for i in range(nb):
        inv_accs.append(add_acc(add_view(offsets["inv"] + i * 64, 64),
                                T_FLOAT, "MAT4", 1))

    # Nodes: bones first (0..nb-1), then armature, then skinned mesh.
    nodes = []
    for bi in range(nb):
        if skin["bones"][bi].get("parent") is None:
            mat = bind_world[bi]
        else:
            p = skin["bones"][bi]["parent"]
            mat = np.linalg.inv(bind_world[p]) @ bind_world[bi]
        t, q, s = _decompose(mat)
        nodes.append({"name": skin["bones"][bi]["name"],
                      "translation": t, "rotation": q, "scale": s})

    arm_node = nb
    mesh_node = nb + 1
    top_bones = [bi for bi in range(nb)
                 if skin["bones"][bi].get("parent") is None]
    nodes.append({"name": name + "_Armature",
                  "children": top_bones + [mesh_node]})
    nodes.append({"name": name, "mesh": 0, "skin": 0})

    # Attach bone children
    for bi in range(nb):
        p = skin["bones"][bi].get("parent")
        if p is not None:
            nodes[p].setdefault("children", []).append(bi)

    asset_obj = {"version": "2.0", "generator": "rig_bind.py"}
    extras = {}
    if skin.get("facing_yaw") is not None:
        extras["facing_yaw"] = float(skin["facing_yaw"])
    if skin.get("is_sd") is not None:
        extras["is_sd"] = bool(skin["is_sd"])
    if extras:
        asset_obj["extras"] = extras
    gltf = {
        "asset": asset_obj,
        "scene": 0,
        "scenes": [{"nodes": [arm_node]}],
        "nodes": nodes,
        "meshes": [{
            "name": name,
            "primitives": [{
                "attributes": {
                    "POSITION": acc["POSITION"], "NORMAL": acc["NORMAL"],
                    "COLOR_0": acc["COLOR_0"], "JOINTS_0": acc["JOINTS_0"],
                    "WEIGHT_0": acc["WEIGHT_0"],
                },
                "indices": acc["indices"],
                "mode": 4,
                "material": 0,
            }],
        }],
        "skins": [{"joints": list(range(nb)), "inverseBindingMatrices": inv_accs}],
        "materials": [{"name": name, "pbrMetallicRoughness": {
            "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
            "metallicFactor": 0.0, "roughnessFactor": 0.9}}],
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(body)}],
    }

    json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    json_bytes += b" " * ((4 - len(json_bytes) % 4) % 4)
    body += b"\0" * ((4 - len(body) % 4) % 4)

    total = 12 + 8 + len(json_bytes) + 8 + len(body)
    out = (b"glTF" + struct.pack("<II", 2, total)
           + struct.pack("<I4s", len(json_bytes), b"JSON") + json_bytes
           + struct.pack("<I4s", len(body), b"BIN") + body)
    with open(path, "wb") as fh:
        fh.write(out)
    return {"bytes": len(out), "verts": V, "faces": F, "bones": nb}


# ---------------------------------------------------------------------------
# Skinned FBX writer wrapper (delegates to fbx_writer.write_fbx_skinned).
# ---------------------------------------------------------------------------

def write_skinned_fbx(path, verts, faces, skin, name="mesh"):
    """Write skinned FBX in the mesh's native Z-up metric space (identity
    Model transform for the mesh, LclTranslation/Rotation = fitted bind).
    """
    bones = [{"name": b["name"], "parent": b.get("parent")}
             for b in skin["bones"]]
    clusters = {}
    for i, (bi, wi) in enumerate(skin["weights"]):
        for j in range(min(4, len(bi))):
            c = clusters.setdefault(bi[j], {
                "bone": skin["bones"][bi[j]]["name"],
                "indices": [], "weights": []})
            c["indices"].append(i)
            c["weights"].append(float(wi[j]))
    skin_dict = {
        "bones": bones,
        "bind_world": [np.array(m, dtype=np.float64) for m in skin["bind_world"]],
        # Keep the armature Model node identity (Z-up metric scene space),
        # matching the GLB writer: the root bone carries the full fitted
        # bind_world, so the skinned mesh is NOT scaled/translated by an
        # inherited armature transform (Blender auto-parents the mesh to the
        # armature, which would otherwise shrink it by the cm->m fit scale).
        "armature_world": np.identity(4, dtype=np.float64),
        "clusters": [clusters[k] for k in sorted(clusters)],
    }
    return fbx_writer.write_fbx_skinned(
        path, verts.astype(np.float64), faces.astype(np.int64),
        skin_dict, name=name)


# ---------------------------------------------------------------------------
# High-level bind
# ---------------------------------------------------------------------------

def _apply_custom_joints(bones, fitted, fitted_tails, fit_meta, custom_joints):
    """Override bone head positions with user-edited world positions and rebuild
    orthonormal bind_world matrices + updated fit_meta landmark heights."""
    nb = len(bones)
    orig_heads = np.array([np.asarray(m, dtype=np.float64)[:3, 3] for m in fitted])
    orig_tails = np.asarray(fitted_tails, dtype=np.float64).copy()
    new_heads = orig_heads.copy()

    for i, b in enumerate(bones):
        pos = custom_joints.get(b["name"])
        if pos is not None and len(pos) == 3:
            new_heads[i] = [float(pos[0]), float(pos[1]), float(pos[2])]

    children = [[] for _ in range(nb)]
    for i, b in enumerate(bones):
        p = b.get("parent")
        if p is not None and 0 <= p < nb:
            children[p].append(i)

    new_tails = orig_tails.copy()
    for i in range(nb):
        if children[i]:
            # Axial torso bones should point to their axial child rather than limb shoulders/thighs
            bname = bones[i]["name"]
            if bname == "Root":
                sp = next((c for c in children[i] if bones[c]["name"] == "Spine"), children[i][0])
                new_tails[i] = new_heads[sp]
            elif bname == "Chest":
                nk = next((c for c in children[i] if bones[c]["name"] == "Neck"), children[i][0])
                new_tails[i] = new_heads[nk]
            else:
                new_tails[i] = new_heads[children[i]].mean(axis=0)
        else:
            # Leaf bone: preserve original head->tail offset vector
            offset = orig_tails[i] - orig_heads[i]
            new_tails[i] = new_heads[i] + offset

    F_ax = fit_meta["F_ax"] if isinstance(fit_meta, dict) else np.array([0.0, -1.0, 0.0])
    U_ax = np.array([0.0, 0.0, 1.0])
    new_fitted = []
    for i, b in enumerate(bones):
        bname = b["name"]
        if bname in ("Root", "Spine", "Chest", "Neck", "Head"):
            ref = F_ax
        elif "Shoulder" in bname:
            ref = U_ax
        elif "Foot" in bname:
            ref = -U_ax
        else:
            ref = -F_ax
        new_fitted.append(_make_bone_matrix(new_heads[i], new_tails[i], ref))

    # Sync fit_meta landmarks to user's edited positions so weight guards align
    if isinstance(fit_meta, dict) and "L_ax" in fit_meta:
        name_i = {b["name"]: i for i, b in enumerate(bones)}
        zmin, H = fit_meta["zmin"], max(fit_meta["H"], 1e-4)
        L_ax = fit_meta["L_ax"]
        l_c = fit_meta["l_c"]
        if "Neck" in name_i:
            fit_meta["zn_neck"] = float(np.clip((new_heads[name_i["Neck"]][2] - zmin) / H, 0.15, 0.92))
        if "Head" in name_i:
            fit_meta["zn_head"] = float(np.clip((new_heads[name_i["Head"]][2] - zmin) / H, fit_meta["zn_neck"] + 0.02, 0.95))
        if "Root" in name_i:
            fit_meta["zn_hip"] = float(np.clip((new_heads[name_i["Root"]][2] - zmin) / H, 0.08, fit_meta["zn_neck"] - 0.08))
        if "LUpperArm" in name_i and "RUpperArm" in name_i:
            l_sh = abs(float(new_heads[name_i["LUpperArm"]] @ L_ax) - l_c)
            r_sh = abs(float(new_heads[name_i["RUpperArm"]] @ L_ax) - l_c)
            fit_meta["sh_half"] = max(0.05 * H, 0.5 * (l_sh + r_sh))
            fit_meta["f_sh"] = 0.5 * (float(new_heads[name_i["LUpperArm"]] @ F_ax) + float(new_heads[name_i["RUpperArm"]] @ F_ax))

    return new_fitted, new_tails


def bind_rig_to_mesh(rig_glb_path, mesh_verts, mesh_faces, mesh_vcolors,
                     mesh_min, mesh_max, out_glb=None, out_fbx=None,
                     mesh_name="mesh", weight_k=4, use_blender_weights=True,
                     meta_yaw=None, custom_joints=None):
    """Full bind: load rig, anatomically fit to mesh (with optional user-edited
    custom_joints), compute skin weights (Blender ARMATURE_AUTO heat diffusion
    with anatomical guard), write skinned GLB + FBX."""
    rig = load_rig_glb(rig_glb_path)
    scale, fitted, arm, fitted_tails, fit_meta = fit_rig_anatomical(
        rig, mesh_verts, meta_yaw=meta_yaw
    )
    fit_mode = "anatomical-slice-fit"
    if custom_joints:
        fitted, fitted_tails = _apply_custom_joints(
            rig["bones"], fitted, fitted_tails, fit_meta, custom_joints
        )
        fit_mode = "custom-edited-joints"
    facing_yaw = fit_meta["yaw"] if isinstance(fit_meta, dict) else float(fit_meta)
    is_sd = bool(fit_meta.get("is_sd", False)) if isinstance(fit_meta, dict) else False
    weights = None
    weight_method = "nearest-bone-segment(k=%d)" % weight_k
    if use_blender_weights:
        weights = blender_auto_weights(
            mesh_verts, mesh_faces, rig["bones"], fitted,
            fallback_k=weight_k, explicit_tails=fitted_tails,
        )
        if weights is not None:
            weight_method = "blender-auto-heat+anatomical-guard"
    if weights is None:
        weights = nearest_bone_weights(mesh_verts, fitted, k=weight_k, bones=rig["bones"])

    if isinstance(fit_meta, dict):
        weights = refine_weights_anatomical(weights, mesh_verts, rig["bones"], fit_meta)

    skin = {
        "bones": rig["bones"],
        "bind_world": fitted,
        "armature_world": arm,
        "weights": weights,
        "facing_yaw": facing_yaw,
        "is_sd": is_sd,
    }

    # Cluster statistics for metadata
    usage = {}
    for bi, wi in weights:
        idxs = bi if isinstance(bi, (list, tuple)) else [bi]
        for j in idxs:
            usage[j] = usage.get(j, 0) + 1
    bone_usage = {rig["bones"][i]["name"]: n for i, n in sorted(usage.items())}

    info = {
        "rig_glb": rig_glb_path,
        "bones": len(rig["bones"]),
        "bone_names": [b["name"] for b in rig["bones"]],
        "weights": weight_method,
        "bind_scale": round(scale, 5),
        "facing_yaw": round(float(facing_yaw), 5),
        "bind_fit": fit_mode,
        "bone_vertex_usage": bone_usage,
    }
    if out_glb:
        info["skinned_glb"] = out_glb
        info["glb_bytes"] = write_skinned_glb(
            out_glb, mesh_verts, mesh_faces, mesh_vcolors, skin,
            name=mesh_name)["bytes"]
    if out_fbx:
        info["skinned_fbx"] = out_fbx
        res = write_skinned_fbx(out_fbx, mesh_verts, mesh_faces, skin,
                                name=mesh_name)
        info["skinned_fbx_bytes"] = int(res.get("bytes", 0))
    return info


def main():
    import argparse
    import os
    import plyfile
    from . import config

    ap = argparse.ArgumentParser(description="Auto-rig & skin a reconstructed mesh PLY")
    ap.add_argument("--mesh_ply", required=True, help="input *_mesh.ply")
    ap.add_argument("--rig", default=str(config.DEFAULT_RIG_GLB), help="input rig GLB")
    ap.add_argument("--out_glb", required=True, help="output *_skinned.glb")
    ap.add_argument("--out_fbx", default=None, help="optional output *_skinned.fbx")
    ap.add_argument("--mesh_name", default=None, help="mesh object name")
    ap.add_argument("--meta_json", default=None, help="optional *_mesh.json to update with rig metadata")
    ap.add_argument("--src_json", default=None, help="optional source *.json containing camera yaw")
    ap.add_argument("--joints_json", default=None, help="optional JSON string or path for custom joint edits")
    ap.add_argument("--no_blender_weights", action="store_true", help="skip Blender heat diffusion weights")
    args = ap.parse_args()

    ply = plyfile.PlyData.read(args.mesh_ply)
    el = ply["vertex"]
    verts = np.stack([el["x"], el["y"], el["z"]], axis=1).astype(np.float32)
    _vnames = set(getattr(el, "names", None) or [p.name for p in el.properties])
    if "red" in _vnames:
        vcolors = np.stack([el["red"], el["green"], el["blue"]], axis=1).astype(np.float32) / 255.0
    else:
        vcolors = np.full((len(verts), 3), 0.7, dtype=np.float32)
    faces = np.vstack(ply["face"]["vertex_indices"]).astype(np.int32)

    meta_yaw = None
    if args.src_json and os.path.exists(args.src_json):
        try:
            with open(args.src_json, "r", encoding="utf-8") as sf:
                cam = json.load(sf).get("camera") or {}
            if "yaw" in cam:
                meta_yaw = float(cam["yaw"])
        except Exception:
            pass

    custom_joints = None
    if args.joints_json:
        if os.path.isfile(args.joints_json):
            with open(args.joints_json, "r", encoding="utf-8") as jf:
                custom_joints = json.load(jf)
        else:
            custom_joints = json.loads(args.joints_json)

    mesh_name = args.mesh_name or os.path.splitext(os.path.basename(args.mesh_ply))[0]
    info = bind_rig_to_mesh(
        args.rig, verts, faces, vcolors,
        verts.min(axis=0).tolist(), verts.max(axis=0).tolist(),
        out_glb=args.out_glb, out_fbx=args.out_fbx,
        mesh_name=mesh_name,
        use_blender_weights=not args.no_blender_weights,
        meta_yaw=meta_yaw,
        custom_joints=custom_joints,
    )

    if args.meta_json and os.path.exists(args.meta_json):
        with open(args.meta_json, "r", encoding="utf-8") as mf:
            meta = json.load(mf)
        meta["rig"] = info
        with open(args.meta_json, "w", encoding="utf-8") as mf:
            json.dump(meta, mf, ensure_ascii=False, indent=2)

    print(json.dumps(info, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
