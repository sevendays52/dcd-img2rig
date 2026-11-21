"""Binary FBX 7.4 writer & Blender-verified exporter for static triangle meshes.

Why Binary FBX 7.4 instead of ASCII:
  Blender (4.x / 5.x) refuses to import ASCII FBX files
  (``RuntimeError: Error: ASCII FBX files are not supported``), and ASCII FBX
  files are also larger and prone to schema quirks. Binary FBX 7.4
  (``Kaydara FBX Binary`` version 7400) is the standard interchange format
  opened identically by Blender, Unity, and Unreal.

Two backends are provided behind ``write_fbx``:
  1. ``blender`` (preferred when ``blender`` is installed): exports via
     Blender's official FBX 7.4 Binary exporter and immediately round-trip
     imports the written ``.fbx`` to verify geometry, face count, vertex
     colors, and bounds.
  2. ``binary`` (pure-Python fallback, stdlib + zlib only): writes a valid
     FBX 7.4 Binary container directly with ``Vertices``,
     ``PolygonVertexIndex`` (last index per triangle bit-inverted ``~idx``),
     ``LayerElementNormal``, ``LayerElementColor`` (RGBA), and ``Connections``.
"""
import math
import os
import shutil
import struct
import subprocess
import tempfile
import time
import zlib

import numpy as np


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _compute_normals(verts, faces):
    """Per-vertex smooth normals (accumulated area-weighted face normals)."""
    v = np.asarray(verts, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    n = np.zeros_like(v)
    if len(f) > 0:
        v0 = v[f[:, 0]]
        v1 = v[f[:, 1]]
        v2 = v[f[:, 2]]
        fn = np.cross(v1 - v0, v2 - v0)
        for col in range(3):
            np.add.at(n[:, col], f[:, 0], fn[:, col])
            np.add.at(n[:, col], f[:, 1], fn[:, col])
            np.add.at(n[:, col], f[:, 2], fn[:, col])
    lengths = np.linalg.norm(n, axis=1, keepdims=True)
    safe = lengths > 1e-12
    n[safe[:, 0]] /= lengths[safe[:, 0]]
    n[~safe[:, 0]] = [0.0, 0.0, 1.0]
    return n


from .config import find_blender


# ---------------------------------------------------------------------------
# Pure-Python Binary FBX 7.4 writer
# ---------------------------------------------------------------------------

_FBX_MAGIC = b"Kaydara FBX Binary  \x00\x1a\x00"
_FBX_VERSION = 7400
_NULL_SENTINEL = b"\x00" * 13
_FOOT_ID = bytes.fromhex("fa bc ab 09 d0 c8 d4 66 b1 76 fb 83 1c f7 26 7e")


def _prop_str(s):
    b = s.encode("utf-8") if isinstance(s, str) else bytes(s)
    return b"S" + struct.pack("<I", len(b)) + b


def _prop_raw(b):
    return b"R" + struct.pack("<I", len(b)) + b


def _prop_i32(v):
    return b"I" + struct.pack("<i", int(v))


def _prop_i64(v):
    return b"L" + struct.pack("<q", int(v))


def _prop_f64(v):
    return b"D" + struct.pack("<d", float(v))


def _prop_char(v):
    return b"C" + struct.pack("<B", int(v) & 0xFF)


def _prop_array(type_code, fmt_char, values):
    arr = np.asarray(values, dtype={"d": "<f8", "i": "<i4"}[fmt_char])
    raw = arr.tobytes()
    comp = zlib.compress(raw, level=6)
    # Use compressed encoding (1) when smaller, else raw (0)
    if len(comp) < len(raw):
        return type_code.encode("ascii") + struct.pack("<III", len(arr), 1, len(comp)) + comp
    return type_code.encode("ascii") + struct.pack("<III", len(arr), 0, len(raw)) + raw


def _node(name, props=None, children=None, force_block=False):
    return (name, props or [], children or [], force_block)


# ---------------------------------------------------------------------------
# Skeleton (skin) helpers
# ---------------------------------------------------------------------------

def _mat_flat(m):
    """16-element float list from a 4x4 np.matrix/array, row-major, col-vector
    convention (translation in the 4th column)."""
    return [float(m[r, c]) for r in range(4) for c in range(4)]


def _trs_from_matrix(m):
    """Decompose a 4x4 matrix into (translation, euler XYZ degrees, scale) using
    the Blender/FBX Lcl convention (R = Rz*Ry*Rx)."""
    m = np.array(m, dtype=np.float64).reshape(4, 4)
    t = [float(m[0, 3]), float(m[1, 3]), float(m[2, 3])]
    lin = m[0:3, 0:3]
    sx = float(np.linalg.norm(lin[:, 0]))
    sy = float(np.linalg.norm(lin[:, 1]))
    sz = float(np.linalg.norm(lin[:, 2]))
    if min(sx, sy, sz) < 1e-9:
        sx = sy = sz = 1.0
    s = [sx, sy, sz]
    r = np.column_stack([lin[:, 0] / sx, lin[:, 1] / sy, lin[:, 2] / sz])
    cy = float(np.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2))
    if cy > 1e-6:
        rx = np.arctan2(r[2, 1], r[2, 2])
        ry = np.arctan2(-r[2, 0], cy)
        rz = np.arctan2(r[1, 0], r[0, 0])
    else:
        rx = np.arctan2(-r[1, 2], r[1, 1])
        ry = np.arctan2(-r[2, 0], cy)
        rz = 0.0
    deg = 180.0 / np.pi
    rot = [float(rx * deg), float(ry * deg), float(rz * deg)]
    return t, rot, s


def _node_model(model_id, name, model_type, props):
    return _node("Model", [_prop_i64(model_id), _prop_str(f"{name}\x00\x01Model"),
                           _prop_str(model_type)], children=[
        _node("Version", [_prop_i32(232)]),
        _node("Properties70", children=props),
        _node("MultiLayer", [_prop_i32(0)]),
        _node("MultiTake", [_prop_i32(0)]),
        _node("Shading", [_prop_char(1)]),
        _node("Culling", [_prop_str("CullingOff")]),
    ])


def _bone_model(model_id, name, lcl_translation, lcl_rotation, lcl_scaling=None):
    props = [
        _p70_vec("Lcl Translation", "Lcl Translation", "", "A", lcl_translation),
        _p70_vec("Lcl Rotation", "Lcl Rotation", "", "A", lcl_rotation),
    ]
    if lcl_scaling is not None:
        props.append(_p70_vec("Lcl Scaling", "Lcl Scaling", "", "A", lcl_scaling))
    props.append(_p70_int("DefaultAttributeIndex", 0))
    props.append(_p70_int("InheritType", 1, "enum", ""))
    return _node_model(model_id, name, "LimbNode", props)


def _node_attribute(id_, name):
    return _node("NodeAttribute",
                 [_prop_i64(id_), _prop_str(f"{name}\x00\x01NodeAttribute"),
                  _prop_str("LimbNode")], children=[
        _node("TypeFlags", [_prop_str("Skeleton")]),
        _node("Properties70", children=[
            _p70_double("Size", 3.3),
        ]),
    ])


def _bind_pose(pose_id, name, nodes):
    """nodes: list of (node_id, world_matrix_4x4) in the scene (FBX) space."""
    return _node("Pose",
                 [_prop_i64(pose_id), _prop_str(f"{name}\x00\x01Pose"), _prop_str("BindPose")],
                 children=[
        _node("Type", [_prop_str("BindPose")]),
        _node("Version", [_prop_i32(100)]),
        _node("NbPoseNodes", [_prop_i32(len(nodes))]),
        *[_node("PoseNode", children=[
            _node("Node", [_prop_i64(nid)]),
            _node("Matrix", [_prop_array("d", "d", np.array(_mat_flat(m)))]),
        ]) for nid, m in nodes],
    ])


def _skin_deformer(skin_id, name):
    return _node("Deformer",
                 [_prop_i64(skin_id), _prop_str(f"{name}\x00\x01Deformer"), _prop_str("Skin")],
                 children=[
        _node("Version", [_prop_i32(101)]),
        _node("Link_DeformAcuracy", [_prop_f64(50.0)]),
    ])


def _cluster_deformer(cluster_id, bone_name, indices, weights,
                      transform, transform_link, transform_associate):
    idx = np.asarray(indices, dtype=np.int32)
    wt = np.asarray(weights, dtype=np.float64)
    return _node("Deformer",
                 [_prop_i64(cluster_id), _prop_str(f"{bone_name}\x00\x01SubDeformer"),
                  _prop_str("Cluster")],
                 children=[
        _node("Version", [_prop_i32(100)]),
        _node("UserData", [_prop_str(""), _prop_str("")]),
        _node("Indexes", [_prop_array("i", "i", idx)]),
        _node("Weights", [_prop_array("d", "d", wt)]),
        _node("Transform", [_prop_array("d", "d", np.array(_mat_flat(transform)))]),
        _node("TransformLink", [_prop_array("d", "d", np.array(_mat_flat(transform_link)))]),
        _node("TransformAssociateModel", [_prop_array("d", "d", np.array(_mat_flat(transform_associate)))]),
    ])


def _skinned_geometry(geom_id, name, v, f, normals):
    """Geometry node (mirrors write_fbx_binary). No LayerElementSkin: the Blender
    FBX exporter wires skinning purely via Deformer/Skin + Connections."""
    poly_idx = f.astype(np.int32).copy()
    if len(f) > 0:
        poly_idx[:, 2] = ~poly_idx[:, 2]
    v_flat = v.reshape(-1)
    poly_flat = poly_idx.reshape(-1)

    geom_children = [
        _node("Properties70", force_block=True),
        _node("GeometryVersion", [_prop_i32(124)]),
        _node("Vertices", [_prop_array("d", "d", v_flat)]),
        _node("PolygonVertexIndex", [_prop_array("i", "i", poly_flat)]),
    ]
    layer_elements = []
    if normals is not None:
        geom_children.append(_node("LayerElementNormal", [_prop_i32(0)], children=[
            _node("Version", [_prop_i32(101)]),
            _node("Name", [_prop_str("")]),
            _node("MappingInformationType", [_prop_str("ByVertice")]),
            _node("ReferenceInformationType", [_prop_str("Direct")]),
            _node("Normals", [_prop_array("d", "d", np.asarray(normals).reshape(-1))]),
        ]))
        layer_elements.append(_node("LayerElement", children=[
            _node("Type", [_prop_str("LayerElementNormal")]),
            _node("TypedIndex", [_prop_i32(0)]),
        ]))
    geom_children.append(_node("LayerElementMaterial", [_prop_i32(0)], children=[
        _node("Version", [_prop_i32(101)]),
        _node("Name", [_prop_str("")]),
        _node("MappingInformationType", [_prop_str("AllSame")]),
        _node("ReferenceInformationType", [_prop_str("IndexToDirect")]),
        _node("Materials", [_prop_array("i", "i", [0])]),
    ]))
    layer_elements.append(_node("LayerElement", children=[
        _node("Type", [_prop_str("LayerElementMaterial")]),
        _node("TypedIndex", [_prop_i32(0)]),
    ]))
    geom_children.append(_node("Layer", [_prop_i32(0)], children=[
        _node("Version", [_prop_i32(100)]),
        *layer_elements,
    ]))
    return _node("Geometry",
                 [_prop_i64(geom_id), _prop_str(f"{name}\x00\x01Geometry"), _prop_str("Mesh")],
                 children=geom_children)


def _material_node(mat_id, name):
    return _node("Material",
                 [_prop_i64(mat_id), _prop_str(f"{name}_Mat\x00\x01Material"), _prop_str("")],
                 children=[
        _node("Version", [_prop_i32(102)]),
        _node("ShadingModel", [_prop_str("Phong")]),
        _node("MultiLayer", [_prop_i32(0)]),
        _node("Properties70", children=[
            _p70_vec("DiffuseColor", "Color", "", "A", (0.8, 0.8, 0.8)),
            _p70_double("DiffuseFactor", 1.0),
        ]),
    ])


def _skinned_preamble():
    """Header + GlobalSettings + Documents + References (identical to write_fbx_binary)."""
    now = time.localtime()
    header_ext = _node("FBXHeaderExtension", children=[
        _node("FBXHeaderVersion", [_prop_i32(1003)]),
        _node("FBXVersion", [_prop_i32(7400)]),
        _node("EncryptionType", [_prop_i32(0)]),
        _node("CreationTimeStamp", children=[
            _node("Version", [_prop_i32(1000)]),
            _node("Year", [_prop_i32(now.tm_year)]),
            _node("Month", [_prop_i32(now.tm_mon)]),
            _node("Day", [_prop_i32(now.tm_mday)]),
            _node("Hour", [_prop_i32(now.tm_hour)]),
            _node("Minute", [_prop_i32(now.tm_min)]),
            _node("Second", [_prop_i32(now.tm_sec)]),
            _node("Millisecond", [_prop_i32(0)]),
        ]),
        _node("Creator", [_prop_str("Idol-test fbx_writer 2.0 (Binary 7400 skinned)")]),
    ])
    global_settings = _node("GlobalSettings", children=[
        _node("Version", [_prop_i32(1000)]),
        _node("Properties70", children=[
            _p70_int("UpAxis", 1),
            _p70_int("UpAxisSign", 1),
            _p70_int("FrontAxis", 2),
            _p70_int("FrontAxisSign", 1),
            _p70_int("CoordAxis", 0),
            _p70_int("CoordAxisSign", 1),
            _p70_int("OriginalUpAxis", 2),
            _p70_int("OriginalUpAxisSign", 1),
            _p70_double("UnitScaleFactor", 1.0),
            _p70_double("OriginalUnitScaleFactor", 1.0),
            _p70_vec("AmbientColor", "ColorRGB", "Color", "", (0.0, 0.0, 0.0)),
            _p70_str("DefaultCamera", "KString", "", "Producer Perspective"),
        ]),
    ])
    documents = _node("Documents", children=[
        _node("Count", [_prop_i32(1)]),
        _node("Document", [_prop_i64(917941052), _prop_str("Scene"), _prop_str("Scene")], children=[
            _node("Properties70", children=[
                _node("P", [_prop_str("SourceObject"), _prop_str("object"), _prop_str(""), _prop_str("")]),
                _p70_str("ActiveAnimStackName", "KString", "", ""),
            ]),
            _node("RootNode", [_prop_i64(0)]),
        ]),
    ])
    references = _node("References", force_block=True)
    return [header_ext, global_settings, documents, references]


def _skinned_definitions(n_models, n_node_attrs, n_pose, n_deformers):
    ot = [
        _node("ObjectType", [_prop_str("GlobalSettings")], children=[_node("Count", [_prop_i32(1)])]),
        _node("ObjectType", [_prop_str("NodeAttribute")], children=[_node("Count", [_prop_i32(n_node_attrs)])]),
        _node("ObjectType", [_prop_str("Geometry")], children=[_node("Count", [_prop_i32(1)])]),
        _node("ObjectType", [_prop_str("Model")], children=[_node("Count", [_prop_i32(n_models)])]),
        _node("ObjectType", [_prop_str("Material")], children=[_node("Count", [_prop_i32(1)])]),
        _node("ObjectType", [_prop_str("Pose")], children=[_node("Count", [_prop_i32(n_pose)])]),
        _node("ObjectType", [_prop_str("Deformer")], children=[_node("Count", [_prop_i32(n_deformers)])]),
    ]
    return _node("Definitions", children=[
        _node("Version", [_prop_i32(100)]),
        _node("Count", [_prop_i32(len(ot))]),
        *ot,
    ])


def _mesh_model(model_id, name, lcl_t=(0.0, 0.0, 0.0), lcl_r=(0.0, 0.0, 0.0),
                lcl_s=(1.0, 1.0, 1.0)):
    props = [
        _p70_vec("Lcl Translation", "Lcl Translation", "", "A", lcl_t),
        _p70_vec("Lcl Rotation", "Lcl Rotation", "", "A", lcl_r),
        _p70_vec("Lcl Scaling", "Lcl Scaling", "", "A", lcl_s),
        _p70_int("DefaultAttributeIndex", 0),
        _p70_int("InheritType", 1, "enum", ""),
    ]
    return _node_model(model_id, name, "Mesh", props)


def _conn(src, dst):
    return _node("C", [_prop_str("OO"), _prop_i64(src), _prop_i64(dst), _prop_i32(0)])


# ---------------------------------------------------------------------------
# Skinned Binary FBX 7.4 (pure Python)
# ---------------------------------------------------------------------------

def write_fbx_skinned(path, verts, faces, skin, name="mesh", compute_normals=True):
    """Write a skinned Binary FBX 7.4 (Binary 7.4, 100% pure Python).

    ``skin`` is a dict produced by the rig/convert pipeline:
      {
        "bones": [ { "name": str, "local_translation": [x,y,z],
                     "local_rotation": [rx,ry,rz] (degrees),
                     "parent": <bone index or None> }, ... ],
        "bind_world": [ 4x4 (np.matrix/array) per bone, in FBX scene space ],
        "armature_world": 4x4 (default identity),
        "clusters": [ { "bone": str, "indices": [...], "weights": [...] }, ... ],
      }

    Conventions verified against the Blender FBX importer (ground truth):
      * bone Model (LimbNode) Lcl translation = bind_world[i] 4th column
      * bone Model Lcl rotation = 0 (orientation carried by the bind matrix)
      * Cluster.Transform         = inverse(bind_world[bone])
      * Cluster.TransformLink     = bind_world[bone]
      * Cluster.TransformAssociateModel = armature_world
      * BindPose.Matrix per bone  = bind_world[bone]
    """
    # Triangles only, matching write_fbx_binary (convert_fbx always outputs (M,3)).
    verts = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    V = len(verts)
    F = len(faces)

    bones = skin["bones"]
    bind_world = [np.array(m, dtype=np.float64) for m in skin["bind_world"]]
    arm_world = np.array(skin.get("armature_world", np.identity(4)), dtype=np.float64)
    clusters = skin["clusters"]

    # Fixed, distinct 64-bit ids
    MESH_ID = 0x40000001
    ARM_ID = 0x40000002
    GEOM_ID = 0x41000001
    MAT_ID = 0x44000001
    SKIN_ID = 0x42000001
    POSE_ID = 0x43000001
    bone_ids = [0x40000100 + i for i in range(len(bones))]
    na_bone_ids = [0x50000100 + i for i in range(len(bones))]
    NA_ARM_ID = 0x50000002
    cluster_ids = [0x42000100 + i for i in range(len(clusters))]
    name_to_bone = {b["name"]: i for i, b in enumerate(bones)}

    normals = _compute_normals(verts, faces) if compute_normals else None

    geometry = _skinned_geometry(GEOM_ID, name, verts, faces, normals)
    material = _material_node(MAT_ID, name)

    n_models = 2 + len(bones)          # mesh + arm + bones
    n_node_attrs = 1 + len(bones)      # arm + bones

    # Models
    models = [
        _mesh_model(MESH_ID, name),
        _node_model(ARM_ID, name + "_Arm", "LimbNode", [
            _p70_int("DefaultAttributeIndex", 0),
            _p70_int("InheritType", 1, "enum", ""),
        ]),
    ]
    for i, b in enumerate(bones):
        # Lcl T/R of the bone Model is the bind pose RELATIVE TO ITS PARENT.
        #   root bone (parent None) -> relative to the armature (arm_world)
        #   child bone              -> relative to its parent bone's bind world
        # Writing the raw bind_world here was only correct for a flat hierarchy
        # (identity parent); deep rigs need the local form or every nested bone
        # inherits its parent's transform twice. FBX stores Lcl Euler, so we
        # decompose the local 4x4 into (translation, euler XYZ).
        par = b.get("parent")
        if par is None:
            local_i = np.linalg.inv(arm_world) @ bind_world[i]
        else:
            local_i = np.linalg.inv(bind_world[par]) @ bind_world[i]
        t, rot, s = _trs_from_matrix(local_i)
        models.append(_bone_model(bone_ids[i], b["name"], t, rot, s))

    # Node attributes (one per bone + armature root)
    node_attrs = [_node_attribute(NA_ARM_ID, name + "_Arm")]
    for i, b in enumerate(bones):
        node_attrs.append(_node_attribute(na_bone_ids[i], b["name"]))

    # Bind pose: armature + mesh + every bone, using bind world matrices
    pose_nodes = [(ARM_ID, arm_world), (MESH_ID, np.identity(4))]
    for i, b in enumerate(bones):
        pose_nodes.append((bone_ids[i], bind_world[i]))
    pose = _bind_pose(POSE_ID, name, pose_nodes)

    # Skin + clusters
    deformers = [_skin_deformer(SKIN_ID, name + "_Skin")]
    for c in clusters:
        bi = name_to_bone[c["bone"]]
        t_link = bind_world[bi]
        t_inv = np.linalg.inv(t_link)
        deformers.append(_cluster_deformer(
            cluster_ids[len(deformers) - 1], c["bone"],
            c["indices"], c["weights"],
            t_inv, t_link, arm_world,
        ))

    # Connections (child -> parent semantics, 'OO')
    conns = []
    conns.append(_conn(ARM_ID, 0))                 # armature root -> scene root
    for i, b in enumerate(bones):
        par = b.get("parent")
        parent_id = bone_ids[par] if par is not None else ARM_ID
        conns.append(_conn(bone_ids[i], parent_id))
    conns.append(_conn(NA_ARM_ID, ARM_ID))
    for i, b in enumerate(bones):
        conns.append(_conn(na_bone_ids[i], bone_ids[i]))
    conns.append(_conn(MESH_ID, 0))                # mesh model -> scene root
    conns.append(_conn(GEOM_ID, MESH_ID))          # geometry -> mesh model
    conns.append(_conn(MAT_ID, MESH_ID))           # material -> mesh model
    conns.append(_conn(SKIN_ID, GEOM_ID))          # skin -> skinned geometry (as Blender exports)
    for i, c in enumerate(clusters):
        conns.append(_conn(cluster_ids[i], SKIN_ID))
        conns.append(_conn(bone_ids[name_to_bone[c["bone"]]], cluster_ids[i]))

    objects = _node("Objects", children=models + [geometry, material]
                    + node_attrs + [pose] + deformers)
    connections = _node("Connections", children=conns)
    takes = _node("Takes", children=[_node("Current", [_prop_str("")])])
    definitions = _skinned_definitions(n_models, n_node_attrs, 1, len(deformers))

    top_nodes = _skinned_preamble() + [definitions, objects, connections, takes]

    chunks = [_FBX_MAGIC, struct.pack("<I", _FBX_VERSION)]
    offset = len(_FBX_MAGIC) + 4
    for n in top_nodes:
        enc = _encode_node(n, offset)
        chunks.append(enc)
        offset += len(enc)
    chunks.append(_NULL_SENTINEL)
    offset += len(_NULL_SENTINEL)
    chunks.append(_FOOT_ID)
    pad = ((offset + 16 + 15) & ~15) - (offset + 16)
    if pad == 0:
        pad = 16
    chunks.append(b"\x00" * pad)
    chunks.append(struct.pack("<I", _FBX_VERSION))
    chunks.append(b"\x00" * 120)
    chunks.append(bytes.fromhex("f8 5a 8c 6a de f5 d9 7e ec e9 0c e3 75 8f 29 0b"))

    data = b"".join(chunks)
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)
    return {
        "bytes": len(data),
        "verts": V,
        "faces": F,
        "bones": len(bones),
        "clusters": len(clusters),
        "ascii": False,
        "verified_by": "python-binary",
        "path": path,
    }


def _encode_node(node, offset):
    name, props, children, force_block = node
    name_b = name.encode("ascii")
    prop_bytes = b"".join(props)
    header_len = 13 + len(name_b)
    child_offset = offset + header_len + len(prop_bytes)

    child_chunks = []
    cur = child_offset
    for ch in children:
        ch_bytes = _encode_node(ch, cur)
        child_chunks.append(ch_bytes)
        cur += len(ch_bytes)

    if children or force_block:
        child_chunks.append(_NULL_SENTINEL)
        cur += len(_NULL_SENTINEL)

    end_offset = cur
    header = struct.pack("<IIIB", end_offset, len(props), len(prop_bytes), len(name_b)) + name_b
    return header + prop_bytes + b"".join(child_chunks)


def _p70_int(name, value, ptype="int", sub="Integer"):
    return _node("P", [_prop_str(name), _prop_str(ptype), _prop_str(sub), _prop_str(""), _prop_i32(value)])


def _p70_bool(name, value):
    return _node("P", [_prop_str(name), _prop_str("bool"), _prop_str(""), _prop_str(""), _prop_i32(1 if value else 0)])


def _p70_double(name, value):
    return _node("P", [_prop_str(name), _prop_str("double"), _prop_str("Number"), _prop_str(""), _prop_f64(value)])


def _p70_vec(name, ptype, sub, flags, xyz):
    return _node("P", [
        _prop_str(name), _prop_str(ptype), _prop_str(sub), _prop_str(flags),
        _prop_f64(xyz[0]), _prop_f64(xyz[1]), _prop_f64(xyz[2]),
    ])


def _p70_str(name, ptype, sub, value):
    return _node("P", [_prop_str(name), _prop_str(ptype), _prop_str(sub), _prop_str(""), _prop_str(value)])


def write_fbx_binary(path, verts, faces, vertex_colors=None, name="mesh",
                     compute_normals=True):
    """Write a valid FBX 7.4 Binary file in pure Python (Z-up input -> Y-up FBX).

    Input vertices are expected in Z-up world space (meters). We convert to
    FBX Y-up centimeter conventions on the Model transform (rotation -90 deg X,
    scale 100) matching Blender's FBX exporter so Unity and Blender both load
    the mesh upright and at the expected scale.
    """
    v = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    V = len(v)
    F = len(f)

    normals = _compute_normals(v, f) if compute_normals else None

    # FBX PolygonVertexIndex encodes each triangle [a, b, c] as [a, b, ~c]
    poly_idx = f.astype(np.int32).copy()
    if F > 0:
        poly_idx[:, 2] = ~poly_idx[:, 2]

    # Flatten vertex positions (N*3 float64) and normals (N*3 float64)
    v_flat = v.reshape(-1)
    poly_flat = poly_idx.reshape(-1)

    MODEL_ID = 276415347
    GEOM_ID = 817038466
    MAT_ID = 512000101
    now = time.localtime()

    header_ext = _node("FBXHeaderExtension", children=[
        _node("FBXHeaderVersion", [_prop_i32(1003)]),
        _node("FBXVersion", [_prop_i32(7400)]),
        _node("EncryptionType", [_prop_i32(0)]),
        _node("CreationTimeStamp", children=[
            _node("Version", [_prop_i32(1000)]),
            _node("Year", [_prop_i32(now.tm_year)]),
            _node("Month", [_prop_i32(now.tm_mon)]),
            _node("Day", [_prop_i32(now.tm_mday)]),
            _node("Hour", [_prop_i32(now.tm_hour)]),
            _node("Minute", [_prop_i32(now.tm_min)]),
            _node("Second", [_prop_i32(now.tm_sec)]),
            _node("Millisecond", [_prop_i32(0)]),
        ]),
        _node("Creator", [_prop_str("Idol-test fbx_writer 2.0 (Binary 7400)")]),
    ])

    global_settings = _node("GlobalSettings", children=[
        _node("Version", [_prop_i32(1000)]),
        _node("Properties70", children=[
            _p70_int("UpAxis", 1),
            _p70_int("UpAxisSign", 1),
            _p70_int("FrontAxis", 2),
            _p70_int("FrontAxisSign", 1),
            _p70_int("CoordAxis", 0),
            _p70_int("CoordAxisSign", 1),
            _p70_int("OriginalUpAxis", 2),
            _p70_int("OriginalUpAxisSign", 1),
            _p70_double("UnitScaleFactor", 1.0),
            _p70_double("OriginalUnitScaleFactor", 1.0),
            _p70_vec("AmbientColor", "ColorRGB", "Color", "", (0.0, 0.0, 0.0)),
            _p70_str("DefaultCamera", "KString", "", "Producer Perspective"),
        ]),
    ])

    documents = _node("Documents", children=[
        _node("Count", [_prop_i32(1)]),
        _node("Document", [_prop_i64(917941052), _prop_str("Scene"), _prop_str("Scene")], children=[
            _node("Properties70", children=[
                _node("P", [_prop_str("SourceObject"), _prop_str("object"), _prop_str(""), _prop_str("")]),
                _p70_str("ActiveAnimStackName", "KString", "", ""),
            ]),
            _node("RootNode", [_prop_i64(0)]),
        ]),
    ])

    references = _node("References", force_block=True)

    definitions = _node("Definitions", children=[
        _node("Version", [_prop_i32(100)]),
        _node("Count", [_prop_i32(4)]),
        _node("ObjectType", [_prop_str("GlobalSettings")], children=[_node("Count", [_prop_i32(1)])]),
        _node("ObjectType", [_prop_str("Geometry")], children=[_node("Count", [_prop_i32(1)])]),
        _node("ObjectType", [_prop_str("Model")], children=[_node("Count", [_prop_i32(1)])]),
        _node("ObjectType", [_prop_str("Material")], children=[_node("Count", [_prop_i32(1)])]),
    ])

    geom_children = [
        _node("Properties70", force_block=True),
        _node("GeometryVersion", [_prop_i32(124)]),
        _node("Vertices", [_prop_array("d", "d", v_flat)]),
        _node("PolygonVertexIndex", [_prop_array("i", "i", poly_flat)]),
    ]
    layer_elements = []

    if normals is not None:
        geom_children.append(_node("LayerElementNormal", [_prop_i32(0)], children=[
            _node("Version", [_prop_i32(101)]),
            _node("Name", [_prop_str("")]),
            _node("MappingInformationType", [_prop_str("ByVertice")]),
            _node("ReferenceInformationType", [_prop_str("Direct")]),
            _node("Normals", [_prop_array("d", "d", normals.reshape(-1))]),
        ]))
        layer_elements.append(_node("LayerElement", children=[
            _node("Type", [_prop_str("LayerElementNormal")]),
            _node("TypedIndex", [_prop_i32(0)]),
        ]))

    if vertex_colors is not None:
        vc = np.asarray(vertex_colors, dtype=np.float64).reshape(-1, 3)
        rgba = np.ones((V, 4), dtype=np.float64)
        rgba[:, :3] = np.clip(vc, 0.0, 1.0)
        geom_children.append(_node("LayerElementColor", [_prop_i32(0)], children=[
            _node("Version", [_prop_i32(101)]),
            _node("Name", [_prop_str("Col")]),
            _node("MappingInformationType", [_prop_str("ByVertice")]),
            _node("ReferenceInformationType", [_prop_str("Direct")]),
            _node("Colors", [_prop_array("d", "d", rgba.reshape(-1))]),
        ]))
        layer_elements.append(_node("LayerElement", children=[
            _node("Type", [_prop_str("LayerElementColor")]),
            _node("TypedIndex", [_prop_i32(0)]),
        ]))

    geom_children.append(_node("LayerElementMaterial", [_prop_i32(0)], children=[
        _node("Version", [_prop_i32(101)]),
        _node("Name", [_prop_str("")]),
        _node("MappingInformationType", [_prop_str("AllSame")]),
        _node("ReferenceInformationType", [_prop_str("IndexToDirect")]),
        _node("Materials", [_prop_array("i", "i", [0])]),
    ]))
    layer_elements.append(_node("LayerElement", children=[
        _node("Type", [_prop_str("LayerElementMaterial")]),
        _node("TypedIndex", [_prop_i32(0)]),
    ]))

    geom_children.append(_node("Layer", [_prop_i32(0)], children=[
        _node("Version", [_prop_i32(100)]),
        *layer_elements,
    ]))

    geom_node = _node(
        "Geometry",
        [_prop_i64(GEOM_ID), _prop_str(f"{name}\x00\x01Geometry"), _prop_str("Mesh")],
        children=geom_children,
    )

    model_node = _node(
        "Model",
        [_prop_i64(MODEL_ID), _prop_str(f"{name}\x00\x01Model"), _prop_str("Mesh")],
        children=[
            _node("Version", [_prop_i32(232)]),
            _node("Properties70", children=[
                _p70_vec("Lcl Translation", "Lcl Translation", "", "A", (0.0, 0.0, 0.0)),
                _p70_vec("Lcl Rotation", "Lcl Rotation", "", "A", (-90.0, 0.0, 0.0)),
                _p70_vec("Lcl Scaling", "Lcl Scaling", "", "A", (100.0, 100.0, 100.0)),
                _p70_int("DefaultAttributeIndex", 0),
                _p70_int("InheritType", 1, "enum", ""),
            ]),
            _node("MultiLayer", [_prop_i32(0)]),
            _node("MultiTake", [_prop_i32(0)]),
            _node("Shading", [_prop_char(1)]),
            _node("Culling", [_prop_str("CullingOff")]),
        ],
    )

    material_node = _node(
        "Material",
        [_prop_i64(MAT_ID), _prop_str(f"{name}_Mat\x00\x01Material"), _prop_str("")],
        children=[
            _node("Version", [_prop_i32(102)]),
            _node("ShadingModel", [_prop_str("Phong")]),
            _node("MultiLayer", [_prop_i32(0)]),
            _node("Properties70", children=[
                _p70_vec("DiffuseColor", "Color", "", "A", (0.8, 0.8, 0.8)),
                _p70_double("DiffuseFactor", 1.0),
            ]),
        ],
    )

    objects = _node("Objects", children=[geom_node, model_node, material_node])

    connections = _node("Connections", children=[
        _node("C", [_prop_str("OO"), _prop_i64(MODEL_ID), _prop_i64(0)]),
        _node("C", [_prop_str("OO"), _prop_i64(GEOM_ID), _prop_i64(MODEL_ID)]),
        _node("C", [_prop_str("OO"), _prop_i64(MAT_ID), _prop_i64(MODEL_ID)]),
    ])

    takes = _node("Takes", children=[
        _node("Current", [_prop_str("")]),
    ])

    top_nodes = [header_ext, global_settings, documents, references, definitions, objects, connections, takes]

    chunks = [_FBX_MAGIC, struct.pack("<I", _FBX_VERSION)]
    offset = len(_FBX_MAGIC) + 4
    for n in top_nodes:
        enc = _encode_node(n, offset)
        chunks.append(enc)
        offset += len(enc)

    # Top-level null sentinel + standard FBX binary footer
    chunks.append(_NULL_SENTINEL)
    offset += len(_NULL_SENTINEL)
    chunks.append(_FOOT_ID)
    pad = ((offset + 16 + 15) & ~15) - (offset + 16)
    if pad == 0:
        pad = 16
    chunks.append(b"\x00" * pad)
    chunks.append(struct.pack("<I", _FBX_VERSION))
    chunks.append(b"\x00" * 120)
    chunks.append(bytes.fromhex("f8 5a 8c 6a de f5 d9 7e ec e9 0c e3 75 8f 29 0b"))

    data = b"".join(chunks)
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)
    return {
        "bytes": len(data),
        "verts": V,
        "faces": F,
        "ascii": False,
        "verified_by": "python-binary",
        "path": path,
    }


# ---------------------------------------------------------------------------
# Blender headless FBX exporter + round-trip verifier
# ---------------------------------------------------------------------------

_BLENDER_UNWRAP_SCRIPT = r"""
import os, struct, sys, bpy

argv = sys.argv[sys.argv.index("--") + 1:]
in_ply, out_blend, out_tri_uv_bin = argv[:3]

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.ply_import(filepath=in_ply, forward_axis="Y", up_axis="Z")

objs = [o for o in bpy.data.objects if o.type == "MESH"]
if not objs:
    raise RuntimeError("No mesh imported from PLY: " + in_ply)

obj = objs[0]
bpy.context.view_layer.objects.active = obj
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")
bpy.ops.uv.smart_project(angle_limit=1.48, island_margin=0.003, area_weight=1.0)
bpy.ops.object.mode_set(mode="OBJECT")

me = obj.data
if any(len(p.vertices) != 3 for p in me.polygons):
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.to_mesh(me)
    bm.free()

for p in me.polygons:
    p.use_smooth = True
me.update()

uv_layer = me.uv_layers.active.data
verts = me.vertices
with open(out_tri_uv_bin, "wb") as fh:
    for p in me.polygons:
        for li in p.loop_indices:
            v = verts[me.loops[li].vertex_index]
            co = v.co
            n = v.normal
            uv = uv_layer[li].uv
            fh.write(struct.pack("<ffffffff", co.x, co.y, co.z, n.x, n.y, n.z, uv.x, uv.y))

bpy.ops.wm.save_as_mainfile(filepath=out_blend)
"""


def unwrap_uv_blender(mesh_ply_path, out_blend_path, out_tri_uv_bin, timeout=180):
    """Run Blender smart_project UV unwrap once and dump (F*3, 5) [x,y,z,u,v] buffer."""
    blender = find_blender()
    if not blender:
        return False
    with tempfile.TemporaryDirectory() as tmpd:
        sp = os.path.join(tmpd, "unwrap.py")
        with open(sp, "w", encoding="utf-8") as fh:
            fh.write(_BLENDER_UNWRAP_SCRIPT)
        proc = subprocess.run(
            [
                blender, "--background", "--factory-startup",
                "--python", sp, "--",
                os.path.abspath(mesh_ply_path),
                os.path.abspath(out_blend_path),
                os.path.abspath(out_tri_uv_bin),
            ],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
        )
        if proc.returncode != 0 or not os.path.exists(out_tri_uv_bin):
            raise RuntimeError(
                f"Blender UV unwrap failed (rc={proc.returncode}): "
                f"{(proc.stderr or proc.stdout or '').strip()[-400:]}"
            )
    return True


_BLENDER_EXPORT_SCRIPT = r"""
import json, os, struct, sys
import bpy

argv = sys.argv[sys.argv.index("--") + 1:]
in_source, out_fbx, verified_ply, verified_uv, texture_path, mesh_name, out_json = argv[:7]

if in_source.lower().endswith(".blend"):
    bpy.ops.wm.open_mainfile(filepath=in_source)
else:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.ply_import(filepath=in_source, forward_axis="Y", up_axis="Z")

objs = [o for o in bpy.data.objects if o.type == "MESH"]
if not objs:
    raise RuntimeError("No mesh imported from source: " + in_source)

obj = objs[0]
obj.name = mesh_name
obj.data.name = mesh_name

for poly in obj.data.polygons:
    poly.use_smooth = True

obj.data.materials.clear()
mat = bpy.data.materials.new(name=mesh_name + "_Mat")
mat.use_nodes = True
nodes = mat.node_tree.nodes
links = mat.node_tree.links
bsdf = nodes.get("Principled BSDF")
if bsdf and "Roughness" in bsdf.inputs:
    bsdf.inputs["Roughness"].default_value = 0.85
if bsdf and "Specular IOR Level" in bsdf.inputs:
    bsdf.inputs["Specular IOR Level"].default_value = 0.15

if texture_path and os.path.exists(texture_path) and obj.data.uv_layers:
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(texture_path)
    if bsdf:
        links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
elif obj.data.color_attributes:
    vcol = nodes.new("ShaderNodeVertexColor")
    vcol.layer_name = obj.data.color_attributes[0].name
    if bsdf:
        links.new(vcol.outputs["Color"], bsdf.inputs["Base Color"])

obj.data.materials.append(mat)

os.makedirs(os.path.dirname(os.path.abspath(out_fbx)), exist_ok=True)
bpy.ops.export_scene.fbx(
    filepath=out_fbx,
    use_selection=False,
    path_mode="COPY",
    embed_textures=True,
    colors_type="SRGB",
    mesh_smooth_type="FACE",
    apply_scale_options="FBX_SCALE_ALL",
)

# Round-trip verify by importing the generated FBX back into an empty Blender scene
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=out_fbx)
rt_objs = [o for o in bpy.data.objects if o.type == "MESH"]
if not rt_objs:
    raise RuntimeError("FBX round-trip produced no mesh objects")

rt = rt_objs[0]
me = rt.data
if any(len(p.vertices) != 3 for p in me.polygons):
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.to_mesh(me)
    bm.free()

n_verts = len(me.vertices)
n_faces = len(me.polygons)

mat_w = rt.matrix_world
verts_w = [mat_w @ v.co for v in me.vertices]
xs = [v.x for v in verts_w]
ys = [v.y for v in verts_w]
zs = [v.z for v in verts_w]

# Write verified mesh PLY & matching UV buffer directly from round-tripped FBX
if verified_ply:
    col_acc = [[0.0, 0.0, 0.0, 0] for _ in range(n_verts)]
    if me.color_attributes:
        ca = me.color_attributes[0]
        if ca.domain == "CORNER":
            for loop in me.loops:
                vi = loop.vertex_index
                item = ca.data[loop.index]
                c = getattr(item, "color_srgb", item.color)
                col_acc[vi][0] += c[0]
                col_acc[vi][1] += c[1]
                col_acc[vi][2] += c[2]
                col_acc[vi][3] += 1
        elif ca.domain == "POINT":
            for vi in range(n_verts):
                item = ca.data[vi]
                c = getattr(item, "color_srgb", item.color)
                col_acc[vi] = [c[0], c[1], c[2], 1]

    with open(verified_ply, "wb") as fh:
        hdr = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {n_verts}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            f"element face {n_faces}\n"
            "property list uchar int vertex_indices\n"
            "end_header\n"
        ).encode("ascii")
        fh.write(hdr)
        for vi, co in enumerate(verts_w):
            r_acc, g_acc, b_acc, cnt = col_acc[vi]
            if cnt > 0:
                r = max(0, min(255, int(round((r_acc / cnt) * 255.0))))
                g = max(0, min(255, int(round((g_acc / cnt) * 255.0))))
                b = max(0, min(255, int(round((b_acc / cnt) * 255.0))))
            else:
                r = g = b = 180
            fh.write(struct.pack("<fffBBB", co.x, co.y, co.z, r, g, b))
        for poly in me.polygons:
            vs = poly.vertices
            fh.write(struct.pack("<Biii", 3, vs[0], vs[1], vs[2]))

if verified_uv and me.uv_layers:
    uv_data = me.uv_layers[0].data
    with open(verified_uv, "wb") as fh:
        for poly in me.polygons:
            for li in poly.loop_indices:
                uv = uv_data[li].uv
                fh.write(struct.pack("<ff", uv.x, uv.y))

info = {
    "verts": n_verts,
    "faces": n_faces,
    "bytes": os.path.getsize(out_fbx),
    "color_layers": len(me.color_attributes),
    "uv_layers": len(me.uv_layers),
    "has_texture": bool(bpy.data.images) and bool(me.uv_layers),
    "bounds": {
        "min": [round(min(xs), 5), round(min(ys), 5), round(min(zs), 5)] if xs else [0, 0, 0],
        "max": [round(max(xs), 5), round(max(ys), 5), round(max(zs), 5)] if xs else [0, 0, 0],
    },
    "verified_by": "blender",
}
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(info, f)
"""


def write_fbx_blender(path, mesh_ply_path, name="mesh", verified_ply_path=None,
                      blend_path=None, texture_path=None, verified_uv_path=None,
                      timeout=180):
    """Export Binary FBX 7.4 via Blender CLI (with optional UV texture) and round-trip verify."""
    blender = find_blender()
    if not blender:
        return None

    with tempfile.TemporaryDirectory() as tmpd:
        script_path = os.path.join(tmpd, "export_verify.py")
        out_json = os.path.join(tmpd, "result.json")
        with open(script_path, "w", encoding="utf-8") as fh:
            fh.write(_BLENDER_EXPORT_SCRIPT)

        in_src = os.path.abspath(blend_path) if blend_path else os.path.abspath(mesh_ply_path)
        cmd = [
            blender, "--background", "--factory-startup",
            "--python", script_path, "--",
            in_src,
            os.path.abspath(path),
            os.path.abspath(verified_ply_path) if verified_ply_path else "",
            os.path.abspath(verified_uv_path) if verified_uv_path else "",
            os.path.abspath(texture_path) if texture_path else "",
            name,
            out_json,
        ]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
        )
        if proc.returncode != 0 or not os.path.exists(out_json):
            raise RuntimeError(
                f"Blender FBX export/verify failed (rc={proc.returncode}): "
                f"{(proc.stderr or proc.stdout or '').strip()[-500:]}"
            )
        import json
        with open(out_json, "r", encoding="utf-8") as fh:
            info = json.load(fh)
        info["ascii"] = False
        info["path"] = path
        return info


def verify_fbx_blender(fbx_path, out_ply_path=None, timeout=120):
    """Verify an existing FBX file by importing it in Blender (optionally extract PLY)."""
    blender = find_blender()
    if not blender:
        return None

    script = r"""
import json, os, struct, sys, bpy, bmesh
argv = sys.argv[sys.argv.index("--") + 1:]
fbx_path, out_ply, out_json = argv[:3]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=fbx_path)
objs = [o for o in bpy.data.objects if o.type == "MESH"]
if not objs:
    raise RuntimeError("No mesh objects inside FBX")
rt = objs[0]
me = rt.data
if any(len(p.vertices) != 3 for p in me.polygons):
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.to_mesh(me)
    bm.free()
n_verts, n_faces = len(me.vertices), len(me.polygons)
mat_w = rt.matrix_world
verts_w = [mat_w @ v.co for v in me.vertices]
if out_ply:
    col_acc = [[0.0, 0.0, 0.0, 0] for _ in range(n_verts)]
    if me.color_attributes:
        ca = me.color_attributes[0]
        if ca.domain == "CORNER":
            for loop in me.loops:
                vi = loop.vertex_index
                item = ca.data[loop.index]
                c = getattr(item, "color_srgb", item.color)
                col_acc[vi][0] += c[0]; col_acc[vi][1] += c[1]; col_acc[vi][2] += c[2]; col_acc[vi][3] += 1
        elif ca.domain == "POINT":
            for vi in range(n_verts):
                item = ca.data[vi]
                c = getattr(item, "color_srgb", item.color)
                col_acc[vi] = [c[0], c[1], c[2], 1]
    with open(out_ply, "wb") as fh:
        fh.write((
            "ply\nformat binary_little_endian 1.0\n"
            f"element vertex {n_verts}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            f"element face {n_faces}\nproperty list uchar int vertex_indices\nend_header\n"
        ).encode("ascii"))
        for vi, co in enumerate(verts_w):
            r_a, g_a, b_a, cnt = col_acc[vi]
            if cnt > 0:
                r = max(0, min(255, int(round((r_a / cnt) * 255.0))))
                g = max(0, min(255, int(round((g_a / cnt) * 255.0))))
                b = max(0, min(255, int(round((b_a / cnt) * 255.0))))
            else:
                r = g = b = 180
            fh.write(struct.pack("<fffBBB", co.x, co.y, co.z, r, g, b))
        for poly in me.polygons:
            vs = poly.vertices
            fh.write(struct.pack("<Biii", 3, vs[0], vs[1], vs[2]))
with open(out_json, "w") as f:
    json.dump({"verts": n_verts, "faces": n_faces, "bytes": os.path.getsize(fbx_path), "verified_by": "blender"}, f)
"""
    with tempfile.TemporaryDirectory() as tmpd:
        sp = os.path.join(tmpd, "verify.py")
        oj = os.path.join(tmpd, "res.json")
        with open(sp, "w", encoding="utf-8") as fh:
            fh.write(script)
        proc = subprocess.run(
            [blender, "--background", "--factory-startup", "--python", sp, "--",
             os.path.abspath(fbx_path), os.path.abspath(out_ply_path) if out_ply_path else "", oj],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
        )
        if proc.returncode != 0 or not os.path.exists(oj):
            raise RuntimeError(
                f"Blender FBX verify failed: {(proc.stderr or proc.stdout or '').strip()[-400:]}"
            )
        import json
        with open(oj, "r", encoding="utf-8") as fh:
            return json.load(fh)


def write_fbx(path, verts, faces, vertex_colors=None, name="mesh",
              compute_normals=True, mesh_ply_path=None, verified_ply_path=None,
              blend_path=None, texture_path=None, verified_uv_path=None):
    """Write a verified Binary FBX 7.4 file.

    Uses Blender headless when available (and round-trip verifies the written
    FBX); otherwise falls back to the pure-Python Binary FBX 7.4 writer.
    """
    if (mesh_ply_path or blend_path) and find_blender():
        return write_fbx_blender(
            path,
            mesh_ply_path=mesh_ply_path,
            name=name,
            verified_ply_path=verified_ply_path,
            blend_path=blend_path,
            texture_path=texture_path,
            verified_uv_path=verified_uv_path,
        )

    res = write_fbx_binary(
        path, verts, faces,
        vertex_colors=vertex_colors,
        name=name,
        compute_normals=compute_normals,
    )
    if find_blender():
        vinfo = verify_fbx_blender(path, out_ply_path=verified_ply_path)
        if vinfo:
            res.update(vinfo)
    return res


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _selftest(path=None):
    if path is None:
        path = os.path.join(tempfile.gettempdir(), "fbx_selftest.fbx")
    verts = np.array([
        [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5],
    ], dtype=np.float32)
    faces = np.array([
        [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
        [0, 4, 5], [0, 5, 1], [3, 2, 6], [3, 6, 7],
        [1, 5, 6], [1, 6, 2], [4, 0, 3], [4, 3, 7],
    ], dtype=np.int32)
    colors = np.tile(np.array([[1.0, 0.2, 0.1]], dtype=np.float32), (8, 1))
    res = write_fbx_binary(path, verts, faces, vertex_colors=colors, name="selftest_box")
    with open(path, "rb") as fh:
        magic = fh.read(23)
    assert magic == _FBX_MAGIC, f"bad magic: {magic!r}"
    if find_blender():
        vinfo = verify_fbx_blender(path)
        assert vinfo["verts"] == 8 and vinfo["faces"] == 12, vinfo
        res.update(vinfo)
    print("selftest OK:", res)
    return res


if __name__ == "__main__":
    import sys
    _selftest(sys.argv[1] if len(sys.argv) > 1 else None)
