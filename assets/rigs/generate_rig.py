"""Build a Z-up humanoid armature and export it as a rig GLB (assets/rigs/sweetie_rig.glb).

Usage:
    python assets/rigs/generate_rig.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from img2rig.config import find_blender

OUT_GLB = Path(__file__).resolve().parent / "sweetie_rig.glb"

# Bone definitions: name -> (head, tail, parent). Z-up, +Y is "back", feet -Z side.
BONES = {
    "Root":      ((0.00, 0.00, -0.16), (0.00, 0.00, -0.02), None),
    "Spine":     ((0.00, 0.00, -0.02), (0.00, 0.00, 0.16), "Root"),
    "Chest":     ((0.00, 0.00, 0.16), (0.00, 0.00, 0.32), "Spine"),
    "Neck":      ((0.00, 0.00, 0.32), (0.00, 0.00, 0.42), "Chest"),
    "Head":      ((0.00, 0.00, 0.42), (0.00, 0.00, 0.52), "Neck"),
    "LShoulder": ((0.02, 0.00, 0.30), (0.16, 0.00, 0.30), "Chest"),
    "LUpperArm": ((0.16, 0.00, 0.30), (0.20, 0.00, 0.14), "LShoulder"),
    "LLowerArm": ((0.20, 0.00, 0.14), (0.22, 0.00, 0.00), "LUpperArm"),
    "LHand":     ((0.22, 0.00, 0.00), (0.235, -0.02, -0.10), "LLowerArm"),
    "RShoulder": ((-0.02, 0.00, 0.30), (-0.16, 0.00, 0.30), "Chest"),
    "RUpperArm": ((-0.16, 0.00, 0.30), (-0.20, 0.00, 0.14), "RShoulder"),
    "RLowerArm": ((-0.20, 0.00, 0.14), (-0.22, 0.00, 0.00), "RUpperArm"),
    "RHand":     ((-0.22, 0.00, 0.00), (-0.235, -0.02, -0.10), "RLowerArm"),
    "LThigh":    ((0.09, 0.00, -0.16), (0.10, 0.00, -0.34), "Root"),
    "LShin":     ((0.10, 0.00, -0.34), (0.10, 0.00, -0.49), "LThigh"),
    "LFoot":     ((0.10, 0.00, -0.49), (0.10, -0.12, -0.51), "LShin"),
    "RThigh":    ((-0.09, 0.00, -0.16), (-0.10, 0.00, -0.34), "Root"),
    "RShin":     ((-0.10, 0.00, -0.34), (-0.10, 0.00, -0.49), "RThigh"),
    "RFoot":     ((-0.10, 0.00, -0.49), (-0.10, -0.12, -0.51), "RShin"),
}

_BLENDER_SCRIPT = r"""
import json, os, sys
import bpy

out_glb = sys.argv[sys.argv.index("--") + 1]
bones = json.loads(os.environ["RIG_BONES_JSON"])

bpy.ops.wm.read_factory_settings(use_empty=True)

arm = bpy.data.armatures.new("SweetieRig")
obj = bpy.data.objects.new("SweetieRig", arm)
bpy.context.collection.objects.link(obj)
bpy.context.view_layer.objects.active = obj
obj.select_set(True)

bpy.ops.object.mode_set(mode="EDIT")
eb = arm.edit_bones
created = {}
def depth(name):
    n, d = name, 0
    while n in bones and bones[n][2]:
        n = bones[n][2]; d += 1
    return d
for name in sorted(bones, key=depth):
    head, tail, parent = bones[name]
    b = eb.new(name)
    b.head = head
    b.tail = tail
    created[name] = b
for name, (head, tail, parent) in bones.items():
    if parent and parent in created:
        created[name].parent = created[parent]
        created[name].use_connect = (parent == "Spine" and name == "Chest") or \
            (parent == "Chest" and name == "Neck") or (parent == "Neck" and name == "Head")
bpy.ops.object.mode_set(mode="OBJECT")

os.makedirs(os.path.dirname(os.path.abspath(out_glb)), exist_ok=True)
bpy.ops.export_scene.gltf(
    filepath=out_glb,
    export_format="GLB",
    use_selection=True,
    export_yup=False,
    export_apply=False,
    export_skins=True,
    export_animations=False,
)
n = len(bpy.data.objects.get("SweetieRig").data.bones)
sys.stderr.write("RIG_OK bones=%d path=%s\n" % (n, out_glb))
"""


def generate_rig(out_glb=None, timeout=120):
    out_glb = str(out_glb or OUT_GLB)
    blender = find_blender()
    if not blender:
        raise RuntimeError("Blender CLI not found; install Blender or set BLENDER_BIN")

    with tempfile.TemporaryDirectory() as tmpd:
        sp = os.path.join(tmpd, "build_rig.py")
        with open(sp, "w", encoding="utf-8") as fh:
            fh.write(_BLENDER_SCRIPT)
        env = dict(os.environ)
        env["RIG_BONES_JSON"] = json.dumps(BONES)
        proc = subprocess.run(
            [blender, "--background", "--factory-startup",
             "--python", sp, "--", os.path.abspath(out_glb)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout, env=env,
        )
        if proc.returncode != 0 or not os.path.exists(out_glb):
            tail = (proc.stderr or proc.stdout or "").strip()[-500:]
            raise RuntimeError(f"Rig GLB generation failed (rc={proc.returncode}): {tail}")
    return {"glb": out_glb, "bones": len(BONES), "bytes": os.path.getsize(out_glb)}


if __name__ == "__main__":
    print(json.dumps(generate_rig(), ensure_ascii=False))
