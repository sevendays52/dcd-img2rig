"""Generate a clean looping Mixamo-compatible Walk cycle FBX preset (assets/rigs/presets/walk.fbx).

Usage:
    python assets/rigs/generate_walk_preset.py
"""

from __future__ import annotations

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

_GEN_WALK_SCRIPT = r"""
import math, os, sys
import bpy
from mathutils import Matrix, Vector

out_fbx = sys.argv[sys.argv.index("--") + 1]

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.fps = 30
scene.render.fps_base = 1.0

arm = bpy.data.armatures.new("MixamoWalk")
obj = bpy.data.objects.new("MixamoWalk", arm)
bpy.context.collection.objects.link(obj)
bpy.context.view_layer.objects.active = obj
obj.select_set(True)

BONES = {
    "mixamorig:Hips":          ((0.00, 0.00, 0.84), (0.00, 0.00, 0.98), None),
    "mixamorig:Spine":         ((0.00, 0.00, 0.98), (0.00, 0.00, 1.16), "mixamorig:Hips"),
    "mixamorig:Spine2":        ((0.00, 0.00, 1.16), (0.00, 0.00, 1.32), "mixamorig:Spine"),
    "mixamorig:Neck":          ((0.00, 0.00, 1.32), (0.00, 0.00, 1.42), "mixamorig:Spine2"),
    "mixamorig:Head":          ((0.00, 0.00, 1.42), (0.00, 0.00, 1.56), "mixamorig:Neck"),
    "mixamorig:LeftShoulder":  ((0.02, 0.00, 1.30), (0.16, 0.00, 1.30), "mixamorig:Spine2"),
    "mixamorig:LeftArm":       ((0.16, 0.00, 1.30), (0.20, 0.00, 1.12), "mixamorig:LeftShoulder"),
    "mixamorig:LeftForeArm":   ((0.20, 0.00, 1.12), (0.22, 0.00, 0.94), "mixamorig:LeftArm"),
    "mixamorig:LeftHand":      ((0.22, 0.00, 0.94), (0.23, -0.02, 0.84), "mixamorig:LeftForeArm"),
    "mixamorig:RightShoulder": ((-0.02, 0.00, 1.30), (-0.16, 0.00, 1.30), "mixamorig:Spine2"),
    "mixamorig:RightArm":      ((-0.16, 0.00, 1.30), (-0.20, 0.00, 1.12), "mixamorig:RightShoulder"),
    "mixamorig:RightForeArm":  ((-0.20, 0.00, 1.12), (-0.22, 0.00, 0.94), "mixamorig:RightArm"),
    "mixamorig:RightHand":     ((-0.22, 0.00, 0.94), (-0.23, -0.02, 0.84), "mixamorig:RightForeArm"),
    "mixamorig:LeftUpLeg":     ((0.09, 0.00, 0.84), (0.10, 0.00, 0.48), "mixamorig:Hips"),
    "mixamorig:LeftLeg":       ((0.10, 0.00, 0.48), (0.10, 0.00, 0.12), "mixamorig:LeftUpLeg"),
    "mixamorig:LeftFoot":      ((0.10, 0.00, 0.12), (0.10, -0.14, 0.08), "mixamorig:LeftLeg"),
    "mixamorig:RightUpLeg":    ((-0.09, 0.00, 0.84), (-0.10, 0.00, 0.48), "mixamorig:Hips"),
    "mixamorig:RightLeg":      ((-0.10, 0.00, 0.48), (-0.10, 0.00, 0.12), "mixamorig:RightUpLeg"),
    "mixamorig:RightFoot":     ((-0.10, 0.00, 0.12), (-0.10, -0.14, 0.08), "mixamorig:RightLeg"),
}

bpy.ops.object.mode_set(mode="EDIT")
eb = arm.edit_bones
created = {}
for name, (h, t, _) in BONES.items():
    b = eb.new(name)
    b.head = h
    b.tail = t
    created[name] = b
for name, (_, _, p) in BONES.items():
    if p:
        created[name].parent = created[p]
bpy.ops.object.mode_set(mode="OBJECT")

rest_w = {b.name: b.matrix_local.to_3x3() for b in arm.bones}
rest_pos = {b.name: b.matrix_local.to_translation().copy() for b in arm.bones}
order = []
seen = set()
def visit(b):
    if b.name in seen: return
    if b.parent: visit(b.parent)
    seen.add(b.name)
    order.append(b.name)
for b in arm.bones:
    visit(b)

for name in order:
    obj.pose.bones[name].rotation_mode = "QUATERNION"

if obj.animation_data is None:
    obj.animation_data_create()
act = bpy.data.actions.new("Walk")
obj.animation_data.action = act

def rot_xyz(rx_deg, ry_deg, rz_deg):
    return (
        Matrix.Rotation(math.radians(rz_deg), 3, "Z")
        @ Matrix.Rotation(math.radians(ry_deg), 3, "Y")
        @ Matrix.Rotation(math.radians(rx_deg), 3, "X")
    )

N_FRAMES = 48
scene.frame_start = 1
scene.frame_end = N_FRAMES

for f in range(1, N_FRAMES + 1):
    ph = 2.0 * math.pi * (f - 1) / (N_FRAMES - 1)
    s = math.sin(ph)
    c = math.cos(ph)
    s2 = math.sin(2.0 * ph)
    c2 = math.cos(2.0 * ph)

    hip_z_bob = 0.032 * (-c2)
    hip_x_sway = 0.025 * s
    hip_yaw = 8.0 * s
    hip_roll = 4.0 * s

    l_thigh_x = -26.0 * s - 4.0
    r_thigh_x =  26.0 * s - 4.0
    l_knee_x = max(0.0, 34.0 * math.sin(ph - 0.85)) + 6.0
    r_knee_x = max(0.0, 34.0 * math.sin(ph + math.pi - 0.85)) + 6.0
    l_foot_x = -0.35 * l_knee_x + 8.0 * c
    r_foot_x = -0.35 * r_knee_x - 8.0 * c

    l_arm_x =  22.0 * s
    r_arm_x = -22.0 * s
    l_fore_x = -14.0 - 10.0 * max(0.0, -s)
    r_fore_x = -14.0 - 10.0 * max(0.0,  s)

    deltas = {
        "mixamorig:Hips":         rot_xyz(3.0 * c2, hip_roll, hip_yaw),
        "mixamorig:Spine":        rot_xyz(-2.0 * c2, -2.5 * s, -4.5 * s),
        "mixamorig:Spine2":       rot_xyz(-1.5 * c2, -2.0 * s, -6.0 * s),
        "mixamorig:Neck":         rot_xyz(1.0 * c2, 1.0 * s, 2.0 * s),
        "mixamorig:Head":         rot_xyz(1.5 * c2, 1.0 * s, 2.5 * s),
        "mixamorig:LeftShoulder": rot_xyz(0.0, 0.0, -3.0 * s),
        "mixamorig:RightShoulder":rot_xyz(0.0, 0.0, -3.0 * s),
        "mixamorig:LeftArm":      rot_xyz(l_arm_x, 0.0, -4.0 * s),
        "mixamorig:LeftForeArm":  rot_xyz(l_arm_x + l_fore_x, 0.0, 0.0),
        "mixamorig:LeftHand":     rot_xyz(l_arm_x + l_fore_x, 0.0, 0.0),
        "mixamorig:RightArm":     rot_xyz(r_arm_x, 0.0, -4.0 * s),
        "mixamorig:RightForeArm": rot_xyz(r_arm_x + r_fore_x, 0.0, 0.0),
        "mixamorig:RightHand":    rot_xyz(r_arm_x + r_fore_x, 0.0, 0.0),
        "mixamorig:LeftUpLeg":    rot_xyz(l_thigh_x, 0.0, 0.0),
        "mixamorig:LeftLeg":      rot_xyz(l_thigh_x + l_knee_x, 0.0, 0.0),
        "mixamorig:LeftFoot":     rot_xyz(l_thigh_x + l_knee_x + l_foot_x, 0.0, 0.0),
        "mixamorig:RightUpLeg":   rot_xyz(r_thigh_x, 0.0, 0.0),
        "mixamorig:RightLeg":     rot_xyz(r_thigh_x + r_knee_x, 0.0, 0.0),
        "mixamorig:RightFoot":    rot_xyz(r_thigh_x + r_knee_x + r_foot_x, 0.0, 0.0),
    }

    scene.frame_set(f)
    for name in order:
        pb = obj.pose.bones[name]
        target_r = deltas[name] @ rest_w[name]
        head_pos = pb.matrix.to_translation()
        if name == "mixamorig:Hips":
            head_pos = rest_pos[name] + Vector((hip_x_sway, 0.0, hip_z_bob))
        m = target_r.to_4x4()
        m.translation = head_pos
        pb.matrix = m
        bpy.context.view_layer.update()
        pb.keyframe_insert(data_path="rotation_quaternion", frame=f)
        if name == "mixamorig:Hips":
            pb.keyframe_insert(data_path="location", frame=f)

os.makedirs(os.path.dirname(out_fbx), exist_ok=True)
bpy.ops.export_scene.fbx(
    filepath=out_fbx,
    use_selection=True,
    add_leaf_bones=False,
    bake_anim=True,
    bake_anim_use_all_bones=True,
    bake_anim_step=1.0,
)
print("WROTE:", out_fbx)
"""


def main():
    blender = find_blender()
    if not blender:
        sys.exit("Blender not found; install Blender or set BLENDER_BIN")
    out_fbx = Path(__file__).resolve().parent / "presets" / "walk.fbx"
    out_fbx.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpd:
        script_path = os.path.join(tmpd, "gen_walk.py")
        with open(script_path, "w", encoding="utf-8") as tf:
            tf.write(_GEN_WALK_SCRIPT)
        proc = subprocess.run(
            [blender, "--background", "--factory-startup",
             "--python", script_path, "--", str(out_fbx)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=60,
        )
        if proc.returncode != 0 or not out_fbx.exists():
            print(proc.stdout)
            print(proc.stderr)
            sys.exit(1)
        print(f"Generated {out_fbx} ({out_fbx.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
