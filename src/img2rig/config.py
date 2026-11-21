"""Cross-platform path, device, model, and Blender configuration for dcd-img2rig."""

from __future__ import annotations

import glob
import os
import shutil
import sys
from pathlib import Path


def _resolve_root() -> Path:
    env_root = os.environ.get("IMG2RIG_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    # When running from source tree: <root>/src/img2rig/config.py
    pkg_root = Path(__file__).resolve().parents[2]
    if (pkg_root / "viewer").is_dir() and (pkg_root / "assets").is_dir():
        return pkg_root
    return Path.cwd().resolve()


ROOT: Path = _resolve_root()
SRC_DIR: Path = ROOT / "src"
VIEWER_DIR: Path = ROOT / "viewer"
ASSETS_DIR: Path = ROOT / "assets"
RIGS_DIR: Path = ASSETS_DIR / "rigs"
DEFAULT_RIG_GLB: Path = RIGS_DIR / "sweetie_rig.glb"
PRESETS_DIR: Path = RIGS_DIR / "presets"

DATA_DIR: Path = Path(os.environ.get("IMG2RIG_DATA_DIR", str(ROOT))).expanduser().resolve()
OUTPUTS_DIR: Path = Path(os.environ.get("IMG2RIG_OUTPUTS_DIR", str(DATA_DIR / "outputs"))).expanduser().resolve()
UPLOADS_DIR: Path = Path(os.environ.get("IMG2RIG_UPLOADS_DIR", str(DATA_DIR / "uploads"))).expanduser().resolve()
ENCODERS_DIR: Path = Path(os.environ.get("IMG2RIG_ENCODERS_DIR", os.environ.get("IDOL_ENCODERS_DIR", str(ROOT / "encoders")))).expanduser().resolve()


def find_blender() -> str | None:
    """Return path to the Blender CLI executable on Windows, macOS, or Linux, or None."""
    env_bin = os.environ.get("BLENDER_BIN")
    candidates: list[str] = []
    if env_bin:
        candidates.append(env_bin)

    which_bin = shutil.which("blender")
    if which_bin:
        candidates.append(which_bin)

    if sys.platform == "win32":
        which_exe = shutil.which("blender.exe")
        if which_exe:
            candidates.append(which_exe)
        prog_dirs = [
            os.environ.get("ProgramFiles", r"C:\Program Files"),
            os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs"),
        ]
        for base in prog_dirs:
            if not base:
                continue
            pattern = os.path.join(base, "Blender Foundation", "Blender *", "blender.exe")
            candidates.extend(sorted(glob.glob(pattern), reverse=True))
            candidates.append(os.path.join(base, "Blender Foundation", "Blender", "blender.exe"))
        # Steam install fallback on Windows
        steam_cand = os.path.join(
            os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
            "Steam", "steamapps", "common", "Blender", "blender.exe",
        )
        candidates.append(steam_cand)
    elif sys.platform == "darwin":
        candidates.extend([
            "/opt/homebrew/bin/blender",
            "/usr/local/bin/blender",
            "/Applications/Blender.app/Contents/MacOS/Blender",
        ])
    else:
        candidates.extend([
            "/usr/bin/blender",
            "/usr/local/bin/blender",
            "/snap/bin/blender",
            os.path.expanduser("~/.local/bin/blender"),
        ])

    for cand in candidates:
        if cand and os.path.isfile(cand) and os.access(cand, os.X_OK):
            return os.path.abspath(cand)
    return None


def resolve_idol_weights() -> str:
    """Return local Idol weights dir if configured/present, else Hub repo id 'speridlabs/idol'."""
    env_w = os.environ.get("IDOL_WEIGHTS", "").strip()
    if env_w:
        return str(Path(env_w).expanduser().resolve())
    local_dir = ROOT / "models" / "idol"
    if (local_dir / "config.json").is_file():
        return str(local_dir)
    return "speridlabs/idol"


def resolve_image_encoders() -> dict[str, str] | None:
    """Return local encoder overrides if present under ENCODERS_DIR, else None (Hub default)."""
    overrides: dict[str, str] = {}
    dinov3_dir = ENCODERS_DIR / "dinov3-h16plus"
    if (dinov3_dir / "config.json").is_file():
        overrides["dinov3"] = str(dinov3_dir)
    flux_dir = ENCODERS_DIR / "flux2-vae-klein"
    if (flux_dir / "vae").is_dir() or (flux_dir / "config.json").is_file():
        overrides["flux2_vae"] = str(flux_dir)
    return overrides or None


def resolve_device_and_compile(
    requested_device: str | None = None,
    requested_dtype: str | None = None,
    no_compile: bool = False,
    force_compile: bool = False,
) -> tuple[str, str, bool]:
    """Choose (device, dinov3_dtype, optimize) safely across Windows/macOS/Linux."""
    import torch

    dev = (requested_device or os.environ.get("IDOL_DEVICE", "")).strip().lower()
    if not dev:
        if torch.cuda.is_available():
            dev = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            dev = "mps"
        else:
            dev = "cpu"

    dtype = (requested_dtype or os.environ.get("IDOL_DINOV3_DTYPE", "")).strip().lower()
    if not dtype:
        if dev == "cpu":
            dtype = "fp32"
        elif dev == "cuda":
            major, _ = torch.cuda.get_device_capability(0)
            dtype = "bf16" if major >= 8 else "fp16"
        else:
            dtype = "bf16"

    if no_compile:
        optimize = False
    elif force_compile or os.environ.get("IDOL_COMPILE", "").strip().lower() in ("1", "true", "yes"):
        optimize = dev == "cuda"
    else:
        # torch.compile often fails on Windows without Triton/MSVC; opt-in on Windows,
        # enabled on Linux CUDA by default.
        optimize = (dev == "cuda") and (sys.platform != "win32")

    return dev, dtype, optimize


def subprocess_env() -> dict[str, str]:
    """Return environment with UTF-8 stdio and src/ on PYTHONPATH for child python processes."""
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    src_str = str(SRC_DIR)
    existing = env.get("PYTHONPATH", "")
    if existing:
        parts = existing.split(os.pathsep)
        if src_str not in parts:
            env["PYTHONPATH"] = src_str + os.pathsep + existing
    else:
        env["PYTHONPATH"] = src_str
    return env
