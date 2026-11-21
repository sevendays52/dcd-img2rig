#!/usr/bin/env python3
"""Cross-platform local web studio server for dcd-img2rig.

- Serves the WebGL2 3DGS + FBX/Skinning Studio UI at http://localhost:8765
- Lists generated assets in OUTPUTS_DIR (/api/plys)
- Runs Idol inference (/api/upload), mesh/FBX conversion (/api/fbx),
  auto-rigging (/api/rig), and Mixamo motion retargeting (/api/motion, /api/motion/preset)
"""

from __future__ import annotations

import argparse
from email.parser import BytesParser
import base64
import http.server
import json
import os
import re
import socketserver
import subprocess
import sys
import threading
import time
import urllib.parse

from . import config

ROOT = str(config.ROOT)
VIEWER_DIR = str(config.VIEWER_DIR)
OUTPUTS_DIR = str(config.OUTPUTS_DIR)
UPLOADS_DIR = str(config.UPLOADS_DIR)

PREVIEW_FRAMES = 16
PREVIEW_FPS = 12
PREVIEW_SIZE = 256

_jobs: dict[str, dict] = {}
_lock = threading.Lock()
_cond = threading.Condition(_lock)
_seq = 0
_worker: threading.Thread | None = None
PROG_RE = re.compile(r"(\d{1,3})%")


def _to_posix_url_path(full_path: str, base_dir: str, url_prefix: str) -> str:
    rel = os.path.relpath(full_path, base_dir).replace(os.sep, "/")
    return f"{url_prefix}/{rel}"


def _read_ply_counts(ply_path: str):
    """Quickly read vertex & face counts from a PLY header without loading body."""
    verts = faces = None
    try:
        with open(ply_path, "rb") as fh:
            for _ in range(40):
                line = fh.readline()
                if not line:
                    break
                s = line.decode("ascii", "ignore").strip()
                if s.startswith("element vertex "):
                    verts = int(s.split()[-1])
                elif s.startswith("element face "):
                    faces = int(s.split()[-1])
                elif s == "end_header":
                    break
    except OSError:
        pass
    return verts, faces


def _find_skinned_glb(stem_path: str) -> str | None:
    exact = stem_path + "_skinned.glb"
    return exact if os.path.exists(exact) else None


def _find_anim_glbs(stem_path: str) -> list[dict[str, str]]:
    base = os.path.basename(stem_path)
    d = os.path.dirname(stem_path)
    prefix = base + "_anim_"
    out = []
    try:
        for f in sorted(os.listdir(d)):
            if f.startswith(prefix) and f.endswith(".glb"):
                anim_name = f[len(prefix):-4]
                full = os.path.join(d, f)
                out.append({
                    "name": anim_name,
                    "path": _to_posix_url_path(full, OUTPUTS_DIR, "outputs"),
                })
    except OSError:
        pass
    return out


def _read_mesh_info(stem_path: str, mesh_ply: str, fbx: str):
    mesh_json = stem_path + "_mesh.json"
    info = {}
    if os.path.exists(mesh_json):
        try:
            with open(mesh_json, "r", encoding="utf-8") as fh:
                info = json.load(fh)
        except Exception:
            info = {}

    has_mesh = os.path.exists(mesh_ply)
    has_fbx = os.path.exists(fbx)
    if not info and not (has_mesh or has_fbx):
        return None

    if has_mesh and (info.get("verts") is None or info.get("faces") is None):
        v, f = _read_ply_counts(mesh_ply)
        if v is not None:
            info.setdefault("verts", v)
        if f is not None:
            info.setdefault("faces", f)
    if has_fbx and info.get("fbx_bytes") is None:
        try:
            info["fbx_bytes"] = os.path.getsize(fbx)
        except OSError:
            pass
    return info or None


def find_plys() -> list[dict]:
    found = []
    if not os.path.isdir(OUTPUTS_DIR):
        return found
    for dirpath, _dirnames, filenames in os.walk(OUTPUTS_DIR):
        for f in filenames:
            if not f.lower().endswith(".ply") or f.endswith("_mesh.ply"):
                continue
            full = os.path.join(dirpath, f)
            stem_path = full[:-4]
            meta_path = stem_path + ".json"
            cond_path = stem_path + "_condition.png"
            prev_gif = stem_path + "_preview.gif"
            prev_png = stem_path + "_preview.png"
            prev_path = prev_gif if os.path.exists(prev_gif) else prev_png
            mesh_ply = stem_path + "_mesh.ply"
            mesh_uv = stem_path + "_mesh_uv.bin"
            tex_png = stem_path + "_texture.png"
            mesh_prev = stem_path + "_mesh_preview.gif"
            fbx = stem_path + ".fbx"
            skin_glb = _find_skinned_glb(stem_path)
            skin_fbx = stem_path + "_skinned.fbx"
            mesh_info = _read_mesh_info(stem_path, mesh_ply, fbx)

            def opt_url(p: str | None) -> str | None:
                return _to_posix_url_path(p, OUTPUTS_DIR, "outputs") if p and os.path.exists(p) else None

            found.append({
                "name": f,
                "path": _to_posix_url_path(full, OUTPUTS_DIR, "outputs"),
                "size": os.path.getsize(full),
                "meta": opt_url(meta_path),
                "condition": opt_url(cond_path),
                "preview": opt_url(prev_path),
                "mesh_ply": opt_url(mesh_ply),
                "mesh_uv": opt_url(mesh_uv),
                "texture": opt_url(tex_png),
                "mesh_preview": opt_url(mesh_prev),
                "fbx": opt_url(fbx),
                "skin_glb": opt_url(skin_glb),
                "skinned_fbx": opt_url(skin_fbx),
                "animations": _find_anim_glbs(stem_path),
                "mesh_info": mesh_info,
            })
    found.sort(key=lambda p: p["name"])
    return found


def _render_preview(stem: str, timeout: int = 180) -> bool:
    ply = os.path.join(OUTPUTS_DIR, stem + ".ply")
    if not os.path.exists(ply):
        return False
    meta = os.path.join(OUTPUTS_DIR, stem + ".json")
    out = os.path.join(OUTPUTS_DIR, stem + "_preview.gif")
    cmd = [
        sys.executable, "-m", "img2rig.preview",
        "--ply", ply,
        "--out", out,
        "--width", str(PREVIEW_SIZE), "--height", str(PREVIEW_SIZE),
        "--frames", str(PREVIEW_FRAMES), "--fps", str(PREVIEW_FPS),
    ]
    if os.path.exists(meta):
        cmd += ["--meta", meta]
    try:
        proc = subprocess.run(
            cmd, cwd=ROOT, env=config.subprocess_env(),
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
        )
        ok = proc.returncode == 0 and os.path.exists(out)
        if not ok:
            print(f"[preview] {stem} 렌더링 실패: "
                  f"{(proc.stderr or proc.stdout or '').strip()[-300:]}", file=sys.stderr)
            return False
        old_png = os.path.join(OUTPUTS_DIR, stem + "_preview.png")
        if os.path.exists(old_png):
            try:
                os.remove(old_png)
            except OSError:
                pass
    except Exception as e:
        print(f"[preview] {stem} 렌더링 오류: {e}", file=sys.stderr)
        return False
    return True


def _backfill_previews():
    pending = []
    for p in find_plys():
        stem = os.path.splitext(p["name"])[0]
        if os.path.exists(os.path.join(OUTPUTS_DIR, stem + "_preview.gif")):
            continue
        pending.append(stem)
    if not pending:
        return
    print(f"  미리보기 백필 대상 {len(pending)}개 (백그라운드 렌더링)…", file=sys.stderr)
    for stem in pending:
        t0 = time.time()
        if _render_preview(stem):
            print(f"  [preview] {stem} 렌더링 완료 ({time.time() - t0:.1f}s)", file=sys.stderr)
        time.sleep(0.2)


def _resolve_ply(name: str) -> str | None:
    if not name or "/" in name or "\\" in name or os.path.basename(name) != name:
        return None
    if not name.lower().endswith(".ply") or name.startswith("."):
        return None
    outputs_real = os.path.realpath(OUTPUTS_DIR)
    full = os.path.realpath(os.path.join(outputs_real, name))
    if os.path.commonpath([outputs_real, full]) != outputs_real:
        return None
    return full


def _parse_progress(line: str, job: dict):
    if "Sampling:" in line:
        if job["phase"] != "sampling":
            job["phase"] = "sampling"
            job["pct"] = 0
        m = PROG_RE.search(line)
        if m:
            pct = int(m.group(1))
            if 0 <= pct <= 100 and pct > job["pct"]:
                job["pct"] = pct
    elif "[load]" in line:
        job["phase"] = "sampling"
    elif "Loading weights" in line and job["phase"] in ("queued", "running"):
        job["phase"] = "loading"


def _worker_loop():
    while True:
        with _cond:
            while True:
                stem = next((s for s, j in _jobs.items() if j["status"] == "queued"), None)
                if stem is not None:
                    break
                _cond.wait(2.0)
        if stem is None:
            continue
        _run_job(stem)


def _gc_jobs():
    now = time.time()
    with _lock:
        stale = [
            k for k, v in _jobs.items()
            if v["status"] in ("done", "error") and now - v.get("_finished_at", 0) > 600
        ]
        for k in stale:
            _jobs.pop(k, None)


def _run_job(stem: str):
    with _lock:
        _kind = _jobs.get(stem, {}).get("kind")
    if _kind in ("probe", "convert"):
        _run_fbx_job(stem)
        return
    t0 = time.time()
    with _lock:
        j = _jobs.get(stem)
        if j is None:
            return
        img_path = j["img"]
        mode = j["mode"]
        gauss = j["gaussians"]
        j["status"] = "running"
        j["phase"] = "loading"
        j["_started_at"] = t0

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    cmd = [
        sys.executable, "-m", "img2rig.inference",
        "--image", img_path,
        "--mode", mode,
        "--num_gaussians", str(gauss),
        "--out", OUTPUTS_DIR,
    ]
    env = config.subprocess_env()
    tail: list[str] = []
    try:
        proc = subprocess.Popen(
            cmd, cwd=ROOT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        buf = b""
        assert proc.stdout is not None
        while True:
            b = proc.stdout.read(1)
            if not b:
                break
            buf += b
            if b in (b"\r", b"\n"):
                line = buf.decode("utf-8", "replace").strip()
                buf = b""
                if line:
                    tail = (tail + [line])[-5:]
                    with _lock:
                        if stem in _jobs:
                            _parse_progress(line, _jobs[stem])
        proc.wait()
        rc = proc.returncode
        ply_path = os.path.join(OUTPUTS_DIR, stem + ".ply")
        if rc == 0 and os.path.exists(ply_path):
            with _lock:
                if stem in _jobs:
                    _jobs[stem].update(phase="preview")
            _render_preview(stem)
        with _lock:
            if stem in _jobs:
                if rc == 0 and os.path.exists(ply_path):
                    _jobs[stem].update(
                        status="done", pct=100, phase="done",
                        ply=f"outputs/{stem}.ply",
                        seconds=round(time.time() - t0, 1),
                        _finished_at=time.time(),
                    )
                else:
                    detail = tail[-1] if tail else ""
                    msg = f"변환 실패 (코드 {rc})" + (f": {detail}" if detail else ". 서버 로그를 확인하세요.")
                    _jobs[stem].update(
                        status="error", msg=msg,
                        seconds=round(time.time() - t0, 1),
                        _finished_at=time.time(),
                    )
    except Exception as e:
        with _lock:
            if stem in _jobs:
                _jobs[stem].update(
                    status="error", msg=f"서버 오류: {e}",
                    seconds=round(time.time() - t0, 1),
                    _finished_at=time.time(),
                )


def _fbx_run_cmd(cmd: list[str], timeout: int = 600):
    env = config.subprocess_env()
    tail: list[str] = []
    try:
        proc = subprocess.Popen(
            cmd, cwd=ROOT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        buf = b""
        assert proc.stdout is not None
        while True:
            b = proc.stdout.read(1)
            if not b:
                break
            buf += b
            if b in (b"\r", b"\n"):
                line = buf.decode("utf-8", "replace").strip()
                buf = b""
                if line:
                    tail = (tail + [line])[-5:]
        proc.wait()
        rc = proc.returncode
        info = None
        for line in reversed(tail):
            if line.startswith("{") and line.endswith("}"):
                info = json.loads(line)
                break
        return rc, info, tail
    except Exception:
        return -1, None, tail


def _render_mesh_preview(mesh_ply: str, out_gif: str, timeout: int = 180) -> bool:
    if not os.path.exists(mesh_ply):
        return False
    cmd = [
        sys.executable, "-m", "img2rig.preview",
        "--ply", mesh_ply, "--mesh",
        "--out", out_gif,
        "--width", str(PREVIEW_SIZE), "--height", str(PREVIEW_SIZE),
        "--frames", str(PREVIEW_FRAMES), "--fps", str(PREVIEW_FPS),
    ]
    try:
        proc = subprocess.run(
            cmd, cwd=ROOT, env=config.subprocess_env(),
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
        )
        ok = proc.returncode == 0 and os.path.exists(out_gif)
        if not ok:
            print(f"[mesh-preview] {os.path.basename(mesh_ply)} 렌더링 실패: "
                  f"{(proc.stderr or proc.stdout or '').strip()[-300:]}", file=sys.stderr)
            return False
    except Exception as e:
        print(f"[mesh-preview] {os.path.basename(mesh_ply)} 오류: {e}", file=sys.stderr)
        return False
    return True


def _run_fbx_job(stem: str):
    t0 = time.time()
    with _lock:
        j = _jobs.get(stem)
        if j is None:
            return
        kind = j["kind"]
        ply_name = j["ply_name"]
        j["status"] = "running"
        j["phase"] = "queued"
        j["_started_at"] = t0
    ply = os.path.join(OUTPUTS_DIR, ply_name)
    out_stem = os.path.join(OUTPUTS_DIR, os.path.splitext(ply_name)[0])
    if not os.path.exists(ply):
        with _lock:
            if stem in _jobs:
                _jobs[stem].update(
                    status="error", phase="error",
                    msg="원본 PLY를 찾을 수 없어요",
                    seconds=round(time.time() - t0, 1), _finished_at=time.time(),
                )
        return
    quality = j.get("quality") or "standard"
    method = j.get("method") or "marching_cubes"
    clean = bool(j.get("clean", True))
    smooth = bool(j.get("smooth", True))
    fill_holes = bool(j.get("fill_holes", True))
    opacity = j.get("opacity", 0.3)

    common_flags = [
        "--opacity", str(opacity),
        "--quality", quality,
        "--method", method,
    ]
    if not clean:
        common_flags.append("--no_clean")
    if not smooth:
        common_flags.append("--no_smooth")
    if not fill_holes:
        common_flags.append("--no_fill_holes")

    base_cmd = [sys.executable, "-m", "img2rig.mesh_convert"]

    if kind == "probe":
        cmd = [*base_cmd, "--ply", ply, "--probe", *common_flags]
        rc, info, tail = _fbx_run_cmd(cmd, timeout=600)
        with _lock:
            if stem in _jobs:
                if rc == 0 and info:
                    _jobs[stem].update(
                        status="done", phase="done", pct=100,
                        base_faces=info.get("base_faces"),
                        base_verts=info.get("base_verts"),
                        kept_points=info.get("kept_points"),
                        quality=info.get("quality", quality),
                        method=info.get("method", method),
                        clean=info.get("clean", clean),
                        smooth=info.get("smooth", smooth),
                        opacity=opacity,
                        seconds=round(time.time() - t0, 1), _finished_at=time.time(),
                    )
                else:
                    _jobs[stem].update(
                        status="error", phase="error",
                        msg=(tail[-1] if tail else f"실패 (코드 {rc})"),
                        seconds=round(time.time() - t0, 1), _finished_at=time.time(),
                    )
        return

    target = j.get("target_faces")
    rig = bool(j.get("rig", False))
    rig_path = str(config.DEFAULT_RIG_GLB)
    cmd = [*base_cmd, "--ply", ply, "--out", out_stem, "--target_faces", str(target or 0), *common_flags]
    if rig and os.path.exists(rig_path):
        cmd.extend(["--rig", rig_path])

    with _lock:
        if stem in _jobs:
            _jobs[stem].update(phase="reconstruct", pct=20)
    rc, info, tail = _fbx_run_cmd(cmd, timeout=1200)
    mesh_ply = out_stem + "_mesh.ply"
    mesh_uv = out_stem + "_mesh_uv.bin"
    tex_png = out_stem + "_texture.png"
    fbx = out_stem + ".fbx"
    sk_glb = out_stem + "_skinned.glb"
    sk_fbx = out_stem + "_skinned.fbx"
    preview = out_stem + "_mesh_preview.gif"
    previewed = False
    if rc == 0 and os.path.exists(mesh_ply):
        with _lock:
            if stem in _jobs:
                _jobs[stem].update(phase="preview", pct=75)
        previewed = _render_mesh_preview(mesh_ply, preview)

    def opt_url(p: str) -> str | None:
        return _to_posix_url_path(p, OUTPUTS_DIR, "outputs") if os.path.exists(p) else None

    with _lock:
        if stem in _jobs:
            if rc == 0 and info:
                _jobs[stem].update(
                    status="done", phase="done", pct=100,
                    fbx=opt_url(fbx),
                    skin_glb=opt_url(sk_glb),
                    skinned_fbx=opt_url(sk_fbx),
                    rig=bool(info.get("rig")),
                    mesh_ply=opt_url(mesh_ply),
                    mesh_uv=opt_url(mesh_uv),
                    texture=opt_url(tex_png),
                    mesh_preview=opt_url(preview) if previewed else None,
                    has_texture=bool(info.get("has_texture")),
                    tex_size=info.get("tex_size"),
                    base_faces=info.get("base_faces"),
                    faces=info.get("faces"),
                    verts=info.get("verts"),
                    fbx_bytes=info.get("fbx_bytes"),
                    verified_by=info.get("verified_by"),
                    bounds=info.get("bounds"),
                    quality=info.get("quality", quality),
                    method=info.get("method", method),
                    clean=info.get("clean", clean),
                    smooth=info.get("smooth", smooth),
                    fill_holes=info.get("fill_holes", fill_holes),
                    opacity=info.get("opacity", opacity),
                    target_faces=info.get("target_faces", target or 0),
                    seconds=round(time.time() - t0, 1), _finished_at=time.time(),
                )
            else:
                _jobs[stem].update(
                    status="error", phase="error",
                    msg=(tail[-1] if tail else f"변환 실패 (코드 {rc})"),
                    seconds=round(time.time() - t0, 1), _finished_at=time.time(),
                )


def _start_worker():
    global _worker
    with _lock:
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(target=_worker_loop, daemon=True)
            _worker.start()


def _parse_upload_body(content_type: str, raw_body: bytes):
    if content_type.startswith("multipart/form-data"):
        header = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8")
        msg = BytesParser().parsebytes(header + raw_body)
        if not msg.is_multipart():
            raise ValueError("잘못된 multipart 요청 형식")
        fields = {}
        img_bytes = b""
        filename = ""
        for part in msg.get_payload():
            name = part.get_param("name", header="content-disposition")
            if not name:
                continue
            payload = part.get_payload(decode=True) or b""
            if name in ("file", "image"):
                img_bytes = payload
                filename = part.get_filename() or ""
            else:
                fields[name] = payload.decode("utf-8", "replace")
        stem = fields.get("stem") or os.path.splitext(os.path.basename(filename))[0]
        return (
            img_bytes,
            fields.get("mode") or "synthetic",
            int(fields.get("gaussians") or 262144),
            stem,
        )

    data = json.loads(raw_body.decode("utf-8"))
    b64 = data.get("image") or ""
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    img_bytes = base64.b64decode(b64)
    return (
        img_bytes,
        data.get("mode") or "synthetic",
        int(data.get("gaussians") or 262144),
        data.get("stem") or "",
    )


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".js": "application/javascript; charset=utf-8",
        ".mjs": "application/javascript; charset=utf-8",
        ".wasm": "application/wasm",
        ".json": "application/json; charset=utf-8",
        ".css": "text/css; charset=utf-8",
        ".html": "text/html; charset=utf-8",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)

    def translate_path(self, path: str) -> str:
        clean = urllib.parse.unquote(path.split("?", 1)[0].split("#", 1)[0]).lstrip("/")
        mappings = (
            ("viewer/", VIEWER_DIR),
            ("outputs/", OUTPUTS_DIR),
            ("uploads/", UPLOADS_DIR),
        )
        for prefix, base_dir in mappings:
            if clean.startswith(prefix):
                rel = clean[len(prefix):]
                base_real = os.path.realpath(base_dir)
                target = os.path.realpath(os.path.join(base_real, *rel.split("/")))
                if os.path.commonpath([base_real, target]) == base_real:
                    return target
                return os.path.join(base_real, "__forbidden__")
        return super().translate_path(path)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _send_json(self, obj, status: int = 200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _job_view(self, j: dict) -> dict:
        seconds = j.get("seconds")
        if j["status"] == "running" and "_started_at" in j:
            seconds = round(time.time() - j["_started_at"], 1)
        return {
            "stem": j["stem"], "mode": j.get("mode"), "kind": j.get("kind"),
            "status": j["status"], "pct": j["pct"], "phase": j["phase"],
            "ply": j.get("ply"), "msg": j.get("msg"), "seconds": seconds,
            "base_faces": j.get("base_faces"), "base_verts": j.get("base_verts"),
            "kept_points": j.get("kept_points"),
            "fbx": j.get("fbx"), "mesh_ply": j.get("mesh_ply"),
            "mesh_uv": j.get("mesh_uv"), "texture": j.get("texture"),
            "has_texture": j.get("has_texture"), "tex_size": j.get("tex_size"),
            "mesh_preview": j.get("mesh_preview"), "faces": j.get("faces"),
            "verts": j.get("verts"), "fbx_bytes": j.get("fbx_bytes"),
            "verified_by": j.get("verified_by"), "bounds": j.get("bounds"),
            "quality": j.get("quality"), "method": j.get("method"),
            "clean": j.get("clean"), "smooth": j.get("smooth"),
            "fill_holes": j.get("fill_holes"),
            "opacity": j.get("opacity"), "target_faces": j.get("target_faces"),
            "skin_glb": j.get("skin_glb"), "skinned_fbx": j.get("skinned_fbx"),
            "rig": j.get("rig"),
        }

    def do_GET(self):
        path = self.path.split("?", 1)[0].lstrip("/")
        if path in ("", "index.html", "viewer/index.html"):
            fp = os.path.join(VIEWER_DIR, "index.html")
            with open(fp, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "api/plys":
            self._send_json({"plys": find_plys()})
        elif path == "api/job":
            qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            want_stem = (qs.get("stem") or [""])[0]
            with _lock:
                j = _jobs.get(want_stem) if want_stem else next(iter(_jobs.values()), None)
                data = self._job_view(j) if j else None
            self._send_json({"job": data})
        else:
            super().do_GET()

    def _handle_fbx(self):
        try:
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n > 0 else b"{}"
            data = json.loads(raw.decode("utf-8") or "{}")
        except Exception:
            self._send_json({"error": "잘못된 요청 형식"}, 400)
            return
        ply = data.get("ply") or ""
        action = data.get("action") or ""
        if action not in ("probe", "convert"):
            self._send_json({"error": "action이 올바르지 않아요 (probe|convert)"}, 400)
            return
        full = _resolve_ply(ply)
        if full is None:
            self._send_json({"error": "파일이 없거나 이름이 올바르지 않아요"}, 400)
            return
        ply_name = os.path.basename(full)
        stem_base = os.path.splitext(ply_name)[0]
        target = data.get("target_faces")
        try:
            target = max(0, int(target)) if target not in (None, "") else 0
        except (TypeError, ValueError):
            target = 0
        try:
            opacity = min(0.95, max(0.05, float(data.get("opacity", 0.3))))
        except (TypeError, ValueError):
            opacity = 0.3
        quality = data.get("quality") or "standard"
        if quality not in ("fast", "standard", "high"):
            quality = "standard"
        method = data.get("method") or "marching_cubes"
        if method not in ("marching_cubes", "poisson", "surface_fit"):
            method = "marching_cubes"
        clean = bool(data.get("clean", True))
        smooth = bool(data.get("smooth", True))
        fill_holes = bool(data.get("fill_holes", True))
        rig = bool(data.get("rig", False))

        _gc_jobs()
        with _lock:
            existing = next(
                (
                    s for s, j in _jobs.items()
                    if j.get("ply_name") == ply_name
                    and j.get("kind") == action
                    and j.get("status") in ("queued", "running")
                ),
                None,
            )
        if existing:
            self._send_json({"stem": existing, "reused": True})
            return

        ts = time.strftime("%Y%m%d-%H%M%S")
        global _seq
        with _lock:
            _seq += 1
            seq = _seq
        job_stem = f"{stem_base}-{action}-{ts}-{seq:02d}"
        with _lock:
            _jobs[job_stem] = {
                "stem": job_stem, "kind": action, "ply_name": ply_name,
                "target_faces": target, "opacity": opacity,
                "quality": quality, "method": method, "clean": clean, "smooth": smooth,
                "fill_holes": fill_holes, "rig": rig,
                "status": "queued", "pct": 0, "phase": "queued",
                "msg": None, "seconds": None,
            }
        _start_worker()
        with _cond:
            _cond.notify_all()
        self._send_json({"stem": job_stem})

    def _run_retarget_cmd(self, sk_glb: str, motion_path: str, out_anim: str, timeout: int = 240) -> dict:
        proc = subprocess.run(
            [
                sys.executable, "-m", "img2rig.motion_retarget",
                "--skinned", sk_glb, "--motion", motion_path, "--out", out_anim,
            ],
            cwd=ROOT, env=config.subprocess_env(),
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
        )
        if proc.returncode != 0 or not os.path.exists(out_anim):
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-5:]
            raise RuntimeError("모션 리타겟팅 실패: " + " / ".join(tail))
        return json.loads(proc.stdout.strip().splitlines()[-1])

    def _auto_rig_mesh(self, stem_path: str, custom_joints=None) -> dict:
        mesh_ply = stem_path + "_mesh.ply"
        if not os.path.exists(mesh_ply):
            raise FileNotFoundError("먼저 메시(FBX) 변환을 실행해 주세요")
        rig_glb = str(config.DEFAULT_RIG_GLB)
        if not os.path.exists(rig_glb):
            raise FileNotFoundError("리그 파일(assets/rigs/sweetie_rig.glb)이 없습니다")

        sk_glb = stem_path + "_skinned.glb"
        sk_fbx = stem_path + "_skinned.fbx"
        mesh_name = os.path.basename(stem_path)
        cmd = [
            sys.executable, "-m", "img2rig.rig_bind",
            "--mesh_ply", mesh_ply,
            "--rig", rig_glb,
            "--out_glb", sk_glb,
            "--out_fbx", sk_fbx,
            "--mesh_name", mesh_name,
            "--meta_json", stem_path + "_mesh.json",
            "--src_json", stem_path + ".json",
        ]
        if custom_joints:
            cmd.extend(["--joints_json", json.dumps(custom_joints)])

        proc = subprocess.run(
            cmd, cwd=ROOT, env=config.subprocess_env(),
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180,
        )
        if proc.returncode != 0 or not os.path.exists(sk_glb):
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-5:]
            raise RuntimeError("자동 리깅 실패: " + " / ".join(tail))
        rig_info = json.loads(proc.stdout.strip().splitlines()[-1])

        retargeted = []
        base = os.path.basename(stem_path)
        for anim in _find_anim_glbs(stem_path):
            aname = anim["name"]
            out_anim = os.path.join(OUTPUTS_DIR, os.path.basename(anim["path"]))
            src_motion = None
            preset_cand = os.path.join(str(config.PRESETS_DIR), f"{aname}.fbx")
            if os.path.exists(preset_cand):
                src_motion = preset_cand
            else:
                mdir = os.path.join(UPLOADS_DIR, "motions")
                for ext in (".fbx", ".glb", ".gltf", ".bvh"):
                    c = os.path.join(mdir, f"{base}_{aname}{ext}")
                    if os.path.exists(c):
                        src_motion = c
                        break
            if src_motion:
                try:
                    self._run_retarget_cmd(sk_glb, src_motion, out_anim, timeout=180)
                    retargeted.append(aname)
                except Exception:
                    pass

        return {
            "skin_glb": _to_posix_url_path(sk_glb, OUTPUTS_DIR, "outputs"),
            "skinned_fbx": _to_posix_url_path(sk_fbx, OUTPUTS_DIR, "outputs") if os.path.exists(sk_fbx) else None,
            "rig": rig_info,
            "animations": _find_anim_glbs(stem_path),
            "retargeted": retargeted,
        }

    def _apply_motion(self, stem_path: str, motion_path: str, name: str) -> dict:
        sk_glb = _find_skinned_glb(stem_path)
        rig_res = None
        if not sk_glb or not os.path.exists(sk_glb):
            rig_res = self._auto_rig_mesh(stem_path)
            sk_glb = stem_path + "_skinned.glb"
        out_anim = f"{stem_path}_anim_{name}.glb"
        info = self._run_retarget_cmd(sk_glb, motion_path, out_anim, timeout=240)
        resp = {
            "ok": True,
            "name": name,
            "anim_glb": _to_posix_url_path(out_anim, OUTPUTS_DIR, "outputs"),
            "animations": _find_anim_glbs(stem_path),
            **info,
        }
        if rig_res:
            resp.update(rig_res)
        return resp

    def _handle_rig(self):
        try:
            n = int(self.headers.get("Content-Length") or 0)
            data = json.loads((self.rfile.read(n) if n > 0 else b"{}").decode("utf-8") or "{}")
        except Exception:
            self._send_json({"error": "잘못된 요청 형식"}, 400)
            return
        full = _resolve_ply(data.get("ply") or "")
        if full is None:
            self._send_json({"error": "PLY 파일을 찾을 수 없어요"}, 400)
            return
        stem_path = full[:-4]
        joints = data.get("joints")
        if joints is not None and not isinstance(joints, dict):
            self._send_json({"error": "joints 형식이 올바르지 않아요"}, 400)
            return
        try:
            res = self._auto_rig_mesh(stem_path, custom_joints=joints)
            self._send_json({"ok": True, **res})
        except Exception as e:
            self._send_json({"error": str(e)}, 500)

    def _handle_motion(self):
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > 80_000_000:
                self._send_json({"error": "모션 파일 크기가 올바르지 않아요 (최대 80MB)"}, 400)
                return
            raw_body = self.rfile.read(n)
            ct = self.headers.get("Content-Type") or ""
            header = f"Content-Type: {ct}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8")
            msg = BytesParser().parsebytes(header + raw_body)
            if not msg.is_multipart():
                raise ValueError("multipart/form-data 필요")
            fields = {}
            motion_bytes = b""
            filename = ""
            for part in msg.get_payload():
                pname = part.get_param("name", header="content-disposition")
                if not pname:
                    continue
                payload = part.get_payload(decode=True) or b""
                if pname == "file":
                    motion_bytes = payload
                    filename = part.get_filename() or "motion.fbx"
                else:
                    fields[pname] = payload.decode("utf-8", "replace")
        except Exception as e:
            self._send_json({"error": f"잘못된 요청 형식: {e}"}, 400)
            return

        full = _resolve_ply(fields.get("ply") or "")
        if full is None:
            self._send_json({"error": "대상 PLY를 찾을 수 없어요"}, 400)
            return
        ext = os.path.splitext(filename)[1].lower()
        if ext not in (".fbx", ".glb", ".gltf", ".bvh"):
            self._send_json({"error": "지원하지 않는 모션 형식입니다 (.fbx, .glb, .bvh 지원)"}, 400)
            return
        if not motion_bytes:
            self._send_json({"error": "빈 모션 파일입니다"}, 400)
            return

        stem_path = full[:-4]
        try:
            raw_mname = os.path.splitext(os.path.basename(filename))[0]
            mslug = "".join(ch for ch in raw_mname.strip() if ch.isalnum() or ch in "-_") or "motion"
            motions_dir = os.path.join(UPLOADS_DIR, "motions")
            os.makedirs(motions_dir, exist_ok=True)
            motion_in = os.path.join(motions_dir, f"{os.path.basename(stem_path)}_{mslug}{ext}")
            with open(motion_in, "wb") as fh:
                fh.write(motion_bytes)
            self._send_json(self._apply_motion(stem_path, motion_in, mslug))
        except Exception as e:
            self._send_json({"error": str(e)}, 500)

    def _handle_motion_preset(self):
        try:
            n = int(self.headers.get("Content-Length") or 0)
            data = json.loads((self.rfile.read(n) if n > 0 else b"{}").decode("utf-8") or "{}")
        except Exception:
            self._send_json({"error": "잘못된 요청 형식"}, 400)
            return
        full = _resolve_ply(data.get("ply") or "")
        if full is None:
            self._send_json({"error": "대상 PLY를 찾을 수 없어요"}, 400)
            return
        preset = "".join(ch for ch in (data.get("preset") or "walk") if ch.isalnum() or ch in "-_")
        preset_fbx = os.path.join(str(config.PRESETS_DIR), f"{preset}.fbx")
        if not os.path.exists(preset_fbx):
            self._send_json({"error": f"프리셋 모션 '{preset}'을 찾을 수 없어요"}, 404)
            return
        try:
            self._send_json(self._apply_motion(full[:-4], preset_fbx, preset))
        except Exception as e:
            self._send_json({"error": str(e)}, 500)

    def do_POST(self):
        path = self.path.split("?", 1)[0].lstrip("/")
        if path == "api/fbx":
            self._handle_fbx()
            return
        if path == "api/rig":
            self._handle_rig()
            return
        if path == "api/motion":
            self._handle_motion()
            return
        if path == "api/motion/preset":
            self._handle_motion_preset()
            return
        if path != "api/upload":
            self.send_error(404)
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > 40_000_000:
                self._send_json({"error": "파일 크기가 올바르지 않아요 (최대 40MB)"}, 413 if n > 40_000_000 else 400)
                return
            raw_body = self.rfile.read(n)
            ct = self.headers.get("Content-Type") or ""
            img_bytes, mode, gauss, raw_stem = _parse_upload_body(ct, raw_body)
        except Exception:
            self._send_json({"error": "잘못된 요청 형식"}, 400)
            return
        if mode not in ("real", "synthetic"):
            self._send_json({"error": "mode가 올바르지 않아요"}, 400)
            return
        if gauss % 32 != 0 or not (32768 <= gauss <= 1048576):
            self._send_json({"error": "gaussians 값이 올바르지 않아요 (32768~1048576, 32의 배수)"}, 400)
            return
        if not img_bytes:
            self._send_json({"error": "빈 이미지"}, 400)
            return
        os.makedirs(UPLOADS_DIR, exist_ok=True)
        _gc_jobs()
        stem = "".join(ch for ch in raw_stem.strip() if ch.isalnum() or ch in "-_.") or "img"
        ts = time.strftime("%Y%m%d-%H%M%S")
        global _seq
        with _lock:
            _seq += 1
            seq = _seq
        full_stem = f"{stem}-{ts}-{seq:02d}"
        img_path = os.path.join(UPLOADS_DIR, full_stem + ".png")
        with open(img_path, "wb") as f:
            f.write(img_bytes)
        with _lock:
            _jobs[full_stem] = {
                "stem": full_stem, "img": img_path, "mode": mode,
                "gaussians": gauss, "status": "queued", "pct": 0,
                "phase": "queued", "ply": None, "msg": None, "seconds": None,
            }
        _start_worker()
        with _cond:
            _cond.notify_all()
        self._send_json({"stem": full_stem})

    def do_DELETE(self):
        path = self.path.split("?", 1)[0].lstrip("/")
        if path != "api/ply":
            self.send_error(404)
            return
        qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        name = (qs.get("name") or [""])[0]
        full = _resolve_ply(name)
        if full is None:
            self._send_json({"error": "잘못된 파일 이름"}, 400)
            return
        if not os.path.exists(full):
            self._send_json({"error": "파일이 없거나 이미 삭제되었어요"}, 404)
            return
        stem = full[:-4]
        victims = [full]
        for suffix in (
            ".json", "_condition.png", "_preview.png", "_preview.gif",
            "_mesh.ply", "_mesh.json", "_mesh_uv.bin", "_texture.png",
            "_mesh_preview.gif", ".fbx", "_skinned.glb", "_skinned.fbx",
        ):
            p = stem + suffix
            if os.path.exists(p):
                victims.append(p)
        d = os.path.dirname(stem)
        base_anim_prefix = os.path.basename(stem) + "_anim_"
        try:
            for fn in os.listdir(d):
                if fn.startswith(base_anim_prefix) and fn.endswith(".glb"):
                    victims.append(os.path.join(d, fn))
        except OSError:
            pass
        up = os.path.join(UPLOADS_DIR, os.path.basename(stem) + ".png")
        if os.path.exists(up):
            victims.append(up)
        deleted, errors = [], []
        for p in victims:
            try:
                os.remove(p)
                deleted.append(os.path.basename(p))
            except OSError as e:
                errors.append(f"{os.path.basename(p)}: {e}")
        self._send_json(
            {"ok": not errors, "deleted": deleted, "errors": errors},
            200 if not errors else 500,
        )

    def log_message(self, fmt, *args):
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    ap = argparse.ArgumentParser(description="dcd-img2rig Web Studio Server")
    ap.add_argument("port_pos", nargs="?", type=int, default=None, help="optional port (positional)")
    ap.add_argument("--host", default=os.environ.get("IMG2RIG_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--no-backfill", action="store_true", help="skip background preview GIF backfill")
    args = ap.parse_args()

    port = args.port or args.port_pos or int(os.environ.get("IMG2RIG_PORT", "8765"))
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(UPLOADS_DIR, exist_ok=True)

    if not args.no_backfill:
        threading.Thread(target=_backfill_previews, daemon=True).start()

    blender = config.find_blender()
    with Server((args.host, port), Handler) as httpd:
        print(f"\n  dcd-img2rig Studio: http://{args.host}:{port}")
        print(f"  outputs dir:        {OUTPUTS_DIR}")
        print(f"  blender:            {blender or 'not found (pure-Python FBX fallback)'}")
        print(f"  existing PLYs:      {len(find_plys())}\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
