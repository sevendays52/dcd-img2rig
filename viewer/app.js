/* Idol PLY & FBX Viewer — gsplat.js (3DGS) + WebGL2 MeshViewer (FBX) */
import * as gs from "./libs/index.es.js";
import { MeshViewer } from "./mesh-viewer.js";

const $ = (id) => document.getElementById(id);
const canvas = $("gl");
const meshCanvas = $("meshGl");
const gl = canvas.getContext("webgl2", { antialias: false, powerPreference: "high-performance" });
if (!gl) {
  $("status").textContent = "이 브라우저는 WebGL2를 지원하지 않습니다.";
  throw new Error("WebGL2 not supported");
}

const FOV_DEG = 42;
const VIEW = {
  smoothTAU: 0.18,
  autoResumeMs: 2400,
  autoSpeed: 0.42,
};

let renderer = null;
let camera = null;
let controls = null;
let scene = null;
let meshViewer = null;
let viewMode = "splat"; // "splat" | "mesh"
let autoRotate = true;
let interacting = false;
let lastInteract = -Infinity;
let lastFrameTime = 0;

const autoOrbit = {
  angle: 0,
  rad: 2,
  dy: -0.4,
};

const state = {
  splat: null,
  name: "",
  isSample: false,
  bounds: null,
  meta: null,
  baseFaces: null,
  baseVerts: null,
  kept: null,
  fbx: null, // {fbx, mesh_ply, mesh_preview, faces, verts, base_faces, fbx_bytes, verified_by, opacity, quality, method, clean, smooth, target_faces}
  meshLoadedUrl: "",
  skinnedLoadedUrl: "",
};

/* ---------- UI helpers ---------- */
function setBusy(msg) {
  $("overlay").hidden = !msg;
  $("overlayText").textContent = msg || "";
}

function setStatus(msg, isError = false) {
  const el = $("status");
  el.textContent = msg;
  el.className = "status" + (isError ? " error" : msg.includes("완료") ? " ok" : "");
}

function escapeHtml(str) {
  return String(str)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function fmtBytes(n) {
  if (n == null) return "–";
  if (n >= 1048576) return (n / 1048576).toFixed(2) + " MB";
  if (n >= 1024) return (n / 1024).toFixed(1) + " KB";
  return n + " B";
}

function fmtCompactFaces(n) {
  if (n == null) return "FBX";
  if (n >= 1000) return (n / 1000).toFixed(1).replace(/\.0$/, "") + "K면";
  return n + "면";
}

function resolveUrl(p) {
  if (!p) return "";
  if (/^(https?:|blob:|data:|\/)/.test(p)) return p;
  return "/" + p.replace(/^\/+/, "");
}

/* ---------- 3D setup ---------- */
function makeRenderer() {
  renderer = new gs.WebGLRenderer(canvas, null);
  renderer.backgroundColor = new gs.Color32(6, 6, 8);
  return renderer;
}

function makeCamera() {
  const camData = new gs.CameraData();
  camData.near = 0.1;
  camData.far = 2000;
  camera = new gs.Camera(camData);
  camera.position = new gs.Vector3(0, 0, -3);
}

function getCenter() {
  return state.bounds ? state.bounds.center() : new gs.Vector3(0, 0, 0);
}

function lookAt(pos, target) {
  camera.position = pos;
  const diff = target.subtract(pos);
  if (diff.magnitude() > 1e-6) {
    const dir = diff.normalize();
    const pitch = Math.asin(-Math.max(-1, Math.min(1, dir.y)));
    const yaw = Math.atan2(dir.x, dir.z);
    camera.rotation = gs.Quaternion.FromEuler(new gs.Vector3(pitch, yaw, 0));
  }
  camera.update();
  if (controls) controls.setCameraTarget(target);
}

function syncControlsToCamera(target) {
  if (!controls) return;
  controls.setCameraTarget(target);
  const prevDamp = controls.dampening;
  controls.dampening = 1;
  controls.update();
  controls.dampening = prevDamp;
}

function syncAutoOrbitFromCamera(target) {
  const dx = camera.position.x - target.x;
  const dy = camera.position.y - target.y;
  const dz = camera.position.z - target.z;
  autoOrbit.rad = Math.sqrt(dx * dx + dz * dz) || 1;
  autoOrbit.dy = dy;
  autoOrbit.angle = Math.atan2(dz, dx);
}

function markInteract() {
  interacting = true;
  lastInteract = performance.now();
}
function markRelease() {
  interacting = false;
  lastInteract = performance.now();
}
function markScroll() {
  lastInteract = performance.now();
}
function resetInteract() {
  interacting = false;
  lastInteract = -Infinity;
}
function isAutoActive(now) {
  return (
    viewMode === "splat" &&
    autoRotate &&
    !interacting &&
    !!state.splat &&
    now - lastInteract > VIEW.autoResumeMs
  );
}

function frameView() {
  if (!state.bounds) return;
  const b = state.bounds;
  const size = b.size();
  const center = b.center();
  const r = 0.5 * Math.max(size.x, size.y, size.z) || 0.5;
  const d = Math.max(0.6, (r / Math.tan((FOV_DEG * Math.PI) / 360)) * 1.45);
  const pos = new gs.Vector3(
    center.x + d * 0.45,
    center.y - d * 0.22,
    center.z - d * 0.85
  );
  lookAt(pos, center);
  syncControlsToCamera(center);
  syncAutoOrbitFromCamera(center);
}

/* ---------- render loop ---------- */
function scheduleNext() {
  if (typeof document !== "undefined" && document.visibilityState === "visible") {
    requestAnimationFrame(loop);
  } else {
    setTimeout(() => loop(performance.now()), 100);
  }
}

function loop(now = 0) {
  scheduleNext();
  if (!renderer) return;

  if (viewMode === "mesh") {
    if (meshViewer) meshViewer.render(now);
    return;
  }

  const dt = lastFrameTime ? Math.min(0.1, (now - lastFrameTime) / 1000) : 0;
  lastFrameTime = now;

  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (w === 0 || h === 0) return;

  if (canvas.width !== w || canvas.height !== h) {
    renderer.setSize(w, h);
    camera.data.fy = h / (2 * Math.tan((FOV_DEG * Math.PI) / 360));
  }

  try {
    if (isAutoActive(now)) {
      autoOrbit.angle += dt * VIEW.autoSpeed;
      const c = getCenter();
      const pos = new gs.Vector3(
        c.x + autoOrbit.rad * Math.cos(autoOrbit.angle),
        c.y + autoOrbit.dy,
        c.z + autoOrbit.rad * Math.sin(autoOrbit.angle)
      );
      lookAt(pos, c);
      if (controls) {
        const prevDamp = controls.dampening;
        controls.dampening = 1;
        controls.update();
        controls.dampening = prevDamp;
      }
    } else {
      if (controls) {
        controls.dampening = dt > 0 ? 1 - Math.exp(-dt / VIEW.smoothTAU) : 1;
        controls.update();
      }
      camera.update();
    }

    if (scene && scene.objects.length > 0) {
      renderer.render(scene, camera);
    }
    window.__loopOk = true;
  } catch (e) {
    window.__loopErr = e && e.message;
    console.error("loop error", e);
  }
}

/* ---------- View Mode Switching (3DGS <-> FBX Mesh <-> Rig/Motion) ---------- */
function updateHeaderAndDock() {
  const hasModel = !!state.splat;
  const fx = state.fbx;
  const hasMesh = !!(fx && fx.mesh_ply);
  const hasRig = !!(fx && fx.skin_glb);
  const animCount = (fx && Array.isArray(fx.animations)) ? fx.animations.length : 0;
  const boneOpen = !$("bonePanel").hidden;

  $("stageDock").hidden = !hasModel;
  $("tabSplat").classList.toggle("on", viewMode === "splat");
  $("tabMesh").classList.toggle("on", viewMode === "mesh" && !boneOpen);
  $("tabMesh").classList.toggle("empty", !hasMesh);
  $("tabRig").classList.toggle("on", viewMode === "mesh" && boneOpen);
  $("tabRig").classList.toggle("empty", !hasRig);

  const pill = $("tabMeshPill");
  if (hasMesh) {
    pill.textContent = fx.faces ? fmtCompactFaces(fx.faces) : "보기";
  } else {
    pill.textContent = "자동 변환";
  }

  const rigPill = $("tabRigPill");
  if (animCount > 0) {
    rigPill.textContent = `🎬 ${animCount}`;
  } else if (hasRig) {
    rigPill.textContent = "19본 ✓";
  } else {
    rigPill.textContent = "자동 리깅";
  }

  $("meshToolbar").hidden = !(viewMode === "mesh" && hasMesh);

  if (viewMode === "mesh" && hasMesh) {
    const fStr = fx.faces != null ? fx.faces.toLocaleString() : "–";
    const vStr = fx.verts != null ? fx.verts.toLocaleString() : "–";
    const bStr = fx.fbx_bytes != null ? fmtBytes(fx.fbx_bytes) : "";
    const texStr = (meshViewer && (meshViewer.skHasTex || meshViewer.hasTex)) ? " · 1024px" : "";
    $("meshDockStat").innerHTML = `<b>${fStr}</b> 면 · <b>${vStr}</b> 점${texStr}${bStr ? " · " + bStr : ""}`;
    const dlUrl = fx.skinned_fbx || fx.fbx;
    $("btnDockFbxDl").href = dlUrl ? resolveUrl(dlUrl) : "#";
    $("btnDockFbxDl").textContent = fx.skinned_fbx ? "⬇ 리깅 FBX" : "⬇ FBX";
    $("btnDockFbxDl").style.display = dlUrl ? "" : "none";
    $("statsInfo").innerHTML =
      `<b>${fStr}</b> 삼각면 · <b>${vStr}</b> 정점${fx.verified_by === "blender" ? " · Blender 검증" : ""}`;
  } else if (state.splat) {
    const b = state.splat.bounds;
    const s = b.size();
    const count = state.splat.data.vertexCount;
    $("statsInfo").innerHTML =
      `<b>${count.toLocaleString()}</b> 가우시안 · 범위 ${s.x.toFixed(2)} × ${s.y.toFixed(2)} × ${s.z.toFixed(2)}`;
  } else {
    $("statsInfo").innerHTML = "";
  }

  const fbxBtn = $("btnFbx");
  const canFbx = !!(state.splat && state.isSample);
  fbxBtn.classList.toggle("hidden", !canFbx);
  fbxBtn.textContent = hasMesh ? "⚙ 메시·FBX 설정" : "⚡ 자동 메시·리깅";

  updateMeshStyleUI();
  updateSidebarFbxCard();
}

function updateSidebarFbxCard() {
  const wrap = $("fbxInfoWrap");
  const fx = state.fbx;
  if (!fx || !fx.fbx) {
    wrap.hidden = true;
    return;
  }
  wrap.hidden = false;
  const fStr = fx.faces != null ? `${fx.faces.toLocaleString()} 삼각면` : "메시 변환 완료";
  const vStr = fx.verts != null ? `${fx.verts.toLocaleString()} 정점` : "";
  const bStr = fx.fbx_bytes != null ? fmtBytes(fx.fbx_bytes) : "";
  const baseStr = fx.base_faces != null ? `원본 ${fx.base_faces.toLocaleString()}면` : "";
  $("fbxSideFaces").textContent = fStr;
  $("fbxSideSub").textContent = [vStr, bStr, baseStr].filter(Boolean).join(" · ") || "Binary FBX 7.4";
  $("fbxSideBadge").hidden = fx.verified_by !== "blender";
  $("fbxSideBadge").textContent = fx.has_texture
    ? `✓ Blender 검증 · ${fx.tex_size || 1024}px 텍스처`
    : "✓ Blender 검증";

  const thumb = $("fbxSideThumb");
  if (fx.mesh_preview) {
    thumb.src = resolveUrl(fx.mesh_preview) + `?t=${Date.now()}`;
    thumb.hidden = false;
  } else {
    thumb.hidden = true;
  }
  $("btnSideDlFbx").href = resolveUrl(fx.fbx);
  $("btnSideViewMesh").textContent = viewMode === "mesh" ? "3DGS 스플랫 보기" : "3D 메시로 보기";
}

function loadImageAsync(src) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => resolve(null);
    img.src = src;
  });
}

async function quickAutoConvertAndRig(plyName, { openRigPanel = false } = {}) {
  if (!plyName || fbxBusy) return false;
  fbxBusy = true;
  setBusy("자동 3D 메시 · 1024px 텍스처 · 19본 리깅 변환 중…");
  try {
    const res = await fetch("/api/fbx", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ply: plyName, action: "convert", ...FBX_DEFAULTS, rig: true }),
    });
    const data = await res.json();
    if (!res.ok || data.error) throw new Error(data.error || `HTTP ${res.status}`);

    for (let i = 0; i < 1800; i++) {
      await new Promise((r) => setTimeout(r, 900));
      const { job } = await fetch(`/api/job?stem=${encodeURIComponent(data.stem)}`).then((r) => r.json());
      if (!job) continue;
      const label = job.phase === "preview" ? "회전 미리보기 렌더링 중…"
        : "3D 메시 · 1024px 텍스처 · 자동 리깅 처리 중…";
      setBusy(`${label} (${Math.round(job.seconds || 0)}s)`);
      if (job.status === "done") {
        state.fbx = {
          fbx: job.fbx,
          skin_glb: job.skin_glb,
          skinned_fbx: job.skinned_fbx,
          rig: job.rig,
          animations: (state.fbx && state.fbx.animations) || [],
          mesh_ply: job.mesh_ply,
          mesh_uv: job.mesh_uv,
          texture: job.texture,
          mesh_preview: job.mesh_preview,
          has_texture: job.has_texture,
          tex_size: job.tex_size,
          faces: job.faces,
          verts: job.verts,
          base_faces: job.base_faces,
          fbx_bytes: job.fbx_bytes,
          verified_by: job.verified_by,
          quality: job.quality,
          method: job.method,
          clean: job.clean,
          smooth: job.smooth,
          fill_holes: job.fill_holes,
          opacity: job.opacity,
          target_faces: job.target_faces,
        };
        state.meshLoadedUrl = "";
        state.skinnedLoadedUrl = "";
        await refreshSamples(plyName);
        fbxBusy = false;
        setBusy("");
        await setViewMode(openRigPanel ? "rig" : "mesh");
        setStatus(`원스톱 변환 완료 · ${(state.fbx.faces || 0).toLocaleString()}면 · 1024px 텍스처 · 19본 리깅 ✓`);
        return true;
      }
      if (job.status === "error") throw new Error(job.msg || "자동 변환 실패");
    }
    throw new Error("변환 시간이 초과되었어요");
  } catch (e) {
    setStatus(`자동 변환 실패: ${e.message}`, true);
    return false;
  } finally {
    fbxBusy = false;
    setBusy("");
  }
}

async function setViewMode(mode) {
  const wantRig = mode === "rig";
  if (mode === "mesh" || wantRig) {
    if (!state.fbx || !state.fbx.mesh_ply) {
      if (state.name && state.isSample) {
        await quickAutoConvertAndRig(state.name, { openRigPanel: wantRig });
      }
      return;
    }
    const url = resolveUrl(state.fbx.mesh_ply);
    let loadedUvBuf = null;
    let loadedTexImg = null;
    if (state.meshLoadedUrl !== url) {
      setBusy("3D 메시 · 1024px 텍스처 로딩 중…");
      try {
        const ts = Date.now();
        const uvUrl = state.fbx.mesh_uv ? resolveUrl(state.fbx.mesh_uv) + `?t=${ts}` : "";
        const texUrl = state.fbx.texture ? resolveUrl(state.fbx.texture) + `?t=${ts}` : "";
        const [res, uvBuf, texImg] = await Promise.all([
          fetch(url + `?t=${ts}`),
          uvUrl ? fetch(uvUrl).then((r) => (r.ok ? r.arrayBuffer() : null)).catch(() => null) : Promise.resolve(null),
          texUrl ? loadImageAsync(texUrl) : Promise.resolve(null),
        ]);
        if (!res.ok) throw new Error(`메시 로딩 실패 (${res.status})`);
        const buf = await res.arrayBuffer();
        loadedUvBuf = uvBuf;
        loadedTexImg = texImg;
        const info = meshViewer.loadFromBuffer(buf, uvBuf, texImg);
        state.meshLoadedUrl = url;
        if (state.fbx.faces == null) state.fbx.faces = info.faceCount;
        if (state.fbx.verts == null) state.fbx.verts = info.vertexCount;
      } finally {
        setBusy("");
      }
    }
    // ---- auto-rig if entering Rig/Motion tab and not rigged yet ----
    if (wantRig && !state.fbx.skin_glb) {
      await autoRigCurrentMesh();
    }
    // ---- skinned GLB (rig) load (preserving crisp 1024px UV texture!) ----
    const skinUrl = state.fbx.skin_glb ? resolveUrl(state.fbx.skin_glb) : "";
    if (skinUrl && state.skinnedLoadedUrl !== skinUrl) {
      setBusy("리깅 스킨 · 텍스처 적용 중…");
      try {
        const sres = await fetch(skinUrl + `?t=${Date.now()}`);
        if (sres.ok) {
          const sbuf = await sres.arrayBuffer();
          meshViewer.loadSkinnedGlb(sbuf, loadedUvBuf, loadedTexImg);
          state.skinnedLoadedUrl = skinUrl;
        }
      } catch (e) {
        console.warn("스킨 GLB 로딩 실패:", e);
      } finally {
        setBusy("");
      }
    }
    viewMode = "mesh";
    canvas.hidden = true;
    meshCanvas.hidden = false;
    meshViewer.active = true;
    meshViewer.setAutoRotate(autoRotate);
    if (wantRig) {
      meshViewer.setBoneShow(true);
      buildBonePanel(true);
    } else {
      meshViewer.setBoneEditMode(false);
      meshViewer.setMeshXray(false);
      meshViewer.setBoneShow(false);
      $("bonePanel").hidden = true;
      buildBonePanel(false);
    }
    const texNote = (meshViewer.skHasTex || meshViewer.hasTex) ? " · 1024px 텍스처" : "";
    const rigNote = meshViewer.skinned ? " · 19본 리깅" : "";
    setStatus(`${state.name} · 3D 메시 (${(state.fbx.faces || 0).toLocaleString()}면${texNote}${rigNote})`);
  } else {
    viewMode = "splat";
    meshViewer.active = false;
    meshCanvas.hidden = true;
    canvas.hidden = false;
    $("bonePanel").hidden = true;
    resetInteract();
    if (state.name) setStatus(`${state.name} · 3D 가우시안 스플랫`);
  }
  updateHeaderAndDock();
}

function updateMeshStyleUI() {
  if (!meshViewer) return;
  $("btnShadeColor").classList.toggle("on", meshViewer.shading === "color");
  $("btnShadeUnlit").classList.toggle("on", meshViewer.shading === "unlit");
  $("btnShadeClay").classList.toggle("on", meshViewer.shading === "clay");
  const wf = $("btnWireframe");
  wf.classList.toggle("on", meshViewer.wireframe);
  wf.textContent = meshViewer.wireframe ? "와이어 ON" : "와이어";
  const bo = $("btnBoneOverlay");
  if (bo) {
    bo.classList.toggle("on", !!meshViewer.boneShow);
    bo.style.display = meshViewer.skinned ? "" : "none";
  }
  const xr = $("btnMeshXray");
  if (xr) {
    xr.classList.toggle("on", !!meshViewer.meshXray);
    xr.style.display = meshViewer.skinned ? "" : "none";
  }
}

/* ---------- bone & motion studio panel ---------- */
let _boneEventsBound = false;
let _activeMotionPath = "";

function syncEditSlidersForBone(name) {
  if (!meshViewer || !name) return;
  const pos = meshViewer.getEditJointPos(name);
  if (!pos) return;
  $("editPosX").value = String(pos[0]);
  $("editPosY").value = String(pos[1]);
  $("editPosZ").value = String(pos[2]);
  $("editPosXVal").textContent = pos[0].toFixed(2);
  $("editPosYVal").textContent = pos[1].toFixed(2);
  $("editPosZVal").textContent = pos[2].toFixed(2);
}

function switchBoneTab(tab) {
  const isMotion = tab === "motion";
  const isEdit = tab === "edit";
  const isPose = tab === "pose";
  $("bpTabMotion").classList.toggle("on", isMotion);
  $("bpTabEdit").classList.toggle("on", isEdit);
  $("bpTabPose").classList.toggle("on", isPose);
  $("bpSecMotion").hidden = !isMotion;
  $("bpSecEdit").hidden = !isEdit;
  $("bpSecPose").hidden = !isPose;
  $("bonePoseStat").textContent = "";

  if (meshViewer && meshViewer.skinned) {
    meshViewer.setBoneEditMode(isEdit);
    if (isEdit) {
      meshViewer.setBoneShow(true);
      meshViewer.setMeshXray($("chkEditXray").checked);
      meshViewer.setAutoRotate(false);
      const curName = $("editBoneSel").value || meshViewer.boneNames[0];
      if (curName) {
        meshViewer.selectBone(curName);
        syncEditSlidersForBone(curName);
      }
    } else {
      meshViewer.setMeshXray(false);
      meshViewer.setAutoRotate(autoRotate);
    }
    updateMeshStyleUI();
  }
}

async function applyEditedJoints() {
  if (!state.name || !meshViewer || !meshViewer.skinned) return;
  const joints = meshViewer.getAllEditedJoints();
  setBusy("수정한 본 위치로 스킨 웨이트 · 모션 재적용 중…");
  try {
    const res = await fetch("/api/rig", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ply: state.name, joints }),
    });
    const data = await res.json();
    if (!res.ok || data.error) throw new Error(data.error || `HTTP ${res.status}`);
    if (!state.fbx) state.fbx = {};
    state.fbx.skin_glb = data.skin_glb;
    state.fbx.skinned_fbx = data.skinned_fbx || null;
    state.fbx.rig = data.rig || true;
    if (data.animations) state.fbx.animations = data.animations;
    state.skinnedLoadedUrl = "";

    const skinUrl = resolveUrl(data.skin_glb);
    const sres = await fetch(skinUrl + `?t=${Date.now()}`);
    if (!sres.ok) throw new Error(`스킨 GLB 로딩 실패 (${sres.status})`);
    meshViewer.loadSkinnedGlb(await sres.arrayBuffer());
    state.skinnedLoadedUrl = skinUrl;
    meshViewer.setBoneEditMode(true);
    meshViewer.setMeshXray($("chkEditXray").checked);
    await refreshSamples(state.name);
    buildBonePanel(true);
    updateHeaderAndDock();
    const reNote = (data.retargeted && data.retargeted.length)
      ? ` · 모션 ${data.retargeted.join(", ")} 자동 재적용`
      : "";
    $("bonePoseStat").textContent = `본 위치 저장 · 재리깅 완료 ✓${reNote}`;
    setStatus(`본 위치 재리깅 완료${reNote}`);
  } catch (e) {
    setStatus(`본 위치 재리깅 실패: ${e.message}`, true);
    $("bonePoseStat").textContent = `오류: ${e.message}`;
  } finally {
    setBusy("");
  }
}

async function autoRigCurrentMesh() {
  if (!state.name) return;
  setBusy("휴머노이드 19본 자동 리깅 중…");
  try {
    const res = await fetch("/api/rig", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ply: state.name }),
    });
    const data = await res.json();
    if (!res.ok || data.error) throw new Error(data.error || `HTTP ${res.status}`);
    if (!state.fbx) state.fbx = {};
    state.fbx.skin_glb = data.skin_glb;
    state.fbx.skinned_fbx = data.skinned_fbx || null;
    state.fbx.rig = data.rig || true;
    state.skinnedLoadedUrl = "";
    const skinUrl = resolveUrl(data.skin_glb);
    const sres = await fetch(skinUrl + `?t=${Date.now()}`);
    if (!sres.ok) throw new Error(`스킨 GLB 로딩 실패 (${sres.status})`);
    meshViewer.loadSkinnedGlb(await sres.arrayBuffer());
    state.skinnedLoadedUrl = skinUrl;
    await refreshSamples(state.name);
    buildBonePanel(true);
    updateHeaderAndDock();
    setStatus(`${state.name} · 19본 자동 리깅 완료`);
  } catch (e) {
    setStatus(`자동 리깅 실패: ${e.message}`, true);
  } finally {
    setBusy("");
  }
}

async function loadAnimGlbUrl(url, label = "") {
  setBusy(label ? `모션 '${label}' 로딩 중…` : "애니메이션 로딩 중…");
  try {
    const res = await fetch(resolveUrl(url) + `?t=${Date.now()}`);
    if (!res.ok) throw new Error(`애니메이션 로딩 실패 (${res.status})`);
    meshViewer.loadSkinnedGlb(await res.arrayBuffer());
    state.skinnedLoadedUrl = resolveUrl(url);
    _activeMotionPath = url;
    buildBonePanel(true);
  } finally {
    setBusy("");
  }
}

async function applyPresetMotion(preset = "walk") {
  if (!state.name) return;
  if (!state.fbx || !state.fbx.mesh_ply) {
    const ok = await quickAutoConvertAndRig(state.name, { openRigPanel: true });
    if (!ok) return;
  }
  const existing = (state.fbx.animations || []).find((a) => a.name === preset);
  if (existing) {
    await loadAnimGlbUrl(existing.path, existing.name);
    setStatus(`모션 재생 중 · ${existing.name}`);
    return;
  }
  setBusy(`'${preset}' 모션 자동 리타겟팅 중…`);
  try {
    const res = await fetch("/api/motion/preset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ply: state.name, preset }),
    });
    const data = await res.json();
    if (!res.ok || data.error) throw new Error(data.error || `HTTP ${res.status}`);
    if (data.skin_glb) state.fbx.skin_glb = data.skin_glb;
    if (data.skinned_fbx) state.fbx.skinned_fbx = data.skinned_fbx;
    state.fbx.animations = data.animations || [];
    await loadAnimGlbUrl(data.anim_glb, data.name);
    await refreshSamples(state.name);
    updateHeaderAndDock();
    setStatus(`'${data.name}' 모션 적용 완료 (${data.frames || "?"}프레임)`);
  } catch (e) {
    setStatus(`프리셋 모션 적용 실패: ${e.message}`, true);
  } finally {
    setBusy("");
  }
}

async function uploadMotionFile(file) {
  if (!file || !state.name) return;
  if (!state.fbx || !state.fbx.mesh_ply) {
    const ok = await quickAutoConvertAndRig(state.name, { openRigPanel: true });
    if (!ok) return;
  }
  setBusy(`Mixamo 모션 자동 리타겟팅 중 (${file.name})…`);
  try {
    const fd = new FormData();
    fd.append("ply", state.name);
    fd.append("file", file);
    const res = await fetch("/api/motion", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok || data.error) throw new Error(data.error || `HTTP ${res.status}`);
    if (!state.fbx) state.fbx = {};
    if (data.skin_glb) state.fbx.skin_glb = data.skin_glb;
    if (data.skinned_fbx) state.fbx.skinned_fbx = data.skinned_fbx;
    state.fbx.animations = data.animations || [];
    if (viewMode !== "mesh") await setViewMode("rig");
    await loadAnimGlbUrl(data.anim_glb, data.name);
    await refreshSamples(state.name);
    updateHeaderAndDock();
    setStatus(`Mixamo 모션 적용 완료 · ${data.name} (${data.frames || "?"}프레임)`);
  } catch (e) {
    setStatus(`모션 적용 실패: ${e.message}`, true);
  } finally {
    setBusy("");
  }
}

function refreshAnimUI() {
  const listBox = $("motionList");
  const ctrlRow = $("animCtrlRow");
  const list = (state.fbx && state.fbx.animations) ? state.fbx.animations : [];
  const activeAnim = meshViewer && meshViewer.activeAnim >= 0 ? meshViewer.animations[meshViewer.activeAnim] : null;
  if (!activeAnim) _activeMotionPath = "";

  const chips = [
    `<button type="button" class="mchip${!_activeMotionPath ? " on" : ""}" data-path="">정지 (Rest)</button>`,
    ...list.map((a) =>
      `<button type="button" class="mchip${_activeMotionPath === a.path ? " on" : ""}" data-path="${escapeHtml(a.path)}" title="${escapeHtml(a.name)}">🎬 ${escapeHtml(a.name)}</button>`
    ),
  ];
  listBox.innerHTML = chips.join("");
  listBox.querySelectorAll(".mchip").forEach((chip) => {
    chip.addEventListener("click", async () => {
      const p = chip.dataset.path;
      if (!p) {
        _activeMotionPath = "";
        meshViewer.setAnimation(-1);
        refreshAnimUI();
      } else {
        await loadAnimGlbUrl(p, chip.textContent.replace(/^🎬\s*/, ""));
      }
    });
  });

  ctrlRow.hidden = !activeAnim;
  if (activeAnim) {
    $("animScrub").max = String(activeAnim.duration || 1);
    $("animScrub").value = String(meshViewer.animTime || 0);
    $("btnAnimPlay").textContent = meshViewer.animPlaying ? "⏸" : "▶";
    $("animTimeStat").textContent = `${(meshViewer.animTime || 0).toFixed(1)}/${(activeAnim.duration || 0).toFixed(1)}s`;
  }
}

function buildBonePanel(openPanel = false) {
  if (!meshViewer || !meshViewer.skinned) {
    $("bonePanel").hidden = true;
    return;
  }
  const panel = $("bonePanel");
  if (openPanel) panel.hidden = false;

  const bones = meshViewer.listBones();
  for (const selId of ["boneSel", "editBoneSel"]) {
    const sel = $(selId);
    if (!sel) continue;
    const prev = sel.value;
    sel.innerHTML = "";
    for (const b of bones) {
      const opt = document.createElement("option");
      opt.value = b.name;
      opt.textContent = b.parentName ? `${b.parentName} / ${b.name}` : b.name;
      sel.appendChild(opt);
    }
    if (prev && bones.some((b) => b.name === prev)) sel.value = prev;
  }
  const sel = $("boneSel");
  const editSel = $("editBoneSel");
  if (editSel && editSel.value) syncEditSlidersForBone(editSel.value);
  refreshAnimUI();

  if (_boneEventsBound) return;
  _boneEventsBound = true;

  $("btnCloseBonePanel").addEventListener("click", () => {
    $("bonePanel").hidden = true;
    if (meshViewer && meshViewer.boneEditMode) {
      meshViewer.setBoneEditMode(false);
      meshViewer.setMeshXray(false);
      meshViewer.setAutoRotate(autoRotate);
    }
    updateHeaderAndDock();
  });
  $("bpTabMotion").addEventListener("click", () => switchBoneTab("motion"));
  $("bpTabEdit").addEventListener("click", () => switchBoneTab("edit"));
  $("bpTabPose").addEventListener("click", () => switchBoneTab("pose"));

  // --- Direct 3D joint selection & drag callbacks ---
  meshViewer.onBoneSelect = (boneName) => {
    if ($("editBoneSel")) $("editBoneSel").value = boneName;
    if ($("boneSel")) $("boneSel").value = boneName;
    syncEditSlidersForBone(boneName);
    $("bonePoseStat").textContent = `선택됨: ${boneName}`;
  };
  meshViewer.onBoneEditChange = (boneName, pos) => {
    if ($("editBoneSel")) $("editBoneSel").value = boneName;
    syncEditSlidersForBone(boneName);
    $("bonePoseStat").textContent = `${boneName} 이동 (${pos[0].toFixed(2)}, ${pos[1].toFixed(2)}, ${pos[2].toFixed(2)})`;
  };

  // --- Bone Position Editor controls ---
  $("editBoneSel").addEventListener("change", (e) => {
    meshViewer.selectBone(e.target.value);
    syncEditSlidersForBone(e.target.value);
  });
  $("chkBoneMirror").addEventListener("change", (e) => meshViewer.setBoneMirror(e.target.checked));
  $("chkBoneChildren").addEventListener("change", (e) => meshViewer.setBoneMoveChildren(e.target.checked));
  $("chkEditXray").addEventListener("change", (e) => {
    meshViewer.setMeshXray(e.target.checked);
    updateMeshStyleUI();
  });
  const onEditSlider = () => {
    const name = $("editBoneSel").value;
    if (!name) return;
    const pos = [+$("editPosX").value, +$("editPosY").value, +$("editPosZ").value];
    meshViewer.setEditJointPos(name, pos);
  };
  for (const id of ["editPosX", "editPosY", "editPosZ"]) {
    $(id).addEventListener("input", onEditSlider);
  }
  $("btnEditReset").addEventListener("click", () => {
    meshViewer.resetEditJoints();
    syncEditSlidersForBone($("editBoneSel").value);
    $("bonePoseStat").textContent = "관절 위치를 원래대로 되돌렸어요";
  });
  $("btnEditApply").addEventListener("click", applyEditedJoints);

  const sliders = ["boneTx","boneTy","boneTz","boneRx","boneRy","boneRz"];
  function _syncSliderReadouts() {
    for (const id of ["boneRx","boneRy","boneRz"]) {
      const el = $(id + "Val");
      if (el) el.textContent = `${Number($(id).value).toFixed(0)}°`;
    }
    for (const id of ["boneTx","boneTy","boneTz"]) {
      const el = $(id + "Val");
      if (el) el.textContent = Number($(id).value).toFixed(2);
    }
  }
  let _raf = 0;
  function _applyBonePose() {
    _syncSliderReadouts();
    if (_raf) return;
    _raf = requestAnimationFrame(() => {
      _raf = 0;
      const name = sel.value;
      if (!name) return;
      const t = [+$("boneTx").value, +$("boneTy").value, +$("boneTz").value];
      const eDeg = [+$("boneRx").value, +$("boneRy").value, +$("boneRz").value];
      meshViewer.setBoneTRS(name, { t, eDeg });
      const nonzero = t.some(v => v !== 0) || eDeg.some(v => v !== 0);
      $("bonePoseStat").textContent = nonzero ? `${name} 포즈 적용됨` : "";
    });
  }
  for (const id of sliders) $(id).addEventListener("input", _applyBonePose);

  $("btnBoneReset").addEventListener("click", () => {
    const name = sel.value;
    if (!name) return;
    meshViewer.resetBonePose(name);
    for (const id of sliders) $(id).value = "0";
    _syncSliderReadouts();
    $("bonePoseStat").textContent = `${name} → rest`;
  });
  $("btnBoneResetAll").addEventListener("click", () => {
    meshViewer.resetBonePose(null);
    for (const id of sliders) $(id).value = "0";
    _syncSliderReadouts();
    $("bonePoseStat").textContent = "모든 bone → rest";
  });
  sel.addEventListener("change", () => {
    for (const id of sliders) $(id).value = "0";
    _syncSliderReadouts();
  });

  $("btnPresetWalk").addEventListener("click", () => applyPresetMotion("walk"));
  const mDrop = $("motionDrop");
  mDrop.addEventListener("click", () => $("motionFileInput").click());
  mDrop.addEventListener("dragover", (e) => { e.preventDefault(); e.stopPropagation(); mDrop.classList.add("over"); });
  mDrop.addEventListener("dragleave", () => mDrop.classList.remove("over"));
  mDrop.addEventListener("drop", (e) => {
    e.preventDefault(); e.stopPropagation();
    mDrop.classList.remove("over");
    const f = e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) uploadMotionFile(f);
  });
  $("motionFileInput").addEventListener("change", (e) => {
    const f = e.target.files && e.target.files[0];
    e.target.value = "";
    if (f) uploadMotionFile(f);
  });
  $("btnAnimPlay").addEventListener("click", () => {
    if (meshViewer.animPlaying) meshViewer.pauseAnimation();
    else meshViewer.playAnimation();
    refreshAnimUI();
  });
  $("animScrub").addEventListener("input", (e) => {
    meshViewer.pauseAnimation();
    meshViewer.seekAnimation(+e.target.value);
    refreshAnimUI();
  });
  $("animSpeed").addEventListener("change", (e) => {
    meshViewer.setAnimSpeed(+e.target.value);
  });
  meshViewer.onAnimTick = (t, dur, playing) => {
    const sc = $("animScrub");
    if (sc && document.activeElement !== sc) sc.value = String(t);
    $("btnAnimPlay").textContent = playing ? "⏸" : "▶";
    $("animTimeStat").textContent = `${t.toFixed(1)}/${dur.toFixed(1)}s`;
  };
}

/* ---------- load PLY ---------- */
function applyMeta(m, plyUrl = "", condHint = "") {
  state.meta = m || null;

  const condImg = $("conditionImg");
  const condWrap = $("conditionWrap");
  let condSrc = "";
  if (condHint) {
    condSrc = resolveUrl(condHint);
  } else if (m && (m.condition_image || (m.images && m.images.condition))) {
    const raw = m.condition_image || m.images.condition;
    condSrc = raw.startsWith("http") ? raw : `/${raw.split("/").slice(-2).join("/")}`;
  } else if (plyUrl && /\.ply$/i.test(plyUrl)) {
    condSrc = resolveUrl(plyUrl.replace(/\.ply$/i, "_condition.png"));
  }

  if (condSrc) {
    condImg.onload = () => { condWrap.hidden = false; };
    condImg.onerror = () => { condWrap.hidden = true; };
    condImg.src = condSrc;
  } else {
    condImg.removeAttribute("src");
    condWrap.hidden = true;
  }

  const info = $("metaInfo");
  if (!m) {
    info.innerHTML = '<div style="font-size:12px;color:var(--dim-2)">모델을 열면 정보가 표시됩니다</div>';
    return;
  }

  const rows = [];
  if (m.num_gaussians != null) rows.push(["가우시안 수", m.num_gaussians.toLocaleString(), true]);
  if (m.mode) rows.push(["모드", m.mode]);
  if (m.seed != null) rows.push(["시드", m.seed]);
  if (m.steps != null) rows.push(["스텝", m.steps]);
  if (m.cfg_strength != null) rows.push(["CFG", m.cfg_strength]);
  if (m.device) rows.push(["디바이스", m.device]);
  if (m.image_size) rows.push(["입력 크기", m.image_size.join(" × ")]);
  if (m.load_seconds != null) rows.push(["모델 로드", `${Number(m.load_seconds).toFixed(1)}s`]);
  if (m.sample_seconds != null) rows.push(["샘플링", `${Number(m.sample_seconds).toFixed(1)}s`]);
  try { if (m.timestamp) rows.push(["생성 시각", new Date(m.timestamp).toLocaleString("ko-KR")]); } catch {}
  info.innerHTML = rows
    .map(([k, v, hi]) => `<div class="row"><span>${escapeHtml(k)}</span><b${hi ? ' class="hi"' : ""}>${escapeHtml(v)}</b></div>`)
    .join("");
}

function resetMeshCache() {
  state.baseFaces = null;
  state.baseVerts = null;
  state.kept = null;
  state.meshLoadedUrl = "";
  if (meshViewer) meshViewer.clear();
}

async function loadPlyFromUrl(url, name, condHint = "", fbxInfo = null) {
  const fetchUrl = resolveUrl(url);
  setBusy(`${name} 로딩 중…`);
  try {
    const [res, metaRes] = await Promise.all([
      fetch(fetchUrl),
      fetch(fetchUrl.replace(/\.ply$/i, "") + ".json").catch(() => null),
    ]);
    if (!res.ok) throw new Error(`fetch 실패: ${res.status}`);
    const buf = await res.arrayBuffer();

    scene = new gs.Scene();
    const splat = gs.PLYLoader.LoadFromArrayBuffer(buf, scene);
    resetMeshCache();
    state.splat = splat;
    state.name = name;
    state.isSample = true;
    state.bounds = splat.bounds;
    state.fbx = fbxInfo;

    $("fileName").textContent = name;
    $("fileName").title = name;

    const meta = metaRes && metaRes.ok ? await metaRes.json().catch(() => null) : null;
    applyMeta(meta, fetchUrl, condHint);

    await setViewMode("splat");
    frameView();
    resetInteract();
    setStatus(`${name} 로드 완료`);
  } finally {
    setBusy("");
  }
}

async function loadPlyFromFile(file) {
  setBusy(`${file.name} 로딩 중…`);
  try {
    const buf = await file.arrayBuffer();
    scene = new gs.Scene();
    const splat = gs.PLYLoader.LoadFromArrayBuffer(buf, scene);
    resetMeshCache();
    state.splat = splat;
    state.name = file.name;
    state.isSample = false;
    state.bounds = splat.bounds;
    state.fbx = null;
    $("fileName").textContent = file.name;
    $("fileName").title = file.name;
    applyMeta(null);
    await setViewMode("splat");
    frameView();
    resetInteract();
    setStatus(`${file.name} 로드 완료`);
  } finally {
    setBusy("");
  }
}

async function loadSample(name, { openMeshView = false } = {}) {
  const { plys } = await fetch("/api/plys").then((r) => r.json());
  const item = plys.find((p) => p.name === name);
  if (!item) throw new Error(`${name} 찾을 수 없음`);
  markActiveSample(name);
  const mi = item.mesh_info || {};
  const fbxInfo = (item.fbx || item.mesh_ply) ? {
    fbx: item.fbx,
    mesh_ply: item.mesh_ply,
    mesh_uv: item.mesh_uv,
    texture: item.texture,
    mesh_preview: item.mesh_preview,
    has_texture: mi.has_texture ?? null,
    tex_size: mi.tex_size ?? null,
    faces: mi.faces ?? null,
    verts: mi.verts ?? null,
    base_faces: mi.base_faces ?? null,
    fbx_bytes: mi.fbx_bytes ?? null,
    verified_by: mi.verified_by ?? null,
    opacity: mi.opacity ?? null,
    quality: mi.quality ?? null,
    method: mi.method ?? null,
    clean: mi.clean ?? null,
    smooth: mi.smooth ?? null,
    fill_holes: mi.fill_holes ?? null,
    target_faces: mi.target_faces ?? null,
    skin_glb: item.skin_glb || null,
    skinned_fbx: item.skinned_fbx || null,
    animations: item.animations || [],
  } : null;

  await loadPlyFromUrl(item.path, item.name, item.condition || "", fbxInfo);
  if (openMeshView && fbxInfo && fbxInfo.mesh_ply) {
    await setViewMode("mesh");
  }
}

function markActiveSample(name) {
  document.querySelectorAll("#samples .sample").forEach((el) => {
    el.classList.toggle("active", el.dataset.name === name);
  });
}

/* ---------- 메시 / FBX 변환 모달 ---------- */
let fbxOpenPly = "";
let fbxBusy = false;

function showFbxError(msg) {
  const el = $("fbxError");
  el.textContent = msg || "";
  el.hidden = !msg;
}

function syncFacePresetChips() {
  const val = String(parseInt($("fbxFaces").value, 10) || 0);
  document.querySelectorAll("#fbxModal .pchip").forEach((btn) => {
    btn.classList.toggle("on", btn.dataset.faces === val);
  });
}

function fbxShowStep(step) {
  $("fbxStepForm").hidden = step !== "form";
  $("fbxStepResult").hidden = step !== "result";
}

function setProbeBusy(on) {
  fbxBusy = on;
  $("btnFbxProbe").disabled = on;
  $("btnFbxConvert").disabled = on;
  $("btnFbxReset").disabled = on;
  $("fbxProbeBusy").hidden = !on;
}

function setConvertBusy(on, text) {
  fbxBusy = on;
  $("btnFbxProbe").disabled = on;
  $("btnFbxConvert").disabled = on;
  $("btnFbxReset").disabled = on;
  $("fbxConvertBusy").hidden = !on;
  if (text != null) $("fbxConvertText").textContent = text;
}

function populateFbxResult(fx) {
  const facesStr = fx.faces != null ? `${fx.faces.toLocaleString()} 면` : "–";
  const vertsStr = fx.verts != null ? `${fx.verts.toLocaleString()} 점` : "–";
  let baseStr = fx.base_faces != null ? `${fx.base_faces.toLocaleString()} 면` : "–";
  if (fx.base_faces && fx.faces && fx.faces < fx.base_faces) {
    const ratio = Math.round((1 - fx.faces / fx.base_faces) * 100);
    baseStr += ` (${ratio}% 경량화)`;
  }
  $("resFaces").textContent = facesStr;
  $("resVerts").textContent = vertsStr;
  $("resBaseFaces").textContent = baseStr;
  $("resBytes").textContent = fmtBytes(fx.fbx_bytes);
  const rigTag = fx.rig ? " · 19본 리깅 ✓" : "";
  $("resVerified").textContent = (fx.verified_by === "blender"
    ? `Binary FBX 7.4 · Blender 검증 ✓${fx.has_texture ? ` (${fx.tex_size || 1024}px 텍스처)` : ""}`
    : "Binary FBX 7.4") + rigTag;

  const img = $("fbxResultImg");
  if (fx.mesh_preview) {
    img.src = resolveUrl(fx.mesh_preview) + `?t=${Date.now()}`;
    img.hidden = false;
  } else {
    img.hidden = true;
  }
  const dlTarget = fx.skinned_fbx || fx.fbx;
  $("btnFbxDownload").href = dlTarget ? resolveUrl(dlTarget) : "#";
  fbxShowStep("result");
}

const FBX_DEFAULTS = {
  quality: "standard",
  method: "marching_cubes",
  opacity: 0.3,
  clean: true,
  smooth: true,
  fill_holes: true,
  rig: true,
  target_faces: 30000,
};

function applyFbxOptions(opts) {
  const o = { ...FBX_DEFAULTS, ...(opts || {}) };
  $("fbxQuality").value = o.quality;
  $("fbxMethod").value = o.method;
  $("fbxOpacity").value = String(o.opacity);
  $("fbxOpacityVal").textContent = Number(o.opacity).toFixed(2);
  $("fbxClean").checked = !!o.clean;
  $("fbxSmooth").checked = !!o.smooth;
  $("fbxFillHoles").checked = !!o.fill_holes;
  $("fbxRig").checked = !!o.rig;
  $("fbxFaces").value = String(o.target_faces);
  syncFacePresetChips();
}

function resetFbxDefaults() {
  if (fbxBusy) return;
  applyFbxOptions(FBX_DEFAULTS);
  $("fbxProbeStat").hidden = true;
  showFbxError("");
  setStatus("FBX 변환 설정을 기본값으로 되돌렸어요");
}

function openFbxModal(name, { forceForm = false } = {}) {
  fbxOpenPly = name;
  $("fbxName").textContent = name;
  showFbxError("");

  // 기존 변환 설정이 있으면 폼에 복원, 없으면 기본값 적용
  const fx = (state.name === name && state.fbx) ? state.fbx : null;
  applyFbxOptions(fx ? {
    quality: fx.quality ?? FBX_DEFAULTS.quality,
    method: fx.method ?? FBX_DEFAULTS.method,
    opacity: fx.opacity ?? FBX_DEFAULTS.opacity,
    clean: fx.clean ?? FBX_DEFAULTS.clean,
    smooth: fx.smooth ?? FBX_DEFAULTS.smooth,
    fill_holes: fx.fill_holes ?? FBX_DEFAULTS.fill_holes,
    rig: fx.rig ?? (!!fx.skin_glb || FBX_DEFAULTS.rig),
    target_faces: fx.target_faces ?? FBX_DEFAULTS.target_faces,
  } : FBX_DEFAULTS);

  if (state.baseFaces && state.name === name) {
    $("fbxBaseFaces").textContent = state.baseFaces.toLocaleString();
    $("fbxBaseVerts").textContent = state.baseVerts != null ? state.baseVerts.toLocaleString() : "–";
    $("fbxKept").textContent = state.kept != null ? state.kept.toLocaleString() : "–";
    $("fbxProbeStat").hidden = false;
  } else {
    $("fbxProbeStat").hidden = true;
  }

  if (!forceForm && fx && fx.fbx) {
    populateFbxResult(fx);
  } else {
    fbxShowStep("form");
  }
  $("fbxModal").classList.add("open");
}

function closeFbxModal() {
  $("fbxModal").classList.remove("open");
  fbxOpenPly = "";
}

function fbxPly() {
  return fbxOpenPly || state.name || "";
}

function readFbxFormOptions() {
  const targetRaw = parseInt($("fbxFaces").value, 10);
  return {
    opacity: +$("fbxOpacity").value,
    quality: $("fbxQuality").value,
    method: $("fbxMethod").value,
    clean: $("fbxClean").checked,
    smooth: $("fbxSmooth").checked,
    fill_holes: $("fbxFillHoles").checked,
    rig: $("fbxRig").checked,
    target_faces: Number.isFinite(targetRaw) && targetRaw >= 0 ? targetRaw : 0,
  };
}

async function fbxPoll(stem, kind) {
  for (let i = 0; i < 1800; i++) {
    await new Promise((r) => setTimeout(r, 900));
    const { job } = await fetch(`/api/job?stem=${encodeURIComponent(stem)}`).then((r) => r.json());
    if (!job) continue;
    if (job.kind === "convert") {
      const label = job.phase === "preview" ? "메시 회전 미리보기 렌더링 중…"
        : job.phase === "reconstruct" ? "표면 재구성 · Binary FBX 생성 중…"
        : "처리 중…";
      setConvertBusy(true, `${label} (${Math.round(job.seconds || 0)}s)`);
    }
    if (job.status === "done") {
      if (kind === "probe") {
        state.baseFaces = job.base_faces;
        state.baseVerts = job.base_verts;
        state.kept = job.kept_points;
        $("fbxBaseFaces").textContent = job.base_faces.toLocaleString();
        $("fbxBaseVerts").textContent = job.base_verts != null ? job.base_verts.toLocaleString() : "–";
        $("fbxKept").textContent = job.kept_points != null ? job.kept_points.toLocaleString() : "–";
        $("fbxProbeStat").hidden = false;
      } else {
        const fx = {
          fbx: job.fbx,
          skin_glb: job.skin_glb,
          skinned_fbx: job.skinned_fbx,
          rig: job.rig,
          mesh_ply: job.mesh_ply,
          mesh_uv: job.mesh_uv,
          texture: job.texture,
          mesh_preview: job.mesh_preview,
          has_texture: job.has_texture,
          tex_size: job.tex_size,
          faces: job.faces,
          verts: job.verts,
          base_faces: job.base_faces,
          fbx_bytes: job.fbx_bytes,
          verified_by: job.verified_by,
          quality: job.quality,
          method: job.method,
          clean: job.clean,
          smooth: job.smooth,
          fill_holes: job.fill_holes,
          opacity: job.opacity,
          target_faces: job.target_faces,
        };
        state.fbx = fx;
        state.baseFaces = job.base_faces;
        state.meshLoadedUrl = ""; // 새 메시 강제 리로드
        state.skinnedLoadedUrl = ""; // 새 스킨 GLB 강제 리로드
        updateHeaderAndDock();
        populateFbxResult(fx);
        setStatus(`FBX 변환 완료 · ${(fx.faces || 0).toLocaleString()}면 (${fmtBytes(fx.fbx_bytes)})`);
      }
      return;
    }
    if (job.status === "error") throw new Error(job.msg || "변환 실패");
  }
  throw new Error("시간이 너무 오래 걸려 취소했어요");
}

async function doFbxProbe() {
  const ply = fbxPly();
  if (!ply || fbxBusy) return;
  showFbxError("");
  setProbeBusy(true);
  try {
    const opts = readFbxFormOptions();
    const res = await fetch("/api/fbx", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ply, action: "probe", ...opts }),
    }).then((r) => r.json());
    if (res.error) throw new Error(res.error);
    await fbxPoll(res.stem, "probe");
  } catch (e) {
    showFbxError(e.message);
    setStatus(e.message, true);
  } finally {
    setProbeBusy(false);
  }
}

async function doFbxConvert() {
  const ply = fbxPly();
  if (!ply || fbxBusy) return;
  showFbxError("");
  setConvertBusy(true, "표면 재구성 · Binary FBX 생성 중…");
  try {
    const opts = readFbxFormOptions();
    const res = await fetch("/api/fbx", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ply, action: "convert", ...opts }),
    }).then((r) => r.json());
    if (res.error) throw new Error(res.error);
    await fbxPoll(res.stem, "convert");
    await refreshSamples();
  } catch (e) {
    showFbxError(e.message);
    setStatus(e.message, true);
  } finally {
    setConvertBusy(false);
  }
}

/* ---------- 샘플 그리드 ---------- */
async function refreshSamples(selectName = "") {
  try {
    const prevChecked = selectName || state.name || "";
    const { plys } = await fetch("/api/plys").then((r) => r.json());
    const box = $("samples");
    if (!plys.length) {
      setBusy("");
      box.innerHTML = `<div class="hint">아직 PLY가 없어요.<br><code>run_idol.py</code>로 생성하거나<br>상단에서 이미지를 변환하세요.</div>`;
      return;
    }

    box.innerHTML = plys.map((p) => {
      const safeName = escapeHtml(p.name);
      const label = escapeHtml(p.name.replace(/\.ply$/i, ""));
      const thumb = p.preview || p.condition;
      const thumbHtml = thumb
        ? `<img class="thumb" src="${resolveUrl(thumb)}" alt="" loading="lazy" onerror="this.outerHTML='<div class=&quot;thumb ph&quot;>✦</div>'">`
        : `<div class="thumb ph">✦</div>`;
      const mi = p.mesh_info || {};
      const fbxLabel = fmtCompactFaces(mi.faces);
      const fbxBtn = p.fbx
        ? `<button class="fbxbadge" title="3D 메시 바로 보기">◇ ${escapeHtml(fbxLabel)}</button>`
        : "";
      const rigBadge = p.skin_glb
        ? `<span class="rigbadge" title="19본 리깅 완료">🦴${(p.animations && p.animations.length) ? ` 🎬${p.animations.length}` : ""}</span>`
        : "";
      return `
      <div class="sample" data-name="${safeName}" title="${label}">
        ${thumbHtml}
        <button class="del" title="결과물 삭제">✕</button>
        <div class="sinfo">
          <div class="sname">${label}</div>
          <div class="ssize">${(p.size / 1024 / 1024).toFixed(1)}MB</div>
          ${(rigBadge || fbxBtn) ? `<div class="sbadges">${fbxBtn}${rigBadge}</div>` : ""}
        </div>
        <div class="check">✓</div>
      </div>`;
    }).join("");

    box.querySelectorAll(".sample").forEach((el) => {
      el.addEventListener("click", async () => {
        try { await loadSample(el.dataset.name); }
        catch (e) { setStatus(e.message, true); }
      });
    });

    /* FBX 배지 클릭: 해당 샘플 로드 후 즉시 FBX 3D 메시 뷰어로 전환 */
    box.querySelectorAll(".fbxbadge").forEach((btn) => {
      btn.addEventListener("click", async (e) => {
        e.stopPropagation();
        const name = btn.closest(".sample").dataset.name;
        try {
          if (state.name !== name) {
            await loadSample(name, { openMeshView: true });
          } else {
            await setViewMode("mesh");
          }
        } catch (err) {
          setStatus(err.message, true);
        }
      });
    });

    box.querySelectorAll(".del").forEach((btn) => {
      let armed = false, timer = null;
      const disarm = () => {
        armed = false;
        btn.classList.remove("arm");
        btn.textContent = "✕";
        btn.title = "결과물 삭제";
      };
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        if (!armed) {
          armed = true;
          btn.classList.add("arm");
          btn.textContent = "정말?";
          btn.title = "한 번 더 클릭하면 삭제됩니다";
          timer = setTimeout(disarm, 3000);
          return;
        }
        clearTimeout(timer);
        disarm();
        deletePly(btn.closest(".sample").dataset.name).catch((err) => {
          setStatus("삭제 실패: " + err.message, true);
        });
      });
    });

    const target = plys.find((p) => p.name === prevChecked) || plys[0];
    if (!state.splat || (selectName && state.name !== selectName)) {
      loadSample(target.name).catch((e) => setStatus(e.message, true));
    } else if (state.name) {
      markActiveSample(state.name);
    }
  } catch (e) {
    setBusy("");
    setStatus("샘플 목록 불러오기 실패: " + e.message, true);
  }
}

/* ---------- 결과물 삭제 ---------- */
async function deletePly(name) {
  const res = await fetch(`/api/ply?name=${encodeURIComponent(name)}`, { method: "DELETE" });
  const data = await res.json().catch(() => ({}));
  if (!res.ok || !data.ok) throw new Error(data.error || `삭제 실패 (${res.status})`);
  const label = String(name).replace(/\.ply$/i, "");
  if (state.name === name) {
    scene = new gs.Scene();
    resetMeshCache();
    state.splat = null;
    state.name = "";
    state.bounds = null;
    state.fbx = null;
    applyMeta(null);
    $("fileName").textContent = "";
    $("fileName").title = "";
    await setViewMode("splat");
    setStatus(`삭제 완료: ${label} (${data.deleted.length}개 파일)`);
  }
  await refreshSamples("");
  return data;
}

/* ---------- 이미지 → 3D 변환 ---------- */
const conv = { file: null, url: null, busy: false, stem: null };

const PHASE_KO = {
  queued: "큐 대기 중…",
  loading: "모델 로딩 중… (첫 실행 시 수 분 소요)",
  running: "모델 준비 중…",
  sampling: "3D 가우시안 샘플링 중…",
  preview: "3D 모델 미리보기 생성 중…",
  done: "완료",
  error: "오류",
};

function convSetUI() {
  const btn = $("btnConvert");
  btn.disabled = conv.busy || !conv.file;
  btn.textContent = conv.busy ? "원스톱 변환 중…" : "원스톱 3D 캐릭터 생성";
}

function showPendingImage(file) {
  if (conv.url) URL.revokeObjectURL(conv.url);
  conv.file = file;
  conv.url = URL.createObjectURL(file);
  const drop = $("imgDrop");
  drop.classList.add("has-img");
  drop.innerHTML =
    `<img id="imgPreview" src="${conv.url}" alt="preview">` +
    `<div class="imgMeta">` +
      `<span class="imgName">${escapeHtml(file.name)}</span>` +
      `<button class="miniBtn" id="btnClearImg" title="이미지 지우기">✕</button>` +
    `</div>`;
  $("btnClearImg").addEventListener("click", (e) => { e.stopPropagation(); clearPendingImage(); });
  convSetUI();
}

function clearPendingImage() {
  if (conv.url) { URL.revokeObjectURL(conv.url); conv.url = null; }
  conv.file = null;
  $("imgDrop").classList.remove("has-img");
  $("imgDrop").innerHTML =
    `<div id="imgDropText"><span class="big">⬆</span>이미지(PNG·JPG)를 여기에 드래그<br>또는 클릭해서 선택</div>`;
  convSetUI();
}

function convStatus(text, cls = "") {
  const el = $("convText");
  el.textContent = text;
  el.className = cls;
}

function setConvStages(phase) {
  const stages = document.querySelectorAll(".stage");
  let idx = 0;
  if (phase === "done") idx = 3;
  else if (phase === "sampling" || phase === "preview") idx = 2;
  else if (phase === "queued" || phase === "loading" || phase === "running") idx = 1;
  stages.forEach((s, i) => {
    s.classList.toggle("active", i === idx);
    s.classList.toggle("done", i < idx);
  });
}

async function pollJob(stem) {
  for (;;) {
    const resp = await fetch(`/api/job?stem=${encodeURIComponent(stem)}`);
    const data = await resp.json();
    const j = data.job;
    if (!j) throw new Error("작업 상태를 찾을 수 없어요");

    const pct = Math.max(0, Math.min(100, j.pct || 0));
    $("convBar").style.width = pct + "%";
    setConvStages(j.phase);
    const phaseKo = j.phase && PHASE_KO[j.phase] ? PHASE_KO[j.phase] : j.phase || "변환 중";
    const pctText = j.phase === "sampling" || j.status === "done" ? ` ${pct}%` : "";
    convStatus(`${phaseKo}${pctText}` + (j.seconds != null ? ` (${j.seconds}s)` : ""));

    if (j.status === "done") {
      convStatus(`3DGS 완료 (${j.seconds}s) — 불러오는 중…`, "done");
      setConvStages("done");
      $("convBar").style.width = "100%";
      const plyPath = j.ply || `outputs/${stem}.ply`;
      const plyName = plyPath.split("/").pop();
      await loadPlyFromUrl(plyPath, plyName);
      await refreshSamples(plyName);
      if ($("convAutoRig") && $("convAutoRig").checked) {
        convStatus("자동 메시·텍스처·19본 리깅 파이프라인 실행 중…");
        await quickAutoConvertAndRig(plyName, { openRigPanel: true });
        convStatus("원스톱 3D 캐릭터 + 리깅 완료 ✓", "done");
      }
      return;
    }
    if (j.status === "error") {
      convStatus(j.msg || "변환에 실패했어요", "error");
      throw new Error(j.msg || "conversion failed");
    }
    await new Promise((res) => setTimeout(res, 1500));
  }
}

async function startConvert() {
  if (!conv.file || conv.busy) return;
  conv.busy = true;
  convSetUI();
  $("convBox").hidden = false;
  $("convBar").style.width = "0%";
  setConvStages("upload");
  convStatus("이미지 업로드 중…");
  try {
    const fd = new FormData();
    fd.append("file", conv.file, conv.file.name);
    fd.append("mode", $("convMode").value);
    fd.append("gaussians", $("convGauss").value);
    const r = await fetch("/api/upload", { method: "POST", body: fd }).then(async (resp) => {
      const text = await resp.text();
      let json = null;
      try { json = JSON.parse(text); } catch {}
      if (!resp.ok) throw new Error((json && json.error) || text || `업로드 실패 (${resp.status})`);
      return json;
    });
    conv.stem = r.stem;
    await pollJob(r.stem);
  } catch (e) {
    convStatus(e.message || "업로드에 실패했어요", "error");
  } finally {
    conv.busy = false;
    convSetUI();
  }
}

/* ---------- 라이트박스 ---------- */
function openLightbox(src, title) {
  if (!src) return;
  $("lbImg").src = src;
  $("lbTitle").textContent = title || "";
  $("lightbox").classList.add("open");
}
function closeLightbox() { $("lightbox").classList.remove("open"); }

/* ---------- events ---------- */
function wireEvents() {
  canvas.addEventListener("mousedown", markInteract);
  window.addEventListener("mouseup", markRelease);
  canvas.addEventListener("wheel", markScroll, { passive: true });
  canvas.addEventListener("touchstart", markInteract, { passive: true });
  window.addEventListener("touchend", markRelease);
  window.addEventListener("touchcancel", markRelease);

  $("btnAuto").addEventListener("click", () => {
    autoRotate = !autoRotate;
    $("btnAuto").classList.toggle("on", autoRotate);
    if (autoRotate && viewMode === "splat") {
      syncAutoOrbitFromCamera(getCenter());
    }
    if (meshViewer) meshViewer.setAutoRotate(autoRotate);
  });

  $("btnRecenter").addEventListener("click", () => {
    if (viewMode === "mesh" && meshViewer) {
      meshViewer.recenter();
    } else {
      frameView();
      resetInteract();
    }
  });

  $("btnSide").addEventListener("click", () => {
    document.body.classList.toggle("side-hidden");
    $("btnSide").classList.toggle("on", document.body.classList.contains("side-hidden"));
  });

  $("btnHome").addEventListener("click", async () => {
    try {
      const { plys } = await fetch("/api/plys").then((r) => r.json());
      if (plys.length) await loadSample(plys[0].name);
      else setStatus("샘플이 없습니다");
    } catch (e) { setStatus(e.message, true); }
  });
  $("btnRefresh").addEventListener("click", () => refreshSamples());

  // 3DGS <-> FBX Mesh <-> Rig/Motion 전환 및 메시 셰이딩 컨트롤
  $("tabSplat").addEventListener("click", () => setViewMode("splat"));
  $("tabMesh").addEventListener("click", () => setViewMode("mesh"));
  $("tabRig").addEventListener("click", () => setViewMode("rig"));
  $("btnShadeColor").addEventListener("click", () => {
    meshViewer.setStyle({ shading: "color" });
    updateMeshStyleUI();
  });
  $("btnShadeUnlit").addEventListener("click", () => {
    meshViewer.setStyle({ shading: "unlit" });
    updateMeshStyleUI();
  });
  $("btnShadeClay").addEventListener("click", () => {
    meshViewer.setStyle({ shading: "clay" });
    updateMeshStyleUI();
  });
  $("btnWireframe").addEventListener("click", () => {
    meshViewer.setStyle({ wireframe: !meshViewer.wireframe });
    updateMeshStyleUI();
  });
  $("btnBoneOverlay").addEventListener("click", () => {
    meshViewer.setBoneShow(!meshViewer.boneShow);
    updateMeshStyleUI();
  });
  $("btnMeshXray").addEventListener("click", () => {
    const next = !meshViewer.meshXray;
    meshViewer.setMeshXray(next);
    if (next && !meshViewer.boneShow) meshViewer.setBoneShow(true);
    if ($("chkEditXray")) $("chkEditXray").checked = next;
    updateMeshStyleUI();
  });
  $("btnMeshConfig").addEventListener("click", () => {
    if (state.name) openFbxModal(state.name, { forceForm: true });
  });

  // 사이드바 FBX 카드 버튼
  $("btnSideViewMesh").addEventListener("click", () => {
    setViewMode(viewMode === "mesh" ? "splat" : "mesh");
  });
  $("btnSideRebuild").addEventListener("click", () => {
    if (state.name) openFbxModal(state.name, { forceForm: true });
  });
  $("fbxSideThumb").addEventListener("click", () => {
    const src = $("fbxSideThumb").getAttribute("src");
    if (src) openLightbox(src, "FBX 메시 회전 미리보기");
  });

  // 헤더 메시/FBX 버튼: 미생성 모델은 1클릭 자동 변환+리깅, 이미 있으면 상세 설정 모달
  $("btnFbx").addEventListener("click", () => {
    if (!state.name) return;
    if (state.fbx && state.fbx.mesh_ply) openFbxModal(state.name, { forceForm: true });
    else quickAutoConvertAndRig(state.name, { openRigPanel: false });
  });
  $("fbxClose").addEventListener("click", closeFbxModal);
  $("fbxModal").addEventListener("click", (e) => { if (e.target === $("fbxModal")) closeFbxModal(); });
  $("fbxOpacity").addEventListener("input", () => {
    $("fbxOpacityVal").textContent = Number($("fbxOpacity").value).toFixed(2);
  });
  $("fbxFaces").addEventListener("input", syncFacePresetChips);
  document.querySelectorAll("#fbxModal .pchip").forEach((chip) => {
    chip.addEventListener("click", () => {
      $("fbxFaces").value = chip.dataset.faces;
      syncFacePresetChips();
    });
  });
  $("btnFbxProbe").addEventListener("click", doFbxProbe);
  $("btnFbxReset").addEventListener("click", resetFbxDefaults);
  $("btnFbxConvert").addEventListener("click", doFbxConvert);
  $("btnFbxView3d").addEventListener("click", async () => {
    closeFbxModal();
    await setViewMode("mesh");
  });
  $("btnFbxAgain").addEventListener("click", () => {
    showFbxError("");
    fbxShowStep("form");
  });

  // 조건 이미지 → 라이트박스
  $("condFrame").addEventListener("click", () => {
    const src = $("conditionImg").getAttribute("src");
    if (src) openLightbox(src, "조건 이미지");
  });
  $("lightbox").addEventListener("click", (e) => {
    if (e.target === $("lightbox") || e.target.classList.contains("lb-close")) closeLightbox();
  });

  // 드래그 & 드롭 (전체)
  const drop = $("dropZone");
  window.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
  window.addEventListener("dragleave", (e) => { if (e.target === window) drop.classList.remove("over"); });
  window.addEventListener("drop", async (e) => {
    e.preventDefault();
    drop.classList.remove("over");
    const f = e.dataTransfer.files && e.dataTransfer.files[0];
    if (!f) return;
    try {
      const low = f.name.toLowerCase();
      if (low.endsWith(".ply")) await loadPlyFromFile(f);
      else if (low.endsWith(".fbx") || low.endsWith(".bvh") || low.endsWith(".glb")) {
        if (!state.name || !state.isSample) throw new Error("먼저 모션을 적용할 캐릭터 모델을 선택하세요");
        await uploadMotionFile(f);
      } else if (/^image\//.test(f.type)) {
        showPendingImage(f);
        setStatus(`${f.name} 선택됨 — 사이드바 "원스톱 3D 캐릭터 생성"으로 자동 변환`);
      } else setStatus("이미지, PLY, 또는 Mixamo 모션(.fbx/.glb/.bvh) 파일을 놓을 수 있어요", true);
    } catch (err) {
      setStatus(err.message, true);
    }
  });

  // 이미지 → 3D
  const imgDrop = $("imgDrop");
  const imgInput = $("imgFile");
  const acceptImage = (f) => {
    if (f && /^image\//.test(f.type)) showPendingImage(f);
  };
  imgDrop.addEventListener("click", () => { if (!conv.busy) imgInput.click(); });
  imgInput.addEventListener("change", () => {
    acceptImage(imgInput.files[0]);
    imgInput.value = "";
  });
  imgDrop.addEventListener("dragover", (e) => {
    e.preventDefault(); e.stopPropagation();
    imgDrop.classList.add("over");
  });
  imgDrop.addEventListener("dragleave", () => imgDrop.classList.remove("over"));
  imgDrop.addEventListener("drop", (e) => {
    e.preventDefault(); e.stopPropagation();
    imgDrop.classList.remove("over");
    acceptImage(e.dataTransfer.files[0]);
  });
  $("btnConvert").addEventListener("click", startConvert);

  // 키보드 단축키
  window.addEventListener("keydown", (e) => {
    const tag = document.activeElement?.tagName;
    if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
    if (e.code === "Space") { e.preventDefault(); $("btnAuto").click(); }
    else if (e.key === "r" || e.key === "R") { $("btnRecenter").click(); }
    else if (e.key === "b" || e.key === "B") { $("btnSide").click(); }
    else if (e.key === "m" || e.key === "M") {
      if (state.fbx && state.fbx.mesh_ply) {
        setViewMode(viewMode === "mesh" ? "splat" : "mesh");
      }
    }
    else if (e.key === "w" || e.key === "W") {
      if (viewMode === "mesh" && meshViewer) {
        meshViewer.setStyle({ wireframe: !meshViewer.wireframe });
        updateMeshStyleUI();
      }
    }
    else if (e.key === "Escape") {
      if ($("fbxModal").classList.contains("open")) closeFbxModal();
      else closeLightbox();
    }
  });
}

/* ---------- boot ---------- */
(function boot() {
  setBusy("WebGL 렌더러 초기화 중…");
  try {
    makeRenderer();
    makeCamera();
    scene = new gs.Scene();
    controls = new gs.OrbitControls(camera, canvas);
    meshViewer = new MeshViewer(meshCanvas);
    updateMeshStyleUI();

    scheduleNext();
    wireEvents();
    refreshSamples();
  } catch (e) {
    setBusy("");
    setStatus("뷰어 초기화 실패: " + e.message, true);
    console.error(e);
  }
})();
