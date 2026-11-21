/* Interactive WebGL2 Triangle Mesh / FBX Previewer (Z-up, zero external deps)
 * Renders FBX-verified binary/ASCII PLY triangle meshes with vertex colors,
 * studio clay shading, and GPU barycentric wireframe overlay.
 */

const VS_SRC = `#version 300 es
precision highp float;
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec3 aColor;
layout(location = 3) in vec3 aBary;
layout(location = 4) in vec2 aUV;

uniform mat4 uProj;
uniform mat4 uView;

out vec3 vPosView;
out vec3 vNormalView;
out vec3 vColor;
out vec3 vBary;
out vec2 vUV;

void main() {
  vec4 vp = uView * vec4(aPos, 1.0);
  vPosView = vp.xyz;
  vNormalView = normalize(mat3(uView) * aNormal);
  vColor = aColor;
  vBary = aBary;
  vUV = aUV;
  gl_Position = uProj * vp;
}
`;

const FS_SRC = `#version 300 es
precision highp float;

in vec3 vPosView;
in vec3 vNormalView;
in vec3 vColor;
in vec3 vBary;
in vec2 vUV;

uniform int uMode;       // 0 = color (faithful high-key lit), 1 = clay, 2 = unlit (1:1 3DGS color)
uniform int uWireframe;  // 0 = off, 1 = overlay
uniform int uHasTex;     // 0 = vertex color, 1 = baked UV texture
uniform sampler2D uTex;
uniform vec3 uWireColor;
uniform float uAlpha;

out vec4 fragColor;

float edgeFactor() {
  vec3 d = fwidth(vBary);
  vec3 a3 = smoothstep(vec3(0.0), d * 1.35, vBary);
  return min(min(a3.x, a3.y), a3.z);
}

void main() {
  vec3 albedo = (uHasTex == 1) ? texture(uTex, vUV).rgb : vColor;
  vec3 lit;
  if (uMode == 2) {
    // 3DGS와 동일한 무조명 원색 모드
    lit = albedo;
  } else {
    vec3 N = normalize(vNormalView);
    if (!gl_FrontFacing) N = -N;
    vec3 V = normalize(-vPosView);

    // Two-point studio key + rim/fill lighting in view space
    vec3 L1 = normalize(vec3(0.45, 0.75, 0.65));
    vec3 L2 = normalize(vec3(-0.65, -0.35, 0.55));
    float diff1 = max(dot(N, L1), 0.0);
    float diff2 = max(dot(N, L2), 0.0);
    vec3 H1 = normalize(L1 + V);
    float spec = pow(max(dot(N, H1), 0.0), 28.0);
    float rim = pow(1.0 - max(dot(N, V), 0.0), 2.5);

    if (uMode == 1) {
      vec3 base = vec3(0.84, 0.79, 0.74);
      lit = base * (0.24 + diff1 * 0.68 + diff2 * 0.26)
          + vec3(1.0, 0.96, 0.92) * spec * 0.24
          + vec3(1.0, 0.62, 0.28) * rim * 0.14;
    } else {
      // High-ambient lighting preserves bright 3DGS colors without muddy shadows
      lit = albedo * (0.76 + diff1 * 0.24 + diff2 * 0.10)
          + vec3(1.0, 0.98, 0.95) * spec * 0.05;
    }
  }

  if (uWireframe == 1) {
    float ef = edgeFactor();
    lit = mix(uWireColor, lit, ef);
  }

  fragColor = vec4(clamp(lit, 0.0, 1.0), uAlpha);
}
`;

const VS_SKINNED_SRC = `#version 300 es
precision highp float;
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec3 aColor;
layout(location = 3) in vec4 aJ;
layout(location = 4) in vec4 aW;
layout(location = 5) in vec3 aBary;
layout(location = 6) in vec2 aUV;

uniform mat4 uProj;
uniform mat4 uView;
uniform mat4 uBones[20];
uniform mat4 uIBMs[20];

out vec3 vPosView;
out vec3 vNormalView;
out vec3 vColor;
out vec3 vBary;
out vec2 vUV;

void main() {
  vec4 j = aJ;
  // glTF skinning: M = sum_j W_j * world_j * IBM_j
  mat4 m =
    (uBones[int(round(j.x))] * uIBMs[int(round(j.x))]) * aW.x
  + (uBones[int(round(j.y))] * uIBMs[int(round(j.y))]) * aW.y
  + (uBones[int(round(j.z))] * uIBMs[int(round(j.z))]) * aW.z
  + (uBones[int(round(j.w))] * uIBMs[int(round(j.w))]) * aW.w;
  vec3 sp = (m * vec4(aPos, 1.0)).xyz;
  vec3 sn = normalize(mat3(m[0].xyz, m[1].xyz, m[2].xyz) * aNormal);
  vec4 vp = uView * vec4(sp, 1.0);
  vPosView = vp.xyz;
  vNormalView = normalize(mat3(uView) * sn);
  vColor = aColor;
  vBary = aBary;
  vUV = aUV;
  gl_Position = uProj * vp;
}
`;

const FS_SKINNED_SRC = FS_SRC; // identical fragment stage (uMode/uWireframe/uHasTex)

// X-Ray 3D octahedral bone overlay (per-vertex colored triangles/lines)
const VS_BONE_SRC = `#version 300 es
precision highp float;
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec4 aColor;
uniform mat4 uProj;
uniform mat4 uView;
out vec4 vColor;
void main() {
  vColor = aColor;
  gl_Position = uProj * (uView * vec4(aPos, 1.0));
}
`;
const FS_BONE_SRC = `#version 300 es
precision highp float;
in vec4 vColor;
out vec4 fragColor;
void main() { fragColor = vColor; }
`;

// Anti-aliased circular joint markers with dark halo border (X-ray)
const VS_JOINT_SRC = `#version 300 es
precision highp float;
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec4 aColor;
layout(location = 2) in float aSize;
uniform mat4 uProj;
uniform mat4 uView;
out vec4 vColor;
void main() {
  vColor = aColor;
  gl_Position = uProj * (uView * vec4(aPos, 1.0));
  gl_PointSize = aSize;
}
`;
const FS_JOINT_SRC = `#version 300 es
precision highp float;
in vec4 vColor;
out vec4 fragColor;
void main() {
  vec2 c = gl_PointCoord - vec2(0.5);
  float r = length(c) * 2.0;
  if (r > 1.0) discard;
  float border = smoothstep(0.66, 0.78, r);
  float aa = 1.0 - smoothstep(0.88, 1.0, r);
  vec3 col = mix(vColor.rgb, vec3(0.05, 0.05, 0.08), border * 0.85);
  // Subtle inner highlight for 3D bead look
  float hi = smoothstep(0.55, 0.0, length(c + vec2(0.14, 0.14)) * 2.0);
  col = mix(col, vec3(1.0), hi * 0.28 * (1.0 - border));
  fragColor = vec4(col, vColor.a * aa);
}
`;

function mat4Identity() {
  return new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
}

function quatToMat4(q, out) {
  const [x, y, z, w] = q;
  const xx = x * x, yy = y * y, zz = z * z;
  const xy = x * y, xz = x * z, yz = y * z;
  const wx = w * x, wy = w * y, wz = w * z;
  out[0] = 1 - 2 * (yy + zz); out[1] = 2 * (xy + wz);   out[2] = 2 * (xz - wy);   out[3] = 0;
  out[4] = 2 * (xy - wz);     out[5] = 1 - 2 * (xx + zz); out[6] = 2 * (yz + wx); out[7] = 0;
  out[8] = 2 * (xz + wy);     out[9] = 2 * (yz - wx);   out[10] = 1 - 2 * (xx + yy); out[11] = 0;
  out[12] = 0; out[13] = 0; out[14] = 0; out[15] = 1;
  return out;
}

function mat4ComposeTRS(t, q, s, out) {
  quatToMat4(q, out);
  out[0] *= s[0]; out[1] *= s[0]; out[2] *= s[0];
  out[4] *= s[1]; out[5] *= s[1]; out[6] *= s[1];
  out[8] *= s[2]; out[9] *= s[2]; out[10] *= s[2];
  out[12] = t[0]; out[13] = t[1]; out[14] = t[2];
  return out;
}

// Build a column-major TRS matrix from an Euler angle triple (DEGREES) + scale +
// translation. Rotation order: R = Rx * Ry * Rz (intrinsic Z-Y-X / extrinsic X-Y-Z).
// Used for the interactive bone-pose panel so the UI stays quaternion-free.
function mat4FromEulerDeg(t, eDeg, s) {
  const toR = (d) => (d * Math.PI) / 180;
  const [ax, ay, az] = eDeg;
  const cx = Math.cos(toR(ax)), sx = Math.sin(toR(ax));
  const cy = Math.cos(toR(ay)), sy = Math.sin(toR(ay));
  const cz = Math.cos(toR(az)), sz = Math.sin(toR(az));
  // Rx * Ry * Rz
  const m00 = cx * cy;
  const m01 = cx * sy * sz - sx * cz;
  const m02 = cx * sy * cz + sx * sz;
  const m10 = sx * cy;
  const m11 = sx * sy * sz + cx * cz;
  const m12 = sx * sy * cz - cx * sz;
  const m20 = -sy;
  const m21 = cy * sz;
  const m22 = cy * cz;
  const out = new Float32Array(16);
  out[0] = m00 * s[0]; out[1] = m10 * s[0]; out[2] = m20 * s[0]; out[3] = 0;
  out[4] = m01 * s[1]; out[5] = m11 * s[1]; out[6] = m21 * s[1]; out[7] = 0;
  out[8] = m02 * s[2]; out[9] = m12 * s[2]; out[10] = m22 * s[2]; out[11] = 0;
  out[12] = t[0]; out[13] = t[1]; out[14] = t[2]; out[15] = 1;
  return out;
}

function mat4Invert(m) {
  // Standard gl-matrix mat4.invert (column-major out[c*4+r]).
  const o = new Float32Array(16);
  const a00 = m[0], a01 = m[1], a02 = m[2], a03 = m[3];
  const a10 = m[4], a11 = m[5], a12 = m[6], a13 = m[7];
  const a20 = m[8], a21 = m[9], a22 = m[10], a23 = m[11];
  const a30 = m[12], a31 = m[13], a32 = m[14], a33 = m[15];
  const b00 = a00 * a11 - a01 * a10, b01 = a00 * a12 - a02 * a10;
  const b02 = a00 * a13 - a03 * a10, b03 = a01 * a12 - a02 * a11;
  const b04 = a01 * a13 - a03 * a11, b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30, b07 = a20 * a32 - a22 * a30;
  const b08 = a20 * a33 - a23 * a30, b09 = a21 * a32 - a22 * a31;
  const b10 = a21 * a33 - a23 * a31, b11 = a22 * a33 - a23 * a32;
  let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (!det) return mat4Identity();
  det = 1.0 / det;
  o[0] = (a11 * b11 - a12 * b10 + a13 * b09) * det;
  o[1] = (a02 * b10 - a01 * b11 - a03 * b09) * det;
  o[2] = (a31 * b05 - a32 * b04 + a33 * b03) * det;
  o[3] = (a22 * b04 - a21 * b05 - a23 * b03) * det;
  o[4] = (a12 * b08 - a10 * b11 - a13 * b07) * det;
  o[5] = (a00 * b11 - a02 * b08 + a03 * b07) * det;
  o[6] = (a30 * b05 - a32 * b04 + a33 * b03) * det;
  o[7] = (a22 * b04 - a20 * b05 - a23 * b03) * det;
  o[8] = (a10 * b10 - a11 * b08 + a13 * b06) * det;
  o[9] = (a01 * b08 - a00 * b10 - a03 * b06) * det;
  o[10] = (a30 * b04 - a31 * b02 + a33 * b00) * det;
  o[11] = (a21 * b02 - a20 * b04 - a23 * b00) * det;
  o[12] = (a11 * b07 - a10 * b09 - a12 * b06) * det;
  o[13] = (a00 * b09 - a01 * b07 + a02 * b06) * det;
  o[14] = (a31 * b01 - a30 * b03 + a32 * b00) * det;
  o[15] = (a20 * b03 - a21 * b01 + a22 * b00) * det;
  return o;
}
// Exported so rigs/_verify_math.js can import and cross-check the REAL viewer code.
export { mat4Invert as _mat4Invert };

/* Minimal glTF 2.0 GLB reader for skinned meshes (1 primitive, 1 skin, <= 20 joints).
 * Returns {pos, normal, color, joints, weights, indices, boneNames, joints, ibm, bounds}. */
function parseSkinnedGlb(buffer) {
  const u8 = new Uint8Array(buffer);
  const dv = new DataView(buffer);
  if (dv.getUint32(0, true) !== 0x46546c67) throw new Error("not a GLB");
  const jsonLen = dv.getUint32(12, true);
  const json = JSON.parse(new TextDecoder().decode(u8.subarray(20, 20 + jsonLen)));
  let binOff = 20 + jsonLen;
  // GLB chunk: <length:4><type:4> — verify type at binOff+4, data starts at binOff+8
  if (binOff + 8 > u8.length || dv.getUint32(binOff + 4, true) !== 0x004e4942) throw new Error("missing BIN chunk");
  const bin = buffer.slice(binOff + 8);
  const accData = (accId) => {
    const a = json.accessors[accId];
    const bv = json.bufferViews[a.bufferView];
    const sizes = { 5126: 4, 5123: 2, 5125: 4 };
    const types = { 5126: Float32Array, 5123: Uint16Array, 5125: Uint32Array };
    const n = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT4: 16 }[a.type];
    return new types[a.componentType](bin, (bv.byteOffset || 0) + (a.byteOffset || 0), a.count * n);
  };

  // find skinned mesh node
  let meshNode = -1;
  for (const n of json.nodes) {
    if (typeof n.skin === "number" && typeof n.mesh === "number") { meshNode = json.nodes.indexOf(n); break; }
  }
  if (meshNode < 0) {
    for (let i = 0; i < json.nodes.length; i++) {
      if (typeof json.nodes[i].mesh === "number") { meshNode = i; break; }
    }
  }
  if (meshNode < 0) throw new Error("no mesh node in GLB");
  const meshDef = json.nodes[meshNode].mesh;
  const prim = json.meshes[meshDef].primitives[0];
  const A = prim.attributes;

  const skinIdx = json.nodes[meshNode].skin;
  const skin = skinIdx != null ? json.skins[skinIdx] : null;
  let joints = skin ? skin.joints : [];
  let ibm = null;
  if (skin && skin.inverseBindingMatrices != null) {
    const ibmAccs = skin.inverseBindingMatrices; // accessor ids, one MAT4 per joint
    if (Array.isArray(ibmAccs) && ibmAccs.length > 0) {
      const one = accData(ibmAccs[0]);
      const stride = one.length / ibmAccs.length; // 16 per joint
      ibm = new Float32Array(ibmAccs.length * 16);
      for (let k = 0; k < ibmAccs.length; k++) ibm.set(accData(ibmAccs[k]), k * 16);
    } else {
      ibm = accData(ibmAccs);
    }
  }
  // JOINTS_0 indices map directly onto skin.joints order -> uBones[] order
  const jointNodeIds = joints;

  const pos = accData(A.POSITION);
  const color = A.COLOR_0 != null ? accData(A.COLOR_0) : null;
  const jointsArr = A.JOINTS_0 != null ? accData(A.JOINTS_0) : null;
  const weights = A.WEIGHT_0 != null ? accData(A.WEIGHT_0) : null;
  const indices = accData(json.meshes[meshDef].primitives[0].indices);

  const bounds = { min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] };
  for (let i = 0; i < pos.length; i += 3) {
    for (let k = 0; k < 3; k++) {
      const v = pos[i + k];
      if (v < bounds.min[k]) bounds.min[k] = v;
      if (v > bounds.max[k]) bounds.max[k] = v;
    }
  }

  const boneNames = jointNodeIds.map((id) => json.nodes[id].name || "bone");
  const nodeToJoint = new Map();
  for (let j = 0; j < jointNodeIds.length; j++) nodeToJoint.set(jointNodeIds[j], j);

  const animations = [];
  for (const anim of json.animations || []) {
    const tracks = new Map(); // jointIdx -> { translation?, rotation?, scale? }
    let duration = 0;
    for (const ch of anim.channels || []) {
      const nid = ch.target && ch.target.node;
      const path = ch.target && ch.target.path;
      if (nid == null || !nodeToJoint.has(nid)) continue;
      if (path !== "translation" && path !== "rotation" && path !== "scale") continue;
      const ji = nodeToJoint.get(nid);
      const sp = anim.samplers[ch.sampler];
      if (!sp) continue;
      const times = accData(sp.input);
      const values = accData(sp.output);
      if (times.length > 0 && times[times.length - 1] > duration) {
        duration = times[times.length - 1];
      }
      let tr = tracks.get(ji);
      if (!tr) { tr = {}; tracks.set(ji, tr); }
      tr[path] = {
        times,
        values,
        interp: sp.interpolation || "LINEAR",
      };
    }
    if (tracks.size > 0) {
      animations.push({
        name: anim.name || `Animation ${animations.length + 1}`,
        duration,
        tracks,
      });
    }
  }

  return {
    json,
    pos,
    normal: A.NORMAL != null ? accData(A.NORMAL) : null,
    color,
    joints: jointsArr,
    weights,
    indices,
    jointsNodeIds: jointNodeIds,
    boneNames,
    ibm,
    bounds,
    animations,
  };
}

function compileShader(gl, type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(sh);
    gl.deleteShader(sh);
    throw new Error("Mesh shader compile error: " + log);
  }
  return sh;
}

function createProgram(gl, vsSrc, fsSrc) {
  const vs = compileShader(gl, gl.VERTEX_SHADER, vsSrc);
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, fsSrc);
  const prog = gl.createProgram();
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    throw new Error("Mesh program link error: " + gl.getProgramInfoLog(prog));
  }
  return prog;
}

function mat4Perspective(fovyRad, aspect, near, far) {
  const f = 1.0 / Math.tan(fovyRad * 0.5);
  const nf = 1.0 / (near - far);
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) * nf, -1,
    0, 0, 2 * far * near * nf, 0,
  ]);
}

function mat4Mul(a, b) {
  const out = new Float32Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
      out[c * 4 + r] = s;
    }
  }
  return out;
}

function mat4LookAt(eye, target, up) {
  let zx = eye[0] - target[0];
  let zy = eye[1] - target[1];
  let zz = eye[2] - target[2];
  let len = Math.hypot(zx, zy, zz) || 1;
  zx /= len; zy /= len; zz /= len;

  let xx = up[1] * zz - up[2] * zy;
  let xy = up[2] * zx - up[0] * zz;
  let xz = up[0] * zy - up[1] * zx;
  len = Math.hypot(xx, xy, xz);
  if (len < 1e-6) {
    xx = 1; xy = 0; xz = 0;
  } else {
    xx /= len; xy /= len; xz /= len;
  }

  const yx = zy * xz - zz * xy;
  const yy = zz * xx - zx * xz;
  const yz = zx * xy - zy * xx;

  return new Float32Array([
    xx, yx, zx, 0,
    xy, yy, zy, 0,
    xz, yz, zz, 0,
    -(xx * eye[0] + xy * eye[1] + xz * eye[2]),
    -(yx * eye[0] + yy * eye[1] + yz * eye[2]),
    -(zx * eye[0] + zy * eye[1] + zz * eye[2]),
    1,
  ]);
}

export function parseMeshPly(buffer, uvBuffer = null) {
  const bytes = new Uint8Array(buffer);
  // Locate "end_header\n" (or "\r\n") in first 4 KB
  const probeLen = Math.min(bytes.length, 4096);
  const hdrText = new TextDecoder("ascii").decode(bytes.subarray(0, probeLen));
  const m = hdrText.match(/end_header\r?\n/);
  if (!m) throw new Error("잘못된 PLY 헤더입니다");
  const headerStr = hdrText.slice(0, m.index + m[0].length);
  const bodyOffset = new TextEncoder().encode(headerStr).length;

  let format = "ascii";
  let vertexCount = 0;
  let faceCount = 0;
  let curElem = "";
  const vProps = [];
  const fProps = [];

  for (const rawLine of headerStr.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith("comment") || line === "ply" || line === "end_header") continue;
    const parts = line.split(/\s+/);
    if (parts[0] === "format") {
      format = parts[1];
    } else if (parts[0] === "element") {
      curElem = parts[1];
      if (curElem === "vertex") vertexCount = parseInt(parts[2], 10);
      else if (curElem === "face") faceCount = parseInt(parts[2], 10);
    } else if (parts[0] === "property") {
      if (curElem === "vertex") {
        vProps.push({ type: parts[1], name: parts[2] });
      } else if (curElem === "face") {
        if (parts[1] === "list") fProps.push({ list: true, cntType: parts[2], idxType: parts[3], name: parts[4] });
        else fProps.push({ list: false, type: parts[1], name: parts[2] });
      }
    }
  }

  if (format !== "binary_little_endian") {
    throw new Error("지원하지 않는 메시 PLY 포맷: " + format);
  }

  const TYPE_INFO = {
    char: [1, "getInt8"], int8: [1, "getInt8"],
    uchar: [1, "getUint8"], uint8: [1, "getUint8"],
    short: [2, "getInt16"], int16: [2, "getInt16"],
    ushort: [2, "getUint16"], uint16: [2, "getUint16"],
    int: [4, "getInt32"], int32: [4, "getInt32"],
    uint: [4, "getUint32"], uint32: [4, "getUint32"],
    float: [4, "getFloat32"], float32: [4, "getFloat32"],
    double: [8, "getFloat64"], float64: [8, "getFloat64"],
  };

  let vStride = 0;
  const vOffsets = {};
  for (const p of vProps) {
    const ti = TYPE_INFO[p.type];
    if (!ti) throw new Error("알 수 없는 PLY 속성 타입: " + p.type);
    vOffsets[p.name] = { off: vStride, size: ti[0], getter: ti[1] };
    vStride += ti[0];
  }

  const dv = new DataView(buffer, bodyOffset);
  const posArr = new Float32Array(vertexCount * 3);
  const colArr = new Float32Array(vertexCount * 3);
  const hasColor = Boolean(vOffsets.red && vOffsets.green && vOffsets.blue);

  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

  for (let i = 0; i < vertexCount; i++) {
    const base = i * vStride;
    const x = dv[vOffsets.x.getter](base + vOffsets.x.off, true);
    const y = dv[vOffsets.y.getter](base + vOffsets.y.off, true);
    const z = dv[vOffsets.z.getter](base + vOffsets.z.off, true);
    posArr[i * 3] = x;
    posArr[i * 3 + 1] = y;
    posArr[i * 3 + 2] = z;
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
    if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;

    if (hasColor) {
      colArr[i * 3] = dv[vOffsets.red.getter](base + vOffsets.red.off, true) / 255.0;
      colArr[i * 3 + 1] = dv[vOffsets.green.getter](base + vOffsets.green.off, true) / 255.0;
      colArr[i * 3 + 2] = dv[vOffsets.blue.getter](base + vOffsets.blue.off, true) / 255.0;
    } else {
      colArr[i * 3] = 0.78;
      colArr[i * 3 + 1] = 0.78;
      colArr[i * 3 + 2] = 0.80;
    }
  }

  let off = vertexCount * vStride;
  const tris = [];
  for (let i = 0; i < faceCount; i++) {
    for (const fp of fProps) {
      if (fp.list) {
        const cTi = TYPE_INFO[fp.cntType];
        const iTi = TYPE_INFO[fp.idxType];
        const cnt = dv[cTi[1]](off, true);
        off += cTi[0];
        if (cnt === 3) {
          const i0 = dv[iTi[1]](off, true);
          const i1 = dv[iTi[1]](off + iTi[0], true);
          const i2 = dv[iTi[1]](off + iTi[0] * 2, true);
          tris.push(i0, i1, i2);
        } else if (cnt > 3) {
          const i0 = dv[iTi[1]](off, true);
          for (let k = 1; k < cnt - 1; k++) {
            const ik = dv[iTi[1]](off + iTi[0] * k, true);
            const ik1 = dv[iTi[1]](off + iTi[0] * (k + 1), true);
            tris.push(i0, ik, ik1);
          }
        }
        off += cnt * iTi[0];
      } else {
        off += TYPE_INFO[fp.type][0];
      }
    }
  }

  const triIndices = new Uint32Array(tris);
  const triCount = (triIndices.length / 3) | 0;

  // Compute smooth per-vertex normals
  const vNorm = new Float32Array(vertexCount * 3);
  for (let t = 0; t < triCount; t++) {
    const a = triIndices[t * 3];
    const b = triIndices[t * 3 + 1];
    const c = triIndices[t * 3 + 2];
    const ax = posArr[a * 3], ay = posArr[a * 3 + 1], az = posArr[a * 3 + 2];
    const ux = posArr[b * 3] - ax, uy = posArr[b * 3 + 1] - ay, uz = posArr[b * 3 + 2] - az;
    const vx = posArr[c * 3] - ax, vy = posArr[c * 3 + 1] - ay, vz = posArr[c * 3 + 2] - az;
    const nx = uy * vz - uz * vy;
    const ny = uz * vx - ux * vz;
    const nz = ux * vy - uy * vx;
    vNorm[a * 3] += nx; vNorm[a * 3 + 1] += ny; vNorm[a * 3 + 2] += nz;
    vNorm[b * 3] += nx; vNorm[b * 3 + 1] += ny; vNorm[b * 3 + 2] += nz;
    vNorm[c * 3] += nx; vNorm[c * 3 + 1] += ny; vNorm[c * 3 + 2] += nz;
  }
  for (let i = 0; i < vertexCount; i++) {
    const x = vNorm[i * 3], y = vNorm[i * 3 + 1], z = vNorm[i * 3 + 2];
    const l = Math.hypot(x, y, z) || 1;
    vNorm[i * 3] = x / l; vNorm[i * 3 + 1] = y / l; vNorm[i * 3 + 2] = z / l;
  }

  // Expand to non-indexed triangle buffers with barycentric + UV coords (14 floats/vertex)
  const totalVerts = triCount * 3;
  const interleaved = new Float32Array(totalVerts * 14);
  const uvArr = uvBuffer ? new Float32Array(uvBuffer) : null;
  const hasUV = uvArr && uvArr.length === totalVerts * 2;
  const barys = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];
  for (let t = 0; t < triCount; t++) {
    for (let k = 0; k < 3; k++) {
      const cornerIdx = t * 3 + k;
      const vi = triIndices[cornerIdx];
      const out = cornerIdx * 14;
      interleaved[out + 0] = posArr[vi * 3];
      interleaved[out + 1] = posArr[vi * 3 + 1];
      interleaved[out + 2] = posArr[vi * 3 + 2];
      interleaved[out + 3] = vNorm[vi * 3];
      interleaved[out + 4] = vNorm[vi * 3 + 1];
      interleaved[out + 5] = vNorm[vi * 3 + 2];
      interleaved[out + 6] = colArr[vi * 3];
      interleaved[out + 7] = colArr[vi * 3 + 1];
      interleaved[out + 8] = colArr[vi * 3 + 2];
      interleaved[out + 9] = barys[k][0];
      interleaved[out + 10] = barys[k][1];
      interleaved[out + 11] = barys[k][2];
      interleaved[out + 12] = hasUV ? uvArr[cornerIdx * 2] : 0;
      interleaved[out + 13] = hasUV ? uvArr[cornerIdx * 2 + 1] : 0;
    }
  }

  return {
    vertexCount,
    faceCount: triCount,
    drawVerts: totalVerts,
    hasUV: !!hasUV,
    interleaved,
    bounds: {
      min: [minX, minY, minZ],
      max: [maxX, maxY, maxZ],
    },
  };
}

export class MeshViewer {
  constructor(canvas) {
    this.canvas = canvas;
    this.gl = canvas.getContext("webgl2", { antialias: true, powerPreference: "high-performance" });
    if (!this.gl) throw new Error("WebGL2 not supported for mesh viewer");

    const gl = this.gl;
    this.prog = createProgram(gl, VS_SRC, FS_SRC);
    this.uProj = gl.getUniformLocation(this.prog, "uProj");
    this.uView = gl.getUniformLocation(this.prog, "uView");
    this.uMode = gl.getUniformLocation(this.prog, "uMode");
    this.uWire = gl.getUniformLocation(this.prog, "uWireframe");
    this.uHasTex = gl.getUniformLocation(this.prog, "uHasTex");
    this.uTex = gl.getUniformLocation(this.prog, "uTex");
    this.uWireCol = gl.getUniformLocation(this.prog, "uWireColor");
    this.uAlpha = gl.getUniformLocation(this.prog, "uAlpha");

    // Skinned (GLB) pipeline
    this.skProg = createProgram(gl, VS_SKINNED_SRC, FS_SKINNED_SRC);
    this.suProj = gl.getUniformLocation(this.skProg, "uProj");
    this.suView = gl.getUniformLocation(this.skProg, "uView");
    this.suMode = gl.getUniformLocation(this.skProg, "uMode");
    this.suWire = gl.getUniformLocation(this.skProg, "uWireframe");
    this.suHasTex = gl.getUniformLocation(this.skProg, "uHasTex");
    this.suTex = gl.getUniformLocation(this.skProg, "uTex");
    this.suWireCol = gl.getUniformLocation(this.skProg, "uWireColor");
    this.suAlpha = gl.getUniformLocation(this.skProg, "uAlpha");
    this.suBones = gl.getUniformLocation(this.skProg, "uBones[0]") || gl.getUniformLocation(this.skProg, "uBones");
    this.suIBMs = gl.getUniformLocation(this.skProg, "uIBMs[0]") || gl.getUniformLocation(this.skProg, "uIBMs");

    this.vao = gl.createVertexArray();
    this.vbo = gl.createBuffer();
    this.skVao = gl.createVertexArray();
    this.skVbo = gl.createBuffer();
    this.skIbo = gl.createBuffer();

    // X-Ray 3D octahedral bone overlay (pos3 + rgba4 = 7 floats)
    this.boneProg = createProgram(gl, VS_BONE_SRC, FS_BONE_SRC);
    this.buProj = gl.getUniformLocation(this.boneProg, "uProj");
    this.buView = gl.getUniformLocation(this.boneProg, "uView");
    this.boneVao = gl.createVertexArray();
    this.boneVbo = gl.createBuffer();
    gl.bindVertexArray(this.boneVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.boneVbo);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 28, 0);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 4, gl.FLOAT, false, 28, 12);
    gl.bindVertexArray(null);

    // Joint circular markers (pos3 + rgba4 + size1 = 8 floats)
    this.jointProg = createProgram(gl, VS_JOINT_SRC, FS_JOINT_SRC);
    this.juProj = gl.getUniformLocation(this.jointProg, "uProj");
    this.juView = gl.getUniformLocation(this.jointProg, "uView");
    this.jointVao = gl.createVertexArray();
    this.jointVbo = gl.createBuffer();
    gl.bindVertexArray(this.jointVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.jointVbo);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 32, 0);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 4, gl.FLOAT, false, 32, 12);
    gl.enableVertexAttribArray(2);
    gl.vertexAttribPointer(2, 1, gl.FLOAT, false, 32, 28);
    gl.bindVertexArray(null);

    this.boneShow = false;
    this.meshXray = false;
    this.boneEditMode = false;
    this.boneMirror = true;
    this.boneMoveChildren = true;
    this.boneHover = -1;
    this.draggingJoint = -1;
    this.facingYaw = 0;
    this.restJointPositions = [];
    this.editJointPositions = [];
    this.onBoneSelect = null;
    this.onBoneEditChange = null;
    this._lastViewProj = null;
    this._lastCamRight = [1, 0, 0];
    this._lastCamUp = [0, 0, 1];
    this._lastViewport = [1, 1];

    this.skinned = false;
    this.skHasTex = false;
    this._lastUvBuffer = null;
    this.boneCount = 0;
    this.boneNames = [];
    this.bindMatrices = [];      // rest WORLD matrix per joint (column-major Float32Array[16])
    this.localRestTRS = [];
    this.animations = [];
    this.activeAnim = -1;
    this.animTime = 0;
    this.animPlaying = false;
    this.animSpeed = 1.0;
    this.animLoop = true;
    this.onAnimTick = null;
    this.localRestM = [];        // rest LOCAL matrix per joint (column-major)
    this.jParentJoint = [];      // parent joint index per joint (-1 = root / non-joint parent)
    this.jParentWorldRest = [];  // rest world matrix of each joint's parent node (null for root)
    this.nodeWorldRest = null;   // Map nodeId -> rest world matrix
    this.localDelta = [];        // per-joint local TRS pose delta (column-major), identity = rest
    this.bonePose = null;        // legacy, unused
    this.boneSel = -1;
    this.tex = gl.createTexture();
    this.hasTex = false;

    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
    const stride = 14 * 4;
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, stride, 0);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 3, gl.FLOAT, false, stride, 12);
    gl.enableVertexAttribArray(2);
    gl.vertexAttribPointer(2, 3, gl.FLOAT, false, stride, 24);
    gl.enableVertexAttribArray(3);
    gl.vertexAttribPointer(3, 3, gl.FLOAT, false, stride, 36);
    gl.enableVertexAttribArray(4);
    gl.vertexAttribPointer(4, 2, gl.FLOAT, false, stride, 48);
    gl.bindVertexArray(null);

    this.drawVerts = 0;
    this.meshInfo = null;
    this.shading = "color"; // "color" | "clay"
    this.wireframe = false;
    this.autoRotate = true;
    this.active = false;

    // Z-up spherical orbit state
    this.target = [0, 0, 0];
    this.cur = { yaw: 0.45, pitch: 0.25, dist: 2.2, tx: 0, ty: 0, tz: 0 };
    this.goal = { ...this.cur };
    this.dragging = false;
    this.dragButton = 0;
    this.lastPtr = [0, 0];
    this.lastInteract = -Infinity;
    this.lastTime = 0;

    this._wireControls();
  }

  _projectToScreen(p3) {
    if (!this._lastViewProj) return null;
    const m = this._lastViewProj;
    const x = m[0] * p3[0] + m[4] * p3[1] + m[8] * p3[2] + m[12];
    const y = m[1] * p3[0] + m[5] * p3[1] + m[9] * p3[2] + m[13];
    const w = m[3] * p3[0] + m[7] * p3[1] + m[11] * p3[2] + m[15];
    if (w <= 1e-4) return null;
    const rect = this.canvas.getBoundingClientRect();
    return [
      rect.left + (x / w * 0.5 + 0.5) * rect.width,
      rect.top + (0.5 - y / w * 0.5) * rect.height,
      w,
    ];
  }

  _displayJointPositions() {
    if (this.boneEditMode && this.editJointPositions.length === this.boneCount) {
      return this.editJointPositions;
    }
    const bm = this._currentBoneMatrices();
    const pts = [];
    for (let j = 0; j < this.boneCount; j++) {
      pts.push([bm[j * 16 + 12], bm[j * 16 + 13], bm[j * 16 + 14]]);
    }
    return pts;
  }

  _pickJointAt(clientX, clientY, radiusPx = 18) {
    if (!this.skinned || !this.boneShow || !this.boneCount) return -1;
    const pts = this._displayJointPositions();
    let best = -1, bestD2 = radiusPx * radiusPx;
    for (let j = 0; j < pts.length; j++) {
      const sp = this._projectToScreen(pts[j]);
      if (!sp) continue;
      const dx = sp[0] - clientX, dy = sp[1] - clientY;
      const d2 = dx * dx + dy * dy;
      if (d2 < bestD2) { bestD2 = d2; best = j; }
    }
    return best;
  }

  _mirrorBoneIndex(j) {
    const name = this.boneNames[j] || "";
    let pair = null;
    if (name.startsWith("L")) pair = "R" + name.slice(1);
    else if (name.startsWith("R")) pair = "L" + name.slice(1);
    return pair ? this.boneNames.indexOf(pair) : -1;
  }

  _subtreeIndices(rootIdx) {
    const out = [rootIdx];
    const stack = [rootIdx];
    while (stack.length) {
      const p = stack.pop();
      for (let j = 0; j < this.boneCount; j++) {
        if (this.jParentJoint[j] === p) {
          out.push(j);
          stack.push(j);
        }
      }
    }
    return out;
  }

  _applyEditDelta(j, delta) {
    if (j < 0 || j >= this.editJointPositions.length) return;
    const targets = this.boneMoveChildren ? this._subtreeIndices(j) : [j];
    for (const ti of targets) {
      this.editJointPositions[ti] = [
        this.editJointPositions[ti][0] + delta[0],
        this.editJointPositions[ti][1] + delta[1],
        this.editJointPositions[ti][2] + delta[2],
      ];
    }
    if (this.boneMirror) {
      const mj = this._mirrorBoneIndex(j);
      if (mj >= 0 && mj !== j) {
        // Reflect delta across bilateral plane (normal = L_axis = [cos(yaw), sin(yaw), 0])
        const lx = Math.cos(this.facingYaw), ly = Math.sin(this.facingYaw);
        const dotL = delta[0] * lx + delta[1] * ly;
        const mDelta = [
          delta[0] - 2 * dotL * lx,
          delta[1] - 2 * dotL * ly,
          delta[2],
        ];
        const mTargets = this.boneMoveChildren ? this._subtreeIndices(mj) : [mj];
        for (const ti of mTargets) {
          this.editJointPositions[ti] = [
            this.editJointPositions[ti][0] + mDelta[0],
            this.editJointPositions[ti][1] + mDelta[1],
            this.editJointPositions[ti][2] + mDelta[2],
          ];
        }
      }
    }
    if (this.onBoneEditChange) {
      this.onBoneEditChange(this.boneNames[j], this.editJointPositions[j]);
    }
  }

  _wireControls() {
    const c = this.canvas;
    c.addEventListener("contextmenu", (e) => e.preventDefault());
    c.addEventListener("pointerdown", (e) => {
      if (!this.active) return;
      c.setPointerCapture(e.pointerId);
      this.lastPtr = [e.clientX, e.clientY];
      this.lastInteract = performance.now();

      if (e.button === 0 && this.skinned && this.boneShow) {
        const hit = this._pickJointAt(e.clientX, e.clientY, 18);
        if (hit >= 0) {
          this.boneSel = hit;
          if (this.onBoneSelect) this.onBoneSelect(this.boneNames[hit]);
          if (this.boneEditMode) {
            this.draggingJoint = hit;
            c.style.cursor = "grabbing";
            return;
          }
        }
      }
      this.dragging = true;
      this.dragButton = e.button;
    });
    c.addEventListener("pointermove", (e) => {
      if (!this.active) return;
      const dx = e.clientX - this.lastPtr[0];
      const dy = e.clientY - this.lastPtr[1];
      this.lastPtr = [e.clientX, e.clientY];

      if (this.draggingJoint >= 0 && this.boneEditMode) {
        this.lastInteract = performance.now();
        // Project pixel drag to world movement in camera view plane at joint depth
        const rect = c.getBoundingClientRect();
        const jp = this.editJointPositions[this.draggingJoint];
        const sp = this._projectToScreen(jp);
        const depth = sp ? sp[2] : this.cur.dist;
        const worldPerPx = (2 * depth * Math.tan((20 * Math.PI) / 180)) / Math.max(1, rect.height);
        const r = this._lastCamRight;
        const u = this._lastCamUp;
        const delta = [
          (r[0] * dx - u[0] * dy) * worldPerPx,
          (r[1] * dx - u[1] * dy) * worldPerPx,
          (r[2] * dx - u[2] * dy) * worldPerPx,
        ];
        this._applyEditDelta(this.draggingJoint, delta);
        return;
      }

      if (!this.dragging) {
        if (this.skinned && this.boneShow) {
          const hit = this._pickJointAt(e.clientX, e.clientY, 18);
          this.boneHover = hit;
          c.style.cursor = hit >= 0 ? (this.boneEditMode ? "grab" : "pointer") : "";
        } else {
          this.boneHover = -1;
          c.style.cursor = "";
        }
        return;
      }
      this.lastInteract = performance.now();

      if (this.dragButton === 2 || e.shiftKey) {
        const panScale = this.goal.dist * 0.0016;
        const cy = Math.cos(this.goal.yaw), sy = Math.sin(this.goal.yaw);
        const right = [cy, sy, 0];
        const cp = Math.cos(this.goal.pitch), sp = Math.sin(this.goal.pitch);
        const up = [-sp * sy, sp * cy, cp];
        this.goal.tx -= (right[0] * dx - up[0] * dy) * panScale;
        this.goal.ty -= (right[1] * dx - up[1] * dy) * panScale;
        this.goal.tz -= (right[2] * dx - up[2] * dy) * panScale;
      } else {
        this.goal.yaw -= dx * 0.0085;
        this.goal.pitch = Math.max(-1.48, Math.min(1.48, this.goal.pitch + dy * 0.0085));
      }
    });
    const endDrag = () => {
      if (this.draggingJoint >= 0) {
        this.draggingJoint = -1;
        this.lastInteract = performance.now();
        c.style.cursor = this.boneHover >= 0 && this.boneEditMode ? "grab" : "";
      }
      if (this.dragging) {
        this.dragging = false;
        this.lastInteract = performance.now();
      }
    };
    c.addEventListener("pointerup", endDrag);
    c.addEventListener("pointercancel", endDrag);
    c.addEventListener("wheel", (e) => {
      if (!this.active) return;
      e.preventDefault();
      const factor = Math.exp(e.deltaY * 0.0011);
      this.goal.dist = Math.max(0.25, Math.min(20.0, this.goal.dist * factor));
      this.lastInteract = performance.now();
    }, { passive: false });
  }

  _uploadTexture(textureImg) {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, textureImg);
    gl.generateMipmap(gl.TEXTURE_2D);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    const aniso = gl.getExtension("EXT_texture_filter_anisotropic") ||
                  gl.getExtension("WEBKIT_EXT_texture_filter_anisotropic");
    if (aniso) {
      const maxA = Math.min(16, gl.getParameter(aniso.MAX_TEXTURE_MAX_ANISOTROPY_EXT) || 1);
      gl.texParameterf(gl.TEXTURE_2D, aniso.TEXTURE_MAX_ANISOTROPY_EXT, maxA);
    }
    this.hasTex = true;
  }

  setBoneShow(on) {
    this.boneShow = !!on;
  }

  loadFromBuffer(buf, uvBuffer = null, textureImg = null) {
    const parsed = parseMeshPly(buf, uvBuffer);
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
    gl.bufferData(gl.ARRAY_BUFFER, parsed.interleaved, gl.STATIC_DRAW);
    this.drawVerts = parsed.drawVerts;
    this._lastUvBuffer = uvBuffer || null;

    if (parsed.hasUV && textureImg) {
      this._uploadTexture(textureImg);
    } else {
      this.hasTex = false;
    }

    this.meshInfo = {
      vertexCount: parsed.vertexCount,
      faceCount: parsed.faceCount,
      hasTexture: this.hasTex,
      bounds: parsed.bounds,
    };
    this.recenter();
    return this.meshInfo;
  }

  _composeWorldMatrices(json) {
    // Returns, indexed by joint order (skin.joints):
    //   bindMatrices[j]  : rest WORLD matrix of joint j (column-major)
    //   localRestM[j]    : rest LOCAL matrix of joint j relative to its PARENT node (column-major)
    //   jParentNode[j]   : node id of joint j's parent node, or -1 if root
    //   nodeWorldRest    : Map nodeId -> rest WORLD matrix (for any node, incl. non-joints)
    const nodeAt = {};
    for (let i = 0; i < json.nodes.length; i++) nodeAt[i] = json.nodes[i];

    // local matrix (relative to parent) for a node
    const localOf = (n) => {
      if (n.matrix) {
        const m = new Float32Array(16);
        for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) m[r * 4 + c] = n.matrix[c * 4 + r];
        return m;
      }
      const t = n.translation || [0, 0, 0];
      const q = n.rotation || [0, 0, 0, 1];
      const s = n.scale || [1, 1, 1];
      const out = new Float32Array(16);
      mat4ComposeTRS(t, q, s, out);
      return out;
    };

    const parentOf = (id) => {
      for (let i = 0; i < json.nodes.length; i++) {
        const ch = json.nodes[i].children;
        if (ch && ch.includes(id)) return i;
      }
      return -1;
    };

    const world = new Map(); // nodeId -> rest world matrix
    const compute = (id) => {
      if (world.has(id)) return world.get(id);
      const n = nodeAt[id] || {};
      const m = localOf(n);
      const parent = parentOf(id);
      const w = parent >= 0 ? mat4Mul(compute(parent), m) : m;
      world.set(id, w);
      return w;
    };

    const skinIdx = json.skins[0];
    const joints = skinIdx ? json.skins[0].joints : [];
    // joint index lookup by node id
    const jointIndexOfNode = new Map();
    for (let j = 0; j < joints.length; j++) jointIndexOfNode.set(joints[j], j);

    const bindMatrices = joints.map((id) => compute(id));
    const localRestM = joints.map((id) => localOf(nodeAt[id] || {}));
    const jParentNode = joints.map((id) => parentOf(id)); // node id of parent, or -1
    return { bindMatrices, localRestM, jParentNode, nodeWorldRest: world, jointIndexOfNode };
  }

  loadSkinnedGlb(buf, uvBuffer = null, textureImg = null) {
    const gl = this.gl;
    const p = parseSkinnedGlb(buf);
    if (p.jointsNodeIds.length > 20) throw new Error("too many joints (>20)");
    this.skinned = true;

    if (uvBuffer) this._lastUvBuffer = uvBuffer;
    if (textureImg) this._uploadTexture(textureImg);

    const uvSrc = uvBuffer || this._lastUvBuffer;
    const uvArr = uvSrc ? new Float32Array(uvSrc) : null;
    const hasSkUV = !!(uvArr && uvArr.length === p.indices.length * 2);
    this.skHasTex = !!(hasSkUV && this.hasTex);

    // interleaved (unwelded per triangle corner for barycentric wireframe):
    // pos(3) normal(3) color(3) joints(4) weights(4) bary(3) uv(2) = 22 floats
    const n = p.pos.length / 3;
    const stride = 22 * 4;
    const data = new Float32Array(p.indices.length * 22);
    const BARY = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    for (let f = 0; f < p.indices.length / 3; f++) {
      for (let k = 0; k < 3; k++) {
        const cornerIdx = f * 3 + k;
        const vi = p.indices[cornerIdx];
        const o = cornerIdx * 22;
        data[o] = p.pos[vi * 3]; data[o + 1] = p.pos[vi * 3 + 1]; data[o + 2] = p.pos[vi * 3 + 2];
        if (p.normal) {
          data[o + 3] = p.normal[vi * 3]; data[o + 4] = p.normal[vi * 3 + 1]; data[o + 5] = p.normal[vi * 3 + 2];
        } else { data[o + 5] = 1; }
        if (p.color) {
          data[o + 6] = p.color[vi * 3]; data[o + 7] = p.color[vi * 3 + 1]; data[o + 8] = p.color[vi * 3 + 2];
        } else { data[o + 6] = data[o + 7] = data[o + 8] = 1; }
        if (p.joints) {
          for (let b = 0; b < 4; b++) data[o + 9 + b] = p.joints[vi * 4 + b];
        }
        if (p.weights) {
          for (let b = 0; b < 4; b++) data[o + 13 + b] = p.weights[vi * 4 + b];
        } else { data[o + 13] = 1; }
        data[o + 17] = BARY[k][0]; data[o + 18] = BARY[k][1]; data[o + 19] = BARY[k][2];
        data[o + 20] = hasSkUV ? uvArr[cornerIdx * 2] : 0;
        data[o + 21] = hasSkUV ? uvArr[cornerIdx * 2 + 1] : 0;
      }
    }

    gl.bindVertexArray(this.skVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.skVbo);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    for (const [loc, size, off] of [[0, 3, 0], [1, 3, 12], [2, 3, 24], [3, 4, 36], [4, 4, 52], [5, 3, 68], [6, 2, 80]]) {
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, size, gl.FLOAT, false, stride, off);
    }
    gl.bindVertexArray(null);
    this.skDrawCount = p.indices.length;

    const { bindMatrices, localRestM, jParentNode, nodeWorldRest, jointIndexOfNode } = this._composeWorldMatrices(p.json);
    this.bindMatrices = bindMatrices;
    this.localRestM = localRestM;
    this.nodeWorldRest = nodeWorldRest;
    this.jointIndexOfNode = jointIndexOfNode;
    // parent joint index per joint (-1 = root / parent is not a deform joint)
    this.jParentJoint = jParentNode.map((pid) => (pid >= 0 && jointIndexOfNode.has(pid) ? jointIndexOfNode.get(pid) : -1));
    // rest WORLD matrix of the parent node for each joint (null for root) — used as the
    // base of the subtree when the parent node itself is not a deform joint
    this.jParentWorldRest = jParentNode.map((pid) => (pid >= 0 ? (nodeWorldRest.get(pid) || null) : null));
    this.boneCount = bindMatrices.length;
    // per-joint LOCAL pose deltas (TRS applied in the joint's own rest frame), default identity
    this.localDelta = bindMatrices.map(() => mat4Identity());
    // IBM: use the GLB-stored inverseBindingMatrices directly. They are
    // column-major (glTF spec) — the same layout the skinned shader feeds to
    // uIBMs — and are the exact inverse of the bindWorld matrices above
    // (verified: bindWorld[j] * storedIBM[j] ~= I to ~1e-7). Using them avoids
    // any float32 mat4Invert rounding and guarantees the rest pose renders the
    // mesh unskinned (shader: M = sum_j w_j * world_j * ibm_j -> identity).
    if (p.ibm && p.ibm.length === this.boneCount * 16) {
      this.ibm = new Float32Array(p.ibm);
    } else {
      // Fallback if the GLB omits/mismatched inverseBindingMatrices.
      this.ibm = new Float32Array(this.boneCount * 16);
      for (let j = 0; j < this.boneCount; j++) {
        this.ibm.set(mat4Invert(bindMatrices[j]), j * 16);
      }
    }
    this.boneNames = (p.boneNames && p.boneNames.length) ? p.boneNames : p.jointsNodeIds.map((id) => p.json.nodes[id].name || "bone");
    this.localRestTRS = p.jointsNodeIds.map((id) => {
      const nd = p.json.nodes[id] || {};
      return {
        t: nd.translation ? [nd.translation[0], nd.translation[1], nd.translation[2]] : [0, 0, 0],
        r: nd.rotation ? [nd.rotation[0], nd.rotation[1], nd.rotation[2], nd.rotation[3]] : [0, 0, 0, 1],
        s: nd.scale ? [nd.scale[0], nd.scale[1], nd.scale[2]] : [1, 1, 1],
      };
    });
    this.animations = p.animations || [];
    this.activeAnim = this.animations.length > 0 ? 0 : -1;
    this.animTime = 0;
    this.animPlaying = this.animations.length > 0;
    this.bonePose = null;
    this.boneSel = this.boneCount > 0 ? 0 : -1;
    this.boneHover = -1;
    this.draggingJoint = -1;
    this.facingYaw = Number((p.json.asset && p.json.asset.extras && p.json.asset.extras.facing_yaw) || 0);
    this.restJointPositions = bindMatrices.map((m) => [m[12], m[13], m[14]]);
    this.editJointPositions = this.restJointPositions.map((pt) => [pt[0], pt[1], pt[2]]);

    this.meshInfo = {
      vertexCount: n,
      faceCount: p.indices.length / 3,
      hasTexture: this.skHasTex,
      bounds: p.bounds,
      skinned: true,
      boneCount: this.boneCount,
      boneNames: this.boneNames,
      animations: this.listAnimations(),
    };
    this.recenter();
    return this.meshInfo;
  }

  listBones() {
    if (!this.skinned) return [];
    return this.boneNames.map((name, i) => ({
      name,
      index: i,
      parent: this.jParentJoint[i],
      parentName: this.jParentJoint[i] >= 0 ? this.boneNames[this.jParentJoint[i]] : null,
    }));
  }

  // Set the local TRS delta of a bone, expressed in its PARENT's (posed) frame —
  // the same intuition Blender's bone-local gizmos give. t/eDeg/s are optional;
  // passing null/omitting a part keeps identity for that part.
  setBoneTRS(name, { t, eDeg, s } = {}) {
    const i = this.boneNames.indexOf(name);
    if (i < 0 || !this.skinned) return false;
    const tr = t ? [t[0], t[1], t[2]] : [0, 0, 0];
    const er = eDeg ? [eDeg[0], eDeg[1], eDeg[2]] : [0, 0, 0];
    const sc = s ? [s[0], s[1], s[2]] : [1, 1, 1];
    const hasAny = (tr[0] !== 0 || tr[1] !== 0 || tr[2] !== 0) ||
                   (er[0] !== 0 || er[1] !== 0 || er[2] !== 0) ||
                   (sc[0] !== 1 || sc[1] !== 1 || sc[2] !== 1);
    this.localDelta[i] = hasAny ? mat4FromEulerDeg(tr, er, sc) : mat4Identity();
    return true;
  }

  // Convenience for UI: read back the current TRS delta for a bone (all in
  // parent-posed frame, degrees). Returns {t, eDeg, s} or null if not skinned.
  getBoneTRS(name) {
    const i = this.boneNames.indexOf(name);
    if (i < 0 || !this.skinned) return null;
    const m = this.localDelta[i];
    return {
      t: [m[12], m[13], m[14]],
      eDeg: [0, 0, 0], // rotation not decomposed back; UI owns its own Euler state
      s: [1, 1, 1],
    };
  }

  resetBonePose(name) {
    if (!this.skinned) return;
    if (name == null) {
      this.localDelta = this.localDelta.map(() => mat4Identity());
    } else {
      const i = this.boneNames.indexOf(name);
      if (i >= 0) this.localDelta[i] = mat4Identity();
    }
  }

  // Recompute posed-world matrices for every joint:
  //   W_j = W_parent(j) * D_j
  // where W_parent(j) is the posed world of j's parent joint when the parent is
  // itself a deform joint, the rest world of the parent node otherwise, or the
  // identity for a root. Children are then computed topologically (parent first),
  // so a parent's delta propagates to its whole subtree — this is what makes the
  // skinned mesh follow when you drag a single bone.
  listAnimations() {
    return this.animations.map((a, index) => ({
      index,
      name: a.name,
      duration: a.duration,
    }));
  }

  setAnimation(idx) {
    if (idx < 0 || idx >= this.animations.length) {
      this.activeAnim = -1;
      this.animPlaying = false;
      this.animTime = 0;
      return;
    }
    this.activeAnim = idx;
    this.animTime = 0;
    this.animPlaying = true;
  }

  playAnimation() {
    if (this.activeAnim < 0 && this.animations.length > 0) this.activeAnim = 0;
    if (this.activeAnim >= 0) this.animPlaying = true;
  }

  pauseAnimation() {
    this.animPlaying = false;
  }

  seekAnimation(sec) {
    const a = this.animations[this.activeAnim];
    if (!a) return;
    this.animTime = Math.max(0, Math.min(a.duration || 0, sec));
  }

  setAnimSpeed(speed) {
    this.animSpeed = Math.max(0.1, Math.min(4.0, Number(speed) || 1.0));
  }

  setAnimLoop(loop) {
    this.animLoop = !!loop;
  }

  _sampleTrack(track, t, dim) {
    const times = track.times;
    const vals = track.values;
    const count = times.length;
    if (count === 0) return null;
    if (count === 1 || t <= times[0]) {
      return vals.subarray(0, dim);
    }
    const last = count - 1;
    if (t >= times[last]) {
      return vals.subarray(last * dim, last * dim + dim);
    }
    let lo = 0, hi = last;
    while (lo + 1 < hi) {
      const mid = (lo + hi) >> 1;
      if (times[mid] <= t) lo = mid;
      else hi = mid;
    }
    if (track.interp === "STEP") {
      return vals.subarray(lo * dim, lo * dim + dim);
    }
    const t0 = times[lo], t1 = times[hi];
    const alpha = t1 > t0 ? (t - t0) / (t1 - t0) : 0;
    const aOff = lo * dim, bOff = hi * dim;
    if (dim === 4) {
      // Normalized quaternion SLERP
      let ax = vals[aOff], ay = vals[aOff + 1], az = vals[aOff + 2], aw = vals[aOff + 3];
      let bx = vals[bOff], by = vals[bOff + 1], bz = vals[bOff + 2], bw = vals[bOff + 3];
      let dot = ax * bx + ay * by + az * bz + aw * bw;
      if (dot < 0) { bx = -bx; by = -by; bz = -bz; bw = -bw; dot = -dot; }
      let s0, s1;
      if (dot > 0.9995) {
        s0 = 1 - alpha;
        s1 = alpha;
      } else {
        const th = Math.acos(Math.min(1, dot));
        const sinTh = Math.sin(th);
        s0 = Math.sin((1 - alpha) * th) / sinTh;
        s1 = Math.sin(alpha * th) / sinTh;
      }
      const rx = s0 * ax + s1 * bx;
      const ry = s0 * ay + s1 * by;
      const rz = s0 * az + s1 * bz;
      const rw = s0 * aw + s1 * bw;
      const len = Math.hypot(rx, ry, rz, rw) || 1;
      return [rx / len, ry / len, rz / len, rw / len];
    }
    const out = new Array(dim);
    for (let k = 0; k < dim; k++) {
      out[k] = vals[aOff + k] + (vals[bOff + k] - vals[aOff + k]) * alpha;
    }
    return out;
  }

  _currentBoneMatrices() {
    const out = new Float32Array(this.boneCount * 16);
    const posed = new Array(this.boneCount).fill(null);
    const parentFirst = [];
    {
      const seen = new Set();
      const visit = (j) => {
        if (seen.has(j)) return;
        seen.add(j);
        const p = this.jParentJoint[j];
        if (p >= 0) visit(p);
        parentFirst.push(j);
      };
      for (let j = 0; j < this.boneCount; j++) visit(j);
    }
    const anim = this.activeAnim >= 0 ? this.animations[this.activeAnim] : null;
    for (const j of parentFirst) {
      const d = this.localDelta[j];
      let localM = this.localRestM[j];
      if (anim && anim.tracks.has(j)) {
        const tr = anim.tracks.get(j);
        const rest = this.localRestTRS[j] || { t: [0, 0, 0], r: [0, 0, 0, 1], s: [1, 1, 1] };
        const t = tr.translation ? this._sampleTrack(tr.translation, this.animTime, 3) : rest.t;
        const r = tr.rotation ? this._sampleTrack(tr.rotation, this.animTime, 4) : rest.r;
        const s = tr.scale ? this._sampleTrack(tr.scale, this.animTime, 3) : rest.s;
        localM = new Float32Array(16);
        mat4ComposeTRS(t, r, s, localM);
      }
      const p = this.jParentJoint[j];
      let parentW;
      if (p >= 0) parentW = posed[p];
      else if (this.jParentWorldRest && this.jParentWorldRest[j]) parentW = this.jParentWorldRest[j];
      else parentW = mat4Identity();
      const jointLocal = (d === null) ? localM : mat4Mul(localM, d);
      const w = mat4Mul(parentW, jointLocal);
      posed[j] = w;
      out.set(w, j * 16);
    }
    return out;
  }

  clear() {
    this.drawVerts = 0;
    this.hasTex = false;
    this.skHasTex = false;
    this._lastUvBuffer = null;
    this.meshInfo = null;
    this.skinned = false;
    this.boneCount = 0;
    this.boneNames = [];
    this.bindMatrices = [];
    this.localRestM = [];
    this.localRestTRS = [];
    this.localDelta = [];
    this.animations = [];
    this.activeAnim = -1;
    this.animTime = 0;
    this.animPlaying = false;
    this.jParentJoint = [];
    this.jParentWorldRest = [];
    this.nodeWorldRest = null;
    this.ibm = null;
    this.bonePose = null;
    this.boneSel = -1;
  }

  recenter() {
    if (!this.meshInfo) return;
    const { min, max } = this.meshInfo.bounds;
    const cx = (min[0] + max[0]) * 0.5;
    const cy = (min[1] + max[1]) * 0.5;
    const cz = (min[2] + max[2]) * 0.5;
    const radius = 0.5 * Math.max(max[0] - min[0], max[1] - min[1], max[2] - min[2]) || 0.5;
    const dist = Math.max(0.8, (radius / Math.tan((40 * Math.PI) / 360)) * 1.48);
    this.goal = { yaw: 0.48, pitch: 0.28, dist, tx: cx, ty: cy, tz: cz };
    this.cur = { ...this.goal };
    this.lastInteract = -Infinity;
  }

  setStyle({ shading, wireframe } = {}) {
    if (shading) this.shading = shading;
    if (typeof wireframe === "boolean") this.wireframe = wireframe;
  }

  setMeshXray(on) {
    this.meshXray = !!on;
  }

  setBoneEditMode(on) {
    this.boneEditMode = !!on;
    if (this.boneEditMode) {
      this.boneShow = true;
      this.activeAnim = -1;
      this.animPlaying = false;
      this.resetBonePose(null);
      if (this.boneSel < 0 && this.boneCount > 0) this.boneSel = 0;
    } else {
      this.draggingJoint = -1;
    }
  }

  setBoneMirror(on) {
    this.boneMirror = !!on;
  }

  setBoneMoveChildren(on) {
    this.boneMoveChildren = !!on;
  }

  selectBone(name) {
    const idx = this.boneNames.indexOf(name);
    if (idx >= 0) this.boneSel = idx;
  }

  getEditJointPos(name) {
    const idx = this.boneNames.indexOf(name);
    if (idx < 0 || idx >= this.editJointPositions.length) return null;
    const p = this.editJointPositions[idx];
    return [p[0], p[1], p[2]];
  }

  setEditJointPos(name, pos) {
    const idx = this.boneNames.indexOf(name);
    if (idx < 0 || idx >= this.editJointPositions.length) return;
    const cur = this.editJointPositions[idx];
    const delta = [pos[0] - cur[0], pos[1] - cur[1], pos[2] - cur[2]];
    this._applyEditDelta(idx, delta);
  }

  resetEditJoints() {
    this.editJointPositions = this.restJointPositions.map((pt) => [pt[0], pt[1], pt[2]]);
    if (this.boneSel >= 0 && this.onBoneEditChange) {
      this.onBoneEditChange(this.boneNames[this.boneSel], this.editJointPositions[this.boneSel]);
    }
  }

  getAllEditedJoints() {
    const out = {};
    for (let j = 0; j < this.boneCount; j++) {
      const p = this.editJointPositions[j] || this.restJointPositions[j];
      if (p) out[this.boneNames[j]] = [Number(p[0].toFixed(5)), Number(p[1].toFixed(5)), Number(p[2].toFixed(5))];
    }
    return out;
  }

  setAutoRotate(on) {
    this.autoRotate = !!on;
    if (this.autoRotate) this.lastInteract = -Infinity;
  }

  render(now) {
    if (!this.active) return;
    const dt = this.lastTime ? Math.min(0.1, (now - this.lastTime) / 1000) : 0;
    this.lastTime = now;

    const gl = this.gl;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.floor(this.canvas.clientWidth * dpr));
    const h = Math.max(1, Math.floor(this.canvas.clientHeight * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }

    gl.viewport(0, 0, w, h);
    gl.clearColor(6 / 255, 6 / 255, 8 / 255, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    if (!this.drawVerts && !this.skinned) return;

    if (this.skinned && this.animPlaying && this.activeAnim >= 0) {
      const anim = this.animations[this.activeAnim];
      if (anim && anim.duration > 0) {
        this.animTime += dt * this.animSpeed;
        if (this.animTime >= anim.duration) {
          if (this.animLoop) {
            this.animTime = this.animTime % anim.duration;
          } else {
            this.animTime = anim.duration;
            this.animPlaying = false;
          }
        }
        if (this.onAnimTick) this.onAnimTick(this.animTime, anim.duration, this.animPlaying);
      }
    }

    if (this.autoRotate && !this.dragging && now - this.lastInteract > 2200) {
      this.goal.yaw += dt * 0.42;
    }

    const alpha = dt > 0 ? 1 - Math.exp(-dt / 0.14) : 1;
    for (const k of ["yaw", "pitch", "dist", "tx", "ty", "tz"]) {
      this.cur[k] += (this.goal[k] - this.cur[k]) * alpha;
    }

    const cp = Math.cos(this.cur.pitch), sp = Math.sin(this.cur.pitch);
    const cy = Math.cos(this.cur.yaw), sy = Math.sin(this.cur.yaw);
    const eye = [
      this.cur.tx + this.cur.dist * cp * sy,
      this.cur.ty - this.cur.dist * cp * cy,
      this.cur.tz + this.cur.dist * sp,
    ];
    const target = [this.cur.tx, this.cur.ty, this.cur.tz];

    const proj = mat4Perspective((40 * Math.PI) / 180, w / h, 0.03, 100.0);
    const view = mat4LookAt(eye, target, [0, 0, 1]);
    this._lastViewProj = mat4Mul(proj, view);
    this._lastCamRight = [view[0], view[4], view[8]];
    this._lastCamUp = [view[1], view[5], view[9]];
    this._lastViewport = [w, h];

    gl.enable(gl.DEPTH_TEST);

    if (this.skinned && this.skDrawCount) {
      const xray = this.meshXray && this.boneShow;
      if (xray) {
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      } else {
        gl.disable(gl.BLEND);
      }
      gl.useProgram(this.skProg);
      gl.uniformMatrix4fv(this.suProj, false, proj);
      gl.uniformMatrix4fv(this.suView, false, view);
      gl.uniform1i(this.suMode, this.shading === "clay" ? 1 : this.shading === "unlit" ? 2 : 0);
      gl.uniform1i(this.suWire, this.wireframe ? 1 : 0);
      gl.uniform1i(this.suHasTex, this.skHasTex ? 1 : 0);
      gl.uniform1f(this.suAlpha, xray ? 0.42 : 1.0);
      if (this.skHasTex) {
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.tex);
        gl.uniform1i(this.suTex, 0);
      }
      gl.uniform3f(this.suWireCol, 1.0, 0.54, 0.12);
      if (this.boneCount) {
        gl.uniformMatrix4fv(this.suBones, false, this._currentBoneMatrices());
        gl.uniformMatrix4fv(this.suIBMs, false, this.ibm);
      }
      gl.bindVertexArray(this.skVao);
      gl.drawArrays(gl.TRIANGLES, 0, this.skDrawCount);
      gl.bindVertexArray(null);

      if (this.boneShow && this.boneCount > 0) {
        this._renderSkeletonXray(proj, view, dpr);
      }
      gl.disable(gl.BLEND);
      return;
    }

    gl.disable(gl.BLEND);
    gl.useProgram(this.prog);
    gl.uniformMatrix4fv(this.uProj, false, proj);
    gl.uniformMatrix4fv(this.uView, false, view);
    gl.uniform1i(this.uMode, this.shading === "clay" ? 1 : this.shading === "unlit" ? 2 : 0);
    gl.uniform1i(this.uWire, this.wireframe ? 1 : 0);
    gl.uniform1i(this.uHasTex, this.hasTex ? 1 : 0);
    gl.uniform1i(this.uTex, 0);
    gl.uniform1f(this.uAlpha, 1.0);
    gl.uniform3f(this.uWireCol, 1.0, 0.54, 0.12);

    if (this.hasTex) {
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.tex);
    }

    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLES, 0, this.drawVerts);
    gl.bindVertexArray(null);
  }

  _boneColor(j, alpha = 1.0) {
    if (j === this.boneSel) return [0.72, 1.0, 0.24, alpha];       // bright lime = selected
    if (j === this.boneHover) return [1.0, 1.0, 1.0, alpha];       // white = hovered
    const name = this.boneNames[j] || "";
    if (name.startsWith("L")) return [0.22, 0.86, 1.0, alpha];     // cyan = Left
    if (name.startsWith("R")) return [1.0, 0.42, 0.62, alpha];     // coral pink = Right
    return [1.0, 0.76, 0.26, alpha];                               // warm gold = Center spine/head
  }

  _renderSkeletonXray(proj, view, dpr) {
    const gl = this.gl;
    const pts = this._displayJointPositions();
    if (!pts.length) return;

    // Build Blender-style 3D octahedral bone prisms (triangles + crisp edge lines)
    const triData = [];
    const lineData = [];
    const pushV = (arr, p, c) => arr.push(p[0], p[1], p[2], c[0], c[1], c[2], c[3]);

    const addOctaBone = (a, b, colFill, colEdge) => {
      const dx = b[0] - a[0], dy = b[1] - a[1], dz = b[2] - a[2];
      const len = Math.hypot(dx, dy, dz);
      if (len < 1e-4) return;
      const ux = dx / len, uy = dy / len, uz = dz / len;
      // Find two perpendicular unit vectors (v1, v2)
      let rx = 0, ry = 0, rz = 1;
      if (Math.abs(uz) > 0.9) { rx = 1; ry = 0; rz = 0; }
      let v1x = uy * rz - uz * ry, v1y = uz * rx - ux * rz, v1z = ux * ry - uy * rx;
      const l1 = Math.hypot(v1x, v1y, v1z) || 1;
      v1x /= l1; v1y /= l1; v1z /= l1;
      const v2x = uy * v1z - uz * v1y, v2y = uz * v1x - ux * v1z, v2z = ux * v1y - uy * v1x;

      const rad = Math.min(0.026, Math.max(0.008, len * 0.14));
      const mx = a[0] + dx * 0.22, my = a[1] + dy * 0.22, mz = a[2] + dz * 0.22;
      const ring = [
        [mx + v1x * rad, my + v1y * rad, mz + v1z * rad],
        [mx + v2x * rad, my + v2y * rad, mz + v2z * rad],
        [mx - v1x * rad, my - v1y * rad, mz - v1z * rad],
        [mx - v2x * rad, my - v2y * rad, mz - v2z * rad],
      ];
      const shade = [1.0, 0.85, 0.72, 0.90];
      for (let k = 0; k < 4; k++) {
        const p0 = ring[k], p1 = ring[(k + 1) % 4];
        const s = shade[k];
        const cf = [colFill[0] * s, colFill[1] * s, colFill[2] * s, colFill[3]];
        // Base pyramid (a -> ring)
        pushV(triData, a, cf); pushV(triData, p0, cf); pushV(triData, p1, cf);
        // Tip pyramid (b -> ring)
        pushV(triData, b, cf); pushV(triData, p1, cf); pushV(triData, p0, cf);
        // Crisp edges
        pushV(lineData, a, colEdge); pushV(lineData, p0, colEdge);
        pushV(lineData, p0, colEdge); pushV(lineData, b, colEdge);
        pushV(lineData, p0, colEdge); pushV(lineData, p1, colEdge);
      }
    };

    for (let j = 0; j < this.boneCount; j++) {
      const p = this.jParentJoint[j];
      if (p < 0) continue;
      // Highlight bone if either endpoint is selected/hovered
      const colIdx = (j === this.boneSel || p === this.boneSel) ? this.boneSel : j;
      const cFill = this._boneColor(colIdx, 0.58);
      const cEdge = this._boneColor(colIdx, 0.96);
      addOctaBone(pts[p], pts[j], cFill, cEdge);
    }

    // Joint circular markers
    const jointData = [];
    for (let j = 0; j < this.boneCount; j++) {
      const p = pts[j];
      const isSel = j === this.boneSel;
      const isHov = j === this.boneHover;
      const c = this._boneColor(j, 0.98);
      const sz = (isSel ? 17.0 : isHov ? 15.0 : (this.boneEditMode ? 12.5 : 10.5)) * dpr;
      jointData.push(p[0], p[1], p[2], c[0], c[1], c[2], c[3], sz);
    }

    // X-Ray pass: draw bones & joints OVER the mesh so they are 100% visible
    gl.disable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    gl.useProgram(this.boneProg);
    gl.uniformMatrix4fv(this.buProj, false, proj);
    gl.uniformMatrix4fv(this.buView, false, view);
    gl.bindVertexArray(this.boneVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.boneVbo);
    if (triData.length) {
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(triData), gl.DYNAMIC_DRAW);
      gl.drawArrays(gl.TRIANGLES, 0, triData.length / 7);
    }
    if (lineData.length) {
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(lineData), gl.DYNAMIC_DRAW);
      gl.drawArrays(gl.LINES, 0, lineData.length / 7);
    }
    gl.bindVertexArray(null);

    if (jointData.length) {
      gl.useProgram(this.jointProg);
      gl.uniformMatrix4fv(this.juProj, false, proj);
      gl.uniformMatrix4fv(this.juView, false, view);
      gl.bindVertexArray(this.jointVao);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.jointVbo);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(jointData), gl.DYNAMIC_DRAW);
      gl.drawArrays(gl.POINTS, 0, jointData.length / 8);
      gl.bindVertexArray(null);
    }

    gl.enable(gl.DEPTH_TEST);
  }
}
