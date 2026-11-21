# dcd-img2rig

이미지 한 장으로 게임용 리깅 FBX까지 한 번에 — 디지털콘텐츠디자인학과(DCD) 학생들의 게임 제작을 위한 3D 에셋 생성 & 자동 리깅 파이프라인입니다.

단일 캐릭터/오브젝트 이미지를 입력하면 **3D Gaussian Splatting(3DGS) 생성 → 삼각 메시 & 텍스처 베이크 → Binary FBX 7.4 내보내기 → 19본 휴머노이드 자동 리깅 → Mixamo 모션 리타게팅**까지 하나의 웹 스튜디오와 CLI로 처리할 수 있습니다.

---

## 주요 기능 & 워크플로우

```text
[1. 이미지 업로드]
        │  Idol (DINOv3 ViT-H+ + FLUX.2 VAE + DiT)
        ▼
[2. 3D Gaussian Splat (.ply)]
        │  Marching Cubes / Poisson / Surface Fit + UV 텍스처 베이크
        ▼
[3. 정제 삼각 메시 & Binary FBX 7.4 (.fbx / _mesh.ply / _texture.png)]
        │  해부학적 19본 피팅 + Blender ARMATURE_AUTO 스키닝
        ▼
[4. 스키닝 캐릭터 (_skinned.glb / _skinned.fbx)]
        │  Mixamo / FBX / GLB / BVH 모션 리타게팅
        ▼
[5. 애니메이션 적용 캐릭터 (_anim_<name>.glb)]  ──▶ Unity / Unreal / Blender
```

- **웹 스튜디오 (`WebGL2`)**: 3DGS 스플랫 뷰어(`gsplat.js`)와 커스텀 WebGL2 메시·스키닝·X-Ray 본 뷰어를 브라우저에서 바로 사용할 수 있습니다.
- **게임 엔진 호환 Binary FBX 7.4**: Blender가 설치되어 있으면 Blender로 UV 전개 및 라운드트립 검증을 수행하며, 없을 경우 내장 순수 Python Binary FBX 7.4 라이터로 자동 폴백합니다.
- **크로스 플랫폼 지원**: Windows (NVIDIA CUDA / CPU), macOS (Apple Silicon MPS / CPU), Linux (CUDA / CPU)를 모두 지원합니다.

---

## 요구 사항

| 항목 | 권장 사양 / 설명 |
|---|---|
| **OS** | Windows 10/11 (64-bit), macOS (Apple Silicon), Linux |
| **Python** | `3.10` ~ `3.12` |
| **GPU / 가속기** | 이미지→3DGS 생성 시 NVIDIA CUDA GPU(VRAM 8GB+ 권장) 또는 Apple Silicon(MPS). 메시 변환·프리뷰·스키닝은 CPU에서도 동작 |
| **Blender** *(권장)* | `Blender 4.x` 이상. UV 텍스처 베이크, FBX 라운드트립 검증, `ARMATURE_AUTO` 스킨 웨이트, 모션 리타게팅에 사용 |
| **브라우저** | **WebGL2** 지원 최신 브라우저 (Chrome, Edge, Safari, Firefox) |

---

## 설치 방법

### 1. 저장소 클론 및 가상환경 생성

**macOS / Linux:**
```bash
git clone https://github.com/sevendays52/dcd-img2rig.git
cd dcd-img2rig

python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
git clone https://github.com/sevendays52/dcd-img2rig.git
cd dcd-img2rig

python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. PyTorch 및 프로젝트 설치

> **Windows / Linux (NVIDIA GPU 사용자)** 는 `pip install -e .` 전에 CUDA용 PyTorch를 먼저 설치하는 것을 권장합니다. (macOS Apple Silicon은 기본 PyTorch가 MPS를 지원합니다.)

```bash
# (Windows/Linux CUDA 예시 — 본인 CUDA 버전에 맞게 PyTorch 공식 안내 참고)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 프로젝트 및 나머지 의존성 설치
pip install -e .
```

### 3. Blender 설치 및 확인

공식 [Blender 다운로드](https://www.blender.org/download/)에서 설치하면 자동으로 경로를 찾습니다.
특수 경로에 설치했거나 여러 버전 중 하나를 지정하려면 `BLENDER_BIN` 환경변수를 설정하세요.

```bash
# macOS / Linux 예시
export BLENDER_BIN="/Applications/Blender.app/Contents/MacOS/Blender"

# Windows PowerShell 예시
$env:BLENDER_BIN="C:\Program Files\Blender Foundation\Blender 4.3\blender.exe"
```

---

## 모델 가중치 & Hugging Face 설정

이미지 → 3DGS 생성([Idol](https://huggingface.co/speridlabs/idol))을 처음 실행할 때는 Hugging Face Hub에서 모델 가중치와 게이티드 인코더(`dinov3`, `FLUX.2 VAE`)를 내려받습니다.

1. 아래 두 모델 페이지에서 **이용 약관(Access)** 에 동의하세요.
   - https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m
   - https://huggingface.co/black-forest-labs/FLUX.2-dev
2. 터미널에서 Hugging Face 토큰으로 로그인합니다.
   ```bash
   hf auth login
   # 또는 환경변수 사용:
   # export HF_TOKEN=hf_xxx          (macOS/Linux)
   # $env:HF_TOKEN="hf_xxx"          (Windows PowerShell)
   ```

> **오프라인 / 로컬 가중치 사용 시**
> 루트의 `encoders/dinov3-h16plus/`, `encoders/flux2-vae-klein/` 또는 `IDOL_WEIGHTS`, `IDOL_ENCODERS_DIR` 환경변수 경로에 로컬 가중치가 있으면 인터넷/게이티드 체크 없이 바로 로드됩니다.

---

## 빠른 시작: 웹 스튜디오 실행

```bash
dcd-img2rig
# 또는: python -m img2rig.server --port 8765
```

실행 후 브라우저에서 **http://127.0.0.1:8765** 로 접속합니다.

### 웹 스튜디오 사용 흐름
1. **이미지 → 3D 변환**: 오른쪽 패널에서 캐릭터/오브젝트 이미지(PNG/JPG)를 업로드하고 `원스톱 3D 캐릭터 생성`을 누릅니다.
   - 렌더링/일러스트/AI 생성 이미지 → **Synthetic** 모드
   - 실사 사진 → **Real** 모드
2. **메시 & Binary FBX 7.4 변환**: 생성된 3D 스플랫을 선택한 뒤 `FBX 3D 메시` 패널에서 면 수(`target_faces`), 재구성 방식(`Marching Cubes` / `Poisson` / `Surface Fit`), 텍스처 베이크 옵션을 조정해 `.fbx`와 `_mesh.ply`를 생성합니다.
3. **자동 리깅 (`🦴 리깅 · 모션`)**: 19본 휴머노이드 리그를 캐릭터 체형에 자동 피팅하고 관절 위치를 미세 조정한 뒤 스키닝(`_skinned.glb`, `_skinned.fbx`)을 생성합니다.
4. **모션 리타게팅**: 내장 `Walk` 프리셋을 바로 적용하거나, [Mixamo](https://www.mixamo.com/) 등에서 내려받은 `.fbx` / `.glb` / `.bvh` 모션 파일을 업로드해 실시간으로 재생·검수할 수 있습니다.

---

## CLI 사용법

웹 UI 없이 각 단계를 커맨드라인에서 단독 실행할 수도 있습니다.

```bash
# 1. 단일 이미지 -> 3DGS (.ply)
img2rig-infer --image path/to/character.png --mode synthetic --out outputs

# 2. 3DGS (.ply) -> 삼각 메시 + Binary FBX 7.4 (+ 자동 리깅 옵션 --rig)
img2rig-fbx --ply outputs/character.ply --out outputs/character --target_faces 30000 --rig assets/rigs/sweetie_rig.glb

# 3. 기존 메시(_mesh.ply)에 별도로 자동 리깅 적용
img2rig-rig --mesh_ply outputs/character_mesh.ply --out_glb outputs/character_skinned.glb --out_fbx outputs/character_skinned.fbx

# 4. Mixamo / FBX / BVH 모션 리타게팅
img2rig-retarget --skinned outputs/character_skinned.glb --motion assets/rigs/presets/walk.fbx --out outputs/character_anim_walk.glb

# 5. 회전 미리보기 GIF 생성
img2rig-preview --ply outputs/character.ply --out outputs/character_preview.gif --frames 16 --width 256 --height 256
```

---

## 생성되는 파일 정리 (`outputs/`)

| 파일명 | 설명 |
|---|---|
| `<name>.ply` | 생성된 원본 3D Gaussian Splat |
| `<name>_condition.png` | 배경 제거 및 크롭된 모델 입력 이미지 |
| `<name>_mesh.ply`, `_mesh_uv.bin`, `_texture.png` | 정제된 삼각 메시, UV 버퍼, 베이크된 텍스처 |
| `<name>.fbx` | 정적 Binary FBX 7.4 파일 |
| `<name>_skinned.glb`, `<name>_skinned.fbx` | 19본 스켈레톤 + 스킨 웨이트가 바인딩된 리깅 모델 |
| `<name>_anim_<motion>.glb` | 모션이 리타게팅된 애니메이션 GLB |
| `<name>_preview.gif`, `<name>_mesh_preview.gif` | 3DGS / 메시 회전 썸네일 |

---

## 디렉터리 구조

```text
dcd-img2rig/
├── pyproject.toml              # 패키지 설정 및 CLI 엔트리포인트
├── .env.example                # 환경변수 설정 예시
├── src/img2rig/
│   ├── config.py               # 크로스 플랫폼 경로, GPU/MPS/CPU, Blender 탐색
│   ├── inference.py            # 이미지 -> 3DGS 추론 (Idol)
│   ├── mesh_convert.py         # 3DGS -> 삼각 메시, UV 텍스처 베이크
│   ├── fbx_writer.py           # Blender / 순수 Python Binary FBX 7.4 라이터
│   ├── rig_bind.py             # 해부학적 19본 피팅 & 스킨 웨이트 바인딩
│   ├── motion_retarget.py      # Mixamo/FBX/BVH 모션 리타게팅
│   ├── preview.py              # CPU 기반 오빗 썸네일 렌더러
│   └── server.py               # 웹 스튜디오 HTTP 서버 & 작업 큐
├── viewer/                     # WebGL2 프론트엔드 (index.html, app.js, mesh-viewer.js, gsplat.js)
├── assets/rigs/                # 기본 휴머노이드 리그(sweetie_rig.glb) 및 Walk 프리셋(walk.fbx)
└── tests/                      # 스모크 / 통합 테스트
```

---

## 환경변수 옵션 (`.env.example` 참고)

| 변수명 | 기본값 | 설명 |
|---|---|---|
| `IMG2RIG_HOST` | `127.0.0.1` | 웹 서버 바인딩 호스트 |
| `IMG2RIG_PORT` | `8765` | 웹 서버 포트 |
| `IMG2RIG_DATA_DIR` | 레포 루트 | `outputs/`, `uploads/`를 저장할 기준 경로 |
| `IDOL_WEIGHTS` | `speridlabs/idol` | 로컬 Idol 가중치 디렉터리 또는 HF repo ID |
| `IDOL_ENCODERS_DIR` | `./encoders` | 로컬 인코더(`dinov3-h16plus`, `flux2-vae-klein`) 경로 |
| `IDOL_DEVICE` | 자동 감지 | `cuda`, `mps`, `cpu` 중 강제 지정 |
| `IDOL_COMPILE` | `0` (Windows) / 자동(Linux CUDA) | `1` 설정 시 CUDA에서 `torch.compile` 활성화 |
| `BLENDER_BIN` | 자동 탐색 | Blender 실행 파일 직접 지정 |

---

## 테스트 실행

```bash
pip install -e ".[dev]"
pytest
# 또는 PyTest 없이 바로 실행:
python tests/test_rig_and_fbx.py
```
