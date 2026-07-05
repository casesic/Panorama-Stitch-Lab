#!/usr/bin/env bash
# Setup + smoke test for the RTMLib YOLOX person-detector backend.
#
# Goals (see task spec):
#   - Operate inside the *current* DenseKP conda/python env (no new env, no
#     system python changes).
#   - Check Python / numpy / opencv / rtmlib / onnxruntime / CUDAExecutionProvider
#     / GPU visibility.
#   - Install ONLY rtmlib (+ tqdm) if missing, with --no-deps, so torch / numpy /
#     opencv / onnxruntime / detectron2 are NOT upgraded or removed.
#   - Download the YOLOX-s HumanArt model into a project-internal directory,
#     idempotently.
#   - Run a real detector smoke test; print SETUP_PASS only if the detector
#     actually runs on CUDAExecutionProvider and returns a valid bbox.
#   - Never silently fall back to CPU.
#
# Usage:  bash scripts/setup_rtmlib_yolox_detector.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TOOL_ROOT="${REPO_ROOT}/tools/CameraHMR"
CACHE_DIR="${TOOL_ROOT}/checkpoints/rtmlib_yolox"
MODEL_URL="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_s_8xb8-300e_humanart-3ef259a7.zip"

PY="$(command -v python)"
PIP="$(command -v pip)"

blocked() {
  echo ""
  echo "REASON: $*"
  echo "SETUP_BLOCKED"
  exit 1
}

echo "=================================================================="
echo " RTMLib YOLOX detector setup"
echo "=================================================================="
echo "repo_root          : ${REPO_ROOT}"
echo "tool_root          : ${TOOL_ROOT}"
echo "model cache dir    : ${CACHE_DIR}"
echo "python             : ${PY}"
echo "pip                : ${PIP}"
echo "CONDA_DEFAULT_ENV  : ${CONDA_DEFAULT_ENV:-<none>}"
echo "CONDA_PREFIX       : ${CONDA_PREFIX:-<none>}"
"${PY}" -c "import sys; print('python_version     :', sys.version.split()[0]); print('sys.executable     :', sys.executable)"
echo "------------------------------------------------------------------"

[ -d "${TOOL_ROOT}" ] || blocked "tool root not found: ${TOOL_ROOT}"
mkdir -p "${CACHE_DIR}"

# --- core deps that must already exist (do NOT install/upgrade these) ---
echo "[1/5] Checking pre-existing core deps (numpy / opencv / onnxruntime)..."
"${PY}" - <<'PY' || blocked "core deps (numpy/opencv/onnxruntime) missing in this env"
import importlib, sys
ok = True
for m in ["numpy", "cv2", "onnxruntime"]:
    try:
        mod = importlib.import_module(m)
        print(f"   {m}: OK {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"   {m}: MISSING ({e})"); ok = False
sys.exit(0 if ok else 1)
PY

# --- onnxruntime CUDAExecutionProvider availability ---
echo "[2/5] Checking onnxruntime CUDAExecutionProvider..."
"${PY}" - <<'PY' || blocked "onnxruntime CUDAExecutionProvider not available (do NOT silently use CPU)"
import sys, onnxruntime as ort
provs = ort.get_available_providers()
print("   available_providers:", provs)
sys.exit(0 if "CUDAExecutionProvider" in provs else 1)
PY

# --- GPU visibility (informational, but required to pass) ---
echo "[3/5] Checking GPU visibility..."
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
else
  echo "   nvidia-smi not found (will rely on onnxruntime CUDA provider check)"
fi

# --- ensure rtmlib + tqdm present (install ONLY these, --no-deps) ---
echo "[4/5] Ensuring rtmlib + tqdm (install with --no-deps if missing)..."
ensure_pkg_nodeps() {
  local mod="$1" pkg="$2"
  if "${PY}" -c "import ${mod}" >/dev/null 2>&1; then
    echo "   ${mod}: already installed"
  else
    echo "   ${mod}: MISSING -> installing ${pkg} (--no-deps, no upgrades to torch/numpy/opencv/onnxruntime/detectron2)"
    "${PIP}" install --no-deps "${pkg}" || return 1
  fi
}
ensure_pkg_nodeps tqdm "tqdm" || blocked "failed to install tqdm"
ensure_pkg_nodeps rtmlib "rtmlib" || blocked "failed to install rtmlib"

"${PY}" - <<'PY' || blocked "rtmlib import failed after install"
from importlib.metadata import version
import rtmlib  # noqa
print("   rtmlib:", version("rtmlib"))
PY

# Guard: detectron2 must remain importable (we must not break it).
"${PY}" -c "import detectron2; print('   detectron2 still OK:', detectron2.__version__)" \
  || echo "   WARNING: detectron2 not importable (legacy backend may be unavailable)"

# --- detector smoke test (downloads/caches model + real CUDA detection) ---
echo "[5/5] Running detector smoke test (model prep + CUDA detection)..."
SMOKE_IMG="${REPO_ROOT}/data_process/Output/20260311_riding/image/equirect_crop_rotate_upright/cam2/00513.jpg"

cd "${TOOL_ROOT}" || blocked "cannot cd to tool root"
RTMLIB_SETUP_SMOKE_IMG="${SMOKE_IMG}" RTMLIB_SETUP_MODEL_URL="${MODEL_URL}" \
  "${PY}" - <<'PY'
import os, sys, glob, hashlib
import numpy as np

img_path = os.environ.get("RTMLIB_SETUP_SMOKE_IMG", "")
if not os.path.exists(img_path):
    # fall back to any upright image, else synthesize
    cands = glob.glob(os.path.join(
        "..", "..", "data_process", "Output", "*", "image",
        "equirect_crop_rotate_upright", "*", "*.jpg"))
    img_path = cands[0] if cands else ""

import cv2
if img_path and os.path.exists(img_path):
    img = cv2.imread(img_path)
    src = img_path
else:
    img = (np.ones((640, 640, 3), np.uint8) * 127)
    src = "<synthetic-gray-640>"

from detectors import build_person_detector
det = build_person_detector(backend="rtmlib_yolox", threshold=0.25)
md = det.metadata

print("   rtmlib_version      :", md["rtmlib_version"])
print("   onnxruntime_version :", md["onnxruntime_version"])
print("   available_providers :", md["available_providers"])
print("   session_providers   :", md["session_providers"])
print("   selected_provider   :", md["selected_provider"])
print("   model_identifier    :", md["detector_model_identifier"])
print("   model_local_path    :", md["detector_model_local_path_or_cache_path"])
print("   model_sha256        :", md["model_sha256_if_local_file_exists"])
print("   input_size          :", md["input_size"])
print("   device              :", md["device"])

# Hard requirement: CUDA, not CPU.
if md["selected_provider"] != "CUDAExecutionProvider":
    print("   FAIL: selected provider is not CUDAExecutionProvider")
    sys.exit(2)
if not os.path.exists(md["detector_model_local_path_or_cache_path"]):
    print("   FAIL: model not present locally")
    sys.exit(3)

dets = det.detect(img)
print(f"   smoke image         : {src}  shape={img.shape}")
print(f"   detections          : {len(dets)}")
valid_bbox = False
for d in dets:
    b = d["bbox_xyxy"]
    ok = bool(b[0] < b[2] and b[1] < b[3] and b.dtype == np.float32)
    print("     bbox", np.round(b, 2).tolist(), "score", round(d["score"], 4),
          "cls", d["class_id"], "valid", ok)
    valid_bbox = valid_bbox or ok

if len(dets) == 0:
    print("   WARN: zero detections on smoke image; cannot confirm a valid bbox")
    sys.exit(4)
if not valid_bbox:
    print("   FAIL: no valid bbox produced")
    sys.exit(5)

print("   SMOKE_OK")
sys.exit(0)
PY
SMOKE_RC=$?

case "${SMOKE_RC}" in
  0) ;;
  2) blocked "detector did not run on CUDAExecutionProvider (CPU fallback rejected)";;
  3) blocked "YOLOX-s model not present locally after setup";;
  4) blocked "smoke image produced zero detections (cannot validate bbox)";;
  5) blocked "detector produced an invalid bbox";;
  *) blocked "smoke test failed (rc=${SMOKE_RC})";;
esac

echo "------------------------------------------------------------------"
echo "All checks passed:"
echo "  - rtmlib import OK"
echo "  - onnxruntime import OK + CUDAExecutionProvider available"
echo "  - YOLOX-s HumanArt model present locally in project dir"
echo "  - single image detected on CUDAExecutionProvider"
echo "  - valid bbox produced"
echo "SETUP_PASS"
