#!/usr/bin/env bash
# WAN2.2 VACE Video Editing — Full Setup Script
# Tested on: Ubuntu 20.04 / 22.04 / 24.04, NVIDIA GPU ≥12GB VRAM, CUDA 11.8 / 12.x
# Usage: bash setup.sh
#        COMFYUI_DIR=/custom/path bash setup.sh   (override install location)

set -euo pipefail

# ── Colours ───────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
log()  { echo -e "${GREEN}[setup]${NC} $*"; }
info() { echo -e "${CYAN}[info]${NC}  $*"; }
warn() { echo -e "${YELLOW}[warn]${NC}  $*"; }
die()  { echo -e "${RED}[error]${NC} $*" >&2; exit 1; }
step() { echo -e "\n${CYAN}━━━ $* ━━━${NC}"; }

# ── Config ────────────────────────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMFYUI_DIR="${COMFYUI_DIR:-$HOME/projects/ComfyUI}"
PYTHON="${PYTHON:-python3}"
COMFYUI_REPO="https://github.com/comfyanonymous/ComfyUI.git"
WANVIDEO_REPO="https://github.com/kijai/ComfyUI-WanVideoWrapper.git"
VHS_REPO="https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git"

echo ""
echo "  WAN2.2 VACE Video Editing Setup"
echo "  ComfyUI target: $COMFYUI_DIR"
echo "  Project dir:    $PROJECT_DIR"
echo ""

# ══════════════════════════════════════════════════════════════════════════
step "1 / 8  System checks"
# ══════════════════════════════════════════════════════════════════════════

# NVIDIA GPU
command -v nvidia-smi > /dev/null 2>&1 || die "nvidia-smi not found. Install NVIDIA drivers first."
nvidia-smi > /dev/null 2>&1 || die "NVIDIA driver not responding. Check: nvidia-smi"
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
info "GPU: $GPU_NAME  (${GPU_MEM} MB)"
[ "$GPU_MEM" -ge 10000 ] || warn "GPU has ${GPU_MEM}MB — minimum recommended is 12GB. May OOM."

# Python 3.10+
command -v "$PYTHON" > /dev/null 2>&1 || die "python3 not found."
PY_VER=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
$PYTHON -c "import sys; assert sys.version_info >= (3,10), 'need 3.10+'" 2>/dev/null || \
  die "Python 3.10+ required. Found $PY_VER. Install: sudo apt install python3.10"
info "Python $PY_VER"

# System tools
for cmd in git curl ffmpeg; do
  command -v "$cmd" > /dev/null 2>&1 || die "'$cmd' not found. Run: sudo apt install $cmd"
done

# Disk space — models alone are ~17 GB; total ~25 GB
AVAIL=$(df -BG "$HOME" | awk 'NR==2{gsub(/G/,"",$4); print $4}')
[ "${AVAIL:-0}" -ge 40 ] || warn "Only ${AVAIL}GB free in \$HOME. Need ~40GB for models + ComfyUI."

log "System checks passed"

# ══════════════════════════════════════════════════════════════════════════
step "2 / 8  ComfyUI"
# ══════════════════════════════════════════════════════════════════════════

if [ -f "$COMFYUI_DIR/main.py" ]; then
  info "ComfyUI already present at $COMFYUI_DIR — skipping clone"
else
  log "Cloning ComfyUI..."
  mkdir -p "$(dirname "$COMFYUI_DIR")"
  git clone "$COMFYUI_REPO" "$COMFYUI_DIR"
fi

# Python venv
if [ ! -f "$COMFYUI_DIR/venv/bin/activate" ]; then
  log "Creating Python venv..."
  $PYTHON -m venv "$COMFYUI_DIR/venv"
fi

VENV_PY="$COMFYUI_DIR/venv/bin/python"
VENV_PIP="$COMFYUI_DIR/venv/bin/pip"

log "Installing ComfyUI requirements..."
$VENV_PIP install --upgrade pip --quiet
$VENV_PIP install -r "$COMFYUI_DIR/requirements.txt" --quiet

# CUDA-enabled torch (install only if torch not already present or is CPU-only)
if ! $VENV_PY -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  log "Installing PyTorch with CUDA support..."
  CUDA_VER=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1)
  if [[ "$CUDA_VER" == 12* ]]; then
    $VENV_PIP install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet
  elif [[ "$CUDA_VER" == 11* ]]; then
    $VENV_PIP install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --quiet
  else
    warn "Could not detect CUDA version ($CUDA_VER). Installing default torch — may be CPU-only."
    $VENV_PIP install torch torchvision torchaudio --quiet
  fi
fi

$VENV_PY -c "import torch; assert torch.cuda.is_available(), 'torch CUDA not available'" || \
  die "PyTorch cannot see the GPU. Check CUDA driver compatibility."
info "PyTorch $($VENV_PY -c 'import torch; print(torch.__version__)') — CUDA available"

# Create model directories
mkdir -p \
  "$COMFYUI_DIR/models/unet/LowNoise" \
  "$COMFYUI_DIR/models/vae" \
  "$COMFYUI_DIR/models/text_encoders" \
  "$COMFYUI_DIR/input" \
  "$COMFYUI_DIR/output"

log "ComfyUI ready"

# ══════════════════════════════════════════════════════════════════════════
step "3 / 8  Custom nodes"
# ══════════════════════════════════════════════════════════════════════════

NODES_DIR="$COMFYUI_DIR/custom_nodes"
mkdir -p "$NODES_DIR"

# WanVideoWrapper
WANVIDEO_DIR="$NODES_DIR/ComfyUI-WanVideoWrapper"
if [ -d "$WANVIDEO_DIR" ]; then
  info "WanVideoWrapper already present — skipping clone"
else
  log "Cloning ComfyUI-WanVideoWrapper..."
  git clone "$WANVIDEO_REPO" "$WANVIDEO_DIR"
fi
if [ -f "$WANVIDEO_DIR/requirements.txt" ]; then
  $VENV_PIP install -r "$WANVIDEO_DIR/requirements.txt" --quiet
fi

# VideoHelperSuite
VHS_DIR="$NODES_DIR/ComfyUI-VideoHelperSuite"
if [ -d "$VHS_DIR" ]; then
  info "VideoHelperSuite already present — skipping clone"
else
  log "Cloning ComfyUI-VideoHelperSuite..."
  git clone "$VHS_REPO" "$VHS_DIR"
fi
if [ -f "$VHS_DIR/requirements.txt" ]; then
  $VENV_PIP install -r "$VHS_DIR/requirements.txt" --quiet
fi

log "Custom nodes installed"

# ══════════════════════════════════════════════════════════════════════════
step "4 / 8  Patch WanVideoWrapper"
# ══════════════════════════════════════════════════════════════════════════
# The stock WanVideoWrapper has 3 bugs when used with the VACE Fun A14B model.
# This patch is idempotent — safe to run multiple times.

NODES_PY="$WANVIDEO_DIR/nodes.py"
[ -f "$NODES_PY" ] || die "nodes.py not found at $NODES_PY"

$VENV_PY - "$NODES_PY" <<'PYEOF'
import sys, re

path = sys.argv[1]
with open(path) as f:
    src = f.read()

original = src
patches_applied = 0

# ── Patch 1: vace_latent ─────────────────────────────────────────────────
# Fix: always clip/fill to 96 channels instead of blindly concatenating z+mask.
OLD_VACE_LATENT = 'def vace_latent(self, z, m):\n        return [torch.cat([zz, mm], dim=0) for zz, mm in zip(z, m)]'
NEW_VACE_LATENT = '''def vace_latent(self, z, m):
        # VACE conv expects exactly 96 input channels.
        # VAE16: z(32) + mask(64) = 96. VAE38: z(96) alone = 96 (no mask needed).
        VACE_CONV_CHANNELS = 96
        result = []
        for zz, mm in zip(z, m):
            if zz.shape[0] >= VACE_CONV_CHANNELS:
                result.append(zz[:VACE_CONV_CHANNELS])
            else:
                needed = VACE_CONV_CHANNELS - zz.shape[0]
                result.append(torch.cat([zz, mm[:needed]], dim=0))
        return result'''

if OLD_VACE_LATENT in src:
    src = src.replace(OLD_VACE_LATENT, NEW_VACE_LATENT)
    patches_applied += 1
    print("[patch] Applied: vace_latent channel fix")
elif 'VACE_CONV_CHANNELS = 96' in src:
    print("[patch] Already applied: vace_latent channel fix")
else:
    print("[warn] Could not find vace_latent — may need manual patch")

# ── Patch 2: vace_encode_masks signature ─────────────────────────────────
# Fix: accept spatial_stride param instead of hardcoding VAE_STRIDE[1]=8.
OLD_MASK_SIG = 'def vace_encode_masks(self, masks, ref_images=None):'
NEW_MASK_SIG = 'def vace_encode_masks(self, masks, ref_images=None, spatial_stride=None):'
if OLD_MASK_SIG in src:
    src = src.replace(OLD_MASK_SIG, NEW_MASK_SIG)
    # Also replace the hardcoded stride uses inside the function body
    # Find the first use of VAE_STRIDE[1] after the new signature
    src = src.replace(
        'def vace_encode_masks(self, masks, ref_images=None, spatial_stride=None):\n',
        'def vace_encode_masks(self, masks, ref_images=None, spatial_stride=None):\n'
        '        if spatial_stride is None:\n'
        '            spatial_stride = VAE_STRIDE[1]\n',
        1
    )
    patches_applied += 1
    print("[patch] Applied: vace_encode_masks spatial_stride param")
elif 'def vace_encode_masks(self, masks, ref_images=None, spatial_stride=None):' in src:
    print("[patch] Already applied: vace_encode_masks spatial_stride param")
else:
    print("[warn] Could not find vace_encode_masks signature — may need manual patch")

# ── Patch 3: replace VAE_STRIDE[1] inside vace_encode_masks body ─────────
# After the function is patched to have a spatial_stride param, replace the
# hardcoded VAE_STRIDE[1] uses within that function's body.
# Strategy: replace occurrences of VAE_STRIDE[1] in the mask-related reshape math.
if 'mask = mask.view(depth, height, VAE_STRIDE[1], width, VAE_STRIDE[1])' in src:
    src = src.replace(
        'mask = mask.view(depth, height, VAE_STRIDE[1], width, VAE_STRIDE[1])',
        'mask = mask.view(depth, height, spatial_stride, width, spatial_stride)'
    )
    src = src.replace(
        'mask = mask.reshape(VAE_STRIDE[1] * VAE_STRIDE[1], depth, height, width)',
        'mask = mask.reshape(spatial_stride * spatial_stride, depth, height, width)'
    )
    patches_applied += 1
    print("[patch] Applied: vace_encode_masks body stride fix")
elif 'mask = mask.view(depth, height, spatial_stride, width, spatial_stride)' in src:
    print("[patch] Already applied: vace_encode_masks body stride fix")
else:
    print("[warn] Could not find VAE_STRIDE[1] in mask reshape — may need manual patch")

# ── Patch 4: WanVideoVACEEncode.process — dynamic spatial_stride + z_dim ─
OLD_TARGET = (
    'target_shape = (16, (num_frames - 1) // VAE_STRIDE[0] + 1,\n'
    '                        height // VAE_STRIDE[1],\n'
    '                        width // VAE_STRIDE[1])'
)
NEW_TARGET = (
    '_spatial_stride = getattr(vae, \'upsampling_factor\', VAE_STRIDE[1])\n'
    '        _z_dim = getattr(vae, \'z_dim\', 16)\n'
    '        target_shape = (_z_dim, (num_frames - 1) // VAE_STRIDE[0] + 1,\n'
    '                        height // _spatial_stride,\n'
    '                        width // _spatial_stride)'
)
if OLD_TARGET in src:
    src = src.replace(OLD_TARGET, NEW_TARGET)
    patches_applied += 1
    print("[patch] Applied: WanVideoVACEEncode.process target_shape fix")
elif '_z_dim = getattr(vae, \'z_dim\', 16)' in src:
    print("[patch] Already applied: WanVideoVACEEncode.process target_shape fix")
else:
    print("[warn] Could not find target_shape assignment — may need manual patch")

# ── Patch 5: pass spatial_stride to vace_encode_masks call ───────────────
OLD_MASK_CALL = 'm0 = self.vace_encode_masks(input_masks, ref_images)'
NEW_MASK_CALL = (
    'spatial_stride = getattr(vae, \'upsampling_factor\', VAE_STRIDE[1])\n'
    '        m0 = self.vace_encode_masks(input_masks, ref_images, spatial_stride=spatial_stride)'
)
if OLD_MASK_CALL in src:
    src = src.replace(OLD_MASK_CALL, NEW_MASK_CALL)
    patches_applied += 1
    print("[patch] Applied: pass spatial_stride to vace_encode_masks")
elif 'm0 = self.vace_encode_masks(input_masks, ref_images, spatial_stride=spatial_stride)' in src:
    print("[patch] Already applied: pass spatial_stride to vace_encode_masks")
else:
    print("[warn] Could not find vace_encode_masks call site — may need manual patch")

if src != original:
    with open(path, 'w') as f:
        f.write(src)
    print(f"[patch] Wrote {patches_applied} patch(es) to {path}")
else:
    print(f"[patch] No changes needed — all patches already applied")
PYEOF

log "WanVideoWrapper patched"

# ══════════════════════════════════════════════════════════════════════════
step "5 / 8  Download models"
# ══════════════════════════════════════════════════════════════════════════

$VENV_PIP install huggingface_hub --quiet

$VENV_PY - "$COMFYUI_DIR" <<'PYEOF'
import sys, os
from pathlib import Path
from huggingface_hub import hf_hub_download

comfy = Path(sys.argv[1])

models = [
    # (repo_id, filename, local_dir)
    (
        "QuantStack/Wan2.2-VACE-Fun-A14B-GGUF",
        "Wan2.2-VACE-Fun-A14B-low-noise-Q4_K_M.gguf",
        comfy / "models/unet/LowNoise",
    ),
    (
        "Wan-AI/Wan2.1-VAE",
        "Wan2.1_VAE.safetensors",
        comfy / "models/vae",
    ),
    (
        "Comfy-Org/mochi_preview_repackaged",
        "text_encoders/umt5-xxl-enc-fp8_e4m3fn.safetensors",
        comfy / "models/text_encoders",
    ),
]

# Try a fallback VAE source if the primary doesn't have it
vae_candidates = [
    ("Wan-AI/Wan2.1-VAE",   "Wan2.1_VAE.safetensors",          "Wan2_1_VAE_bf16.safetensors"),
    ("kijai/WanVideo_comfy", "Wan2_1_VAE_bf16.safetensors",     "Wan2_1_VAE_bf16.safetensors"),
]

for repo, filename, local_dir in models:
    local_dir = Path(local_dir)
    dest = local_dir / Path(filename).name
    # Handle subdirectory filenames (text_encoders/xxx)
    dest = local_dir / Path(filename).name
    if dest.exists():
        size_gb = dest.stat().st_size / 1e9
        print(f"[skip] {dest.name} already exists ({size_gb:.1f} GB)")
        continue
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"[download] {repo} / {filename} → {local_dir}")
    try:
        hf_hub_download(
            repo_id=repo,
            filename=filename,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )
        print(f"[ok] {Path(filename).name}")
    except Exception as e:
        print(f"[error] Failed: {e}")
        raise

# Normalise VAE filename to what the workflow expects
vae_dir = comfy / "models/vae"
expected = vae_dir / "Wan2_1_VAE_bf16.safetensors"
if not expected.exists():
    for candidate in vae_dir.glob("*.safetensors"):
        if "wan" in candidate.name.lower() and "vae" in candidate.name.lower() and "2.2" not in candidate.name:
            candidate.rename(expected)
            print(f"[renamed] {candidate.name} → Wan2_1_VAE_bf16.safetensors")
            break
PYEOF

log "Models downloaded"

# ══════════════════════════════════════════════════════════════════════════
step "6 / 8  Project structure"
# ══════════════════════════════════════════════════════════════════════════

mkdir -p "$PROJECT_DIR"/{workflows,scripts,docs,input,output}

# Copy workflow if not already there
if [ ! -f "$PROJECT_DIR/workflows/vace_prop_addition.json" ]; then
  warn "workflows/vace_prop_addition.json not found in $PROJECT_DIR — download the repo."
fi

# Symlink ComfyUI input/output into project for convenience
ln -sfn "$COMFYUI_DIR" "$PROJECT_DIR/comfyui_link" 2>/dev/null || true

log "Project structure ready"

# ══════════════════════════════════════════════════════════════════════════
step "7 / 8  Placeholder test video"
# ══════════════════════════════════════════════════════════════════════════

TEST_VIDEO="$COMFYUI_DIR/input/source.mp4"
if [ -f "$TEST_VIDEO" ]; then
  info "Test video already exists — skipping"
else
  log "Generating placeholder test video (832×480, 3s, 16fps)..."
  ffmpeg -y -f lavfi \
    -i "color=c=0x2d7a2d:size=832x480:rate=16" \
    -vf "drawtext=text='Replace with your video':fontsize=36:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2" \
    -t 3 -c:v libx264 -pix_fmt yuv420p \
    "$TEST_VIDEO" -loglevel error
  log "Placeholder video created: $TEST_VIDEO"
fi

# ══════════════════════════════════════════════════════════════════════════
step "8 / 8  Verify"
# ══════════════════════════════════════════════════════════════════════════

FAIL=0

check() {
  local label="$1"; local path="$2"
  if [ -e "$path" ]; then
    info "OK  $label"
  else
    echo -e "${RED}[miss]${NC} $label — expected at: $path"
    FAIL=1
  fi
}

check "ComfyUI main.py"             "$COMFYUI_DIR/main.py"
check "Python venv"                 "$COMFYUI_DIR/venv/bin/python"
check "WanVideoWrapper nodes.py"    "$NODES_DIR/ComfyUI-WanVideoWrapper/nodes.py"
check "VideoHelperSuite"            "$NODES_DIR/ComfyUI-VideoHelperSuite"
check "VACE model (GGUF)"           "$COMFYUI_DIR/models/unet/LowNoise/Wan2.2-VACE-Fun-A14B-low-noise-Q4_K_M.gguf"
check "VAE model"                   "$COMFYUI_DIR/models/vae/Wan2_1_VAE_bf16.safetensors"
check "Text encoder (T5)"           "$COMFYUI_DIR/models/text_encoders/umt5-xxl-enc-fp8_e4m3fn.safetensors"
check "Workflow JSON"               "$PROJECT_DIR/workflows/vace_prop_addition.json"
check "Test video"                  "$COMFYUI_DIR/input/source.mp4"

if [ "$FAIL" -eq 0 ]; then
  echo ""
  echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
  echo -e "${GREEN}  Setup complete.${NC}"
  echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
  echo ""
  echo "  Next steps:"
  echo ""
  echo "  1. Start ComfyUI:"
  echo "       cd $COMFYUI_DIR && source venv/bin/activate"
  echo "       python main.py --lowvram"
  echo ""
  echo "  2. Place your video:"
  echo "       cp /path/to/your/video.mp4 $COMFYUI_DIR/input/source.mp4"
  echo ""
  echo "  3. Edit the prompt in:"
  echo "       $PROJECT_DIR/workflows/vace_prop_addition.json  (node 4)"
  echo ""
  echo "  4. Run the workflow:"
  echo "       cd $PROJECT_DIR && python3 scripts/run_workflow.py"
  echo ""
  echo "  Output: $COMFYUI_DIR/output/vace_output_00001.mp4"
  echo ""
else
  echo ""
  die "Setup incomplete — see missing items above."
fi
