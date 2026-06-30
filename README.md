# WAN2.2 VACE Video Editing

Text-driven video editing using the WAN2.2 VACE Fun A14B model. Give it a source video and a text prompt — it produces an edited video that preserves the original motion while applying whatever change the prompt describes.

**What you can do with a prompt:**
- Add a held prop ("person holding a red umbrella")
- Change clothing ("same person in a winter jacket")
- Add background elements ("a cat sitting in the corner")
- Change lighting or style ("same scene at night with neon lights")

The prompt is the only thing you change between runs.

---

## System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| OS | Ubuntu 20.04 | Ubuntu 22.04 / 24.04 |
| GPU | NVIDIA 12 GB VRAM | NVIDIA 24 GB VRAM |
| CUDA | 11.8 | 12.1+ |
| Python | 3.10 | 3.11 |
| RAM | 16 GB | 32 GB |
| Disk | 40 GB free | 60 GB free |
| Tools | `git`, `curl`, `ffmpeg` | same |

> On 12 GB VRAM (e.g. RTX 3060): runs with block swapping, ~21 min per 3-second clip.  
> On 24 GB VRAM (e.g. RTX 3090/4090): no block swapping needed, ~3 min per clip.

---

## One-Command Setup

```bash
git clone https://github.com/lakshmikanthgr/aniamlswap.git
cd aniamlswap
bash setup.sh
```

That's it. The script handles everything:
1. Checks your GPU, Python, and system tools
2. Clones and installs ComfyUI
3. Installs custom nodes (WanVideoWrapper, VideoHelperSuite)
4. Applies required patches to WanVideoWrapper
5. Downloads all three model files (~17 GB total)
6. Creates the project structure and a placeholder test video
7. Verifies everything and prints next steps

**Optional:** override the ComfyUI install location:
```bash
COMFYUI_DIR=/data/ComfyUI bash setup.sh
```

### System tool prerequisites

If any are missing, install them before running setup:
```bash
sudo apt update && sudo apt install -y git curl ffmpeg python3.10 python3.10-venv
```

---

## Usage

### 1. Start ComfyUI

```bash
cd ~/projects/ComfyUI
source venv/bin/activate
python main.py --lowvram
```

Wait for: `To see the GUI go to: http://127.0.0.1:8188`

### 2. Place your source video

```bash
cp /path/to/your/video.mp4 ~/projects/ComfyUI/input/source.mp4
```

**Video tips:**
- Resolution: 832×480 (other sizes work, affects VRAM)
- Frame rate: 16 fps (the model resamples to this)
- Length: 3 seconds / 49 frames is the tested default
- Format: any format ffmpeg can read (mp4, mov, avi, etc.)

### 3. Set your prompt

Open `workflows/vace_prop_addition.json` and edit **node 4**:

```json
"positive_prompt": "describe the full scene with the change you want",
"negative_prompt": "blurry, distorted, floating objects, flickering, inconsistent motion"
```

The positive prompt should describe the **full scene**, not just what changed. See [Prompt Guide](#prompt-guide) below.

Also update **node 1** if your filename differs from `source.mp4`:
```json
"video": "your_filename.mp4"
```

And **node 3** if your video has different dimensions or length:
```json
"width": 832,
"height": 480,
"num_frames": 49
```

### 4. Run

```bash
python3 scripts/run_workflow.py
```

Output is saved to: `~/projects/ComfyUI/output/vace_output_00001.mp4`

---

## Prompt Guide

### Template

```
[subject description], [what you want added or changed], [motion/pose context], [quality terms]
```

### Examples

**Add a held object:**
```
positive: a person walking down the street holding a large red balloon,
          same natural walk cycle, photorealistic, consistent lighting
negative: blurry, floating objects, disconnected prop, flickering
```

**Change clothing:**
```
positive: same person wearing a bright yellow raincoat and boots,
          identical movement and pose, photorealistic
negative: morphing, inconsistent texture, flickering outfit
```

**Add a background element:**
```
positive: same room scene, a small dog sitting near the doorway in the background,
          natural lighting, photorealistic
negative: distorted, extra people, flickering background
```

**Style change:**
```
positive: same scene in a hand-drawn animation style, warm colors, consistent motion
negative: photorealistic, blurry, flickering
```

### Key rules

- **Always describe the original subject** — the model needs to know what to keep, not just what to add.
- **Keep the negative prompt generic** — `blurry, distorted, flickering, inconsistent motion` covers most cases.
- **Strength (node 3)** controls how closely the output follows the source motion:
  - `0.85–1.0` — strong motion preservation
  - `0.5–0.8` — more creative freedom
  - below `0.5` — starts ignoring source motion

---

## Adjusting for Your GPU

Edit **node 9** (`WanVideoBlockSwap`) in the workflow:

| GPU VRAM | `blocks_to_swap` | `vace_blocks_to_swap` | Speed |
|---|---|---|---|
| 12 GB | 30 | 8 | ~94 s/step |
| 16 GB | 20 | 5 | ~40 s/step |
| 24 GB+ | 0 | 0 | ~8 s/step |

A14B has 40 transformer blocks and 15 VACE blocks total.

---

## Models Used

| Model | Size | Source |
|---|---|---|
| `Wan2.2-VACE-Fun-A14B-low-noise-Q4_K_M.gguf` | 10.8 GB | QuantStack/Wan2.2-VACE-Fun-A14B-GGUF |
| `Wan2_1_VAE_bf16.safetensors` | ~400 MB | Wan-AI/Wan2.1-VAE |
| `umt5-xxl-enc-fp8_e4m3fn.safetensors` | ~5 GB | Comfy-Org/mochi_preview_repackaged |

All downloaded automatically by `setup.sh`.

---

## Project Structure

```
aniamlswap/
├── setup.sh                        # Full setup script — run this first
├── README.md                       # This file
├── workflows/
│   └── vace_prop_addition.json     # ComfyUI workflow — edit prompts here
├── scripts/
│   └── run_workflow.py             # Submit and poll the workflow via API
├── docs/
│   ├── architecture.md             # Node-by-node technical breakdown
│   ├── user_guide.md               # Detailed usage guide
│   └── test_report.md              # Initial run log and bug fixes
├── input/                          # Place source videos here (or use ComfyUI/input/)
└── output/                         # Outputs land in ComfyUI/output/
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `nvidia-smi not found` | NVIDIA drivers not installed | Install drivers: `ubuntu-drivers autoinstall` |
| `torch.cuda.is_available() = False` | Wrong PyTorch build | Reinstall: see CUDA version in `nvidia-smi`, match torch index URL |
| OOM during sampling | Not enough VRAM | Increase `blocks_to_swap` to 35-38 in node 9 |
| Sampling takes >2 hrs | Too many steps or wrong settings | Reduce `steps` in node 6 to 15; confirm block swap is set |
| Output video is black | Wrong VAE or failed decode | Confirm VAE is `Wan2_1_VAE_bf16.safetensors`, not `wan2.2_vae.safetensors` |
| `git pull` breaks the workflow | WanVideoWrapper update overwrote patches | Re-run `bash setup.sh` — the patch step is idempotent |
| Model not found in ComfyUI | Model in wrong directory | Check: `ls ~/projects/ComfyUI/models/unet/LowNoise/` |

For the full list of bugs hit during initial setup and their fixes, see [docs/test_report.md](docs/test_report.md).

---

## Docs

- [Architecture](docs/architecture.md) — how each node works, latent shapes, VACE channel math
- [User Guide](docs/user_guide.md) — prompts, masks, reference images, detailed settings
- [Test Report](docs/test_report.md) — every error hit during bring-up and the fix applied
