# User Guide — WAN2.2 VACE Prop-Addition

## What This Does

Takes a source video and adds props to it using AI video diffusion. The model sees the original motion and generates frames where new objects (props) are integrated, matching the existing lighting, movement, and style.

Example: source video of two birds → output video of the same birds, one holding a guitar.

---

## Prerequisites

| Component | Requirement |
|---|---|
| GPU | RTX 3060 12GB minimum (block swap required). 24GB+ recommended for speed. |
| ComfyUI | Installed at `~/projects/ComfyUI` with Python venv |
| OS | Ubuntu / Linux |

### Required models (download once)

| Model | Size | Location |
|---|---|---|
| `Wan2.2-VACE-Fun-A14B-low-noise-Q4_K_M.gguf` | 10.8 GB | `ComfyUI/models/unet/LowNoise/` |
| `Wan2_1_VAE_bf16.safetensors` | ~400 MB | `ComfyUI/models/vae/` |
| `umt5-xxl-enc-fp8_e4m3fn.safetensors` | ~5 GB | `ComfyUI/models/text_encoders/` |

Download the main model:
```bash
cd ~/projects/ComfyUI
source venv/bin/activate
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='QuantStack/Wan2.2-VACE-Fun-A14B-GGUF',
    filename='Wan2.2-VACE-Fun-A14B-low-noise-Q4_K_M.gguf',
    local_dir='models/unet/LowNoise',
    local_dir_use_symlinks=False
)
"
```

### Required custom nodes

Install via ComfyUI Manager or manually:

```bash
cd ~/projects/ComfyUI/custom_nodes

# WanVideoWrapper (main inference)
git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git
cd ComfyUI-WanVideoWrapper && pip install -r requirements.txt && cd ..

# VideoHelperSuite (load/save video)
git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
cd ComfyUI-VideoHelperSuite && pip install -r requirements.txt && cd ..
```

> **IMPORTANT:** After cloning WanVideoWrapper, apply the patches described in [Patches](#patches-required) before running. The unpatched version has bugs with the VACE Fun A14B model.

---

## Quick Start

### 1. Start ComfyUI

```bash
cd ~/projects/ComfyUI
source venv/bin/activate
python main.py --lowvram
```

Wait for: `To see the GUI go to: http://127.0.0.1:8188`

### 2. Prepare your source video

Copy your video to `ComfyUI/input/`:
```bash
cp /path/to/your/video.mp4 ~/projects/ComfyUI/input/birds_source.mp4
```

Target: 832×480, 16 fps, under 49 frames (~3 seconds). Other sizes work but affect VRAM.

### 3. Edit the workflow

Open `workflows/vace_prop_addition.json` and change:

**Node 1** — your video filename:
```json
"video": "your_video.mp4"
```

**Node 3** — match your video dimensions and frame count:
```json
"width": 832,
"height": 480,
"num_frames": 49
```

**Node 4** — your prompts:
```json
"positive_prompt": "describe what you want added or changed",
"negative_prompt": "describe what to avoid"
```

### 4. Run the workflow

```bash
cd ~/projects/video_swap
python3 scripts/run_workflow.py
```

Output will be at `~/projects/ComfyUI/output/birds_prop_test_00001.mp4`.

Expected runtime on RTX 3060 12GB: **~21 minutes** (94s/step × 20 steps).

---

## Prompt Writing Tips

**Positive prompt** — describe the full scene including the prop:
```
two birds on a branch, one bird holding a small red guitar,
natural lighting, photorealistic, consistent motion
```

**Negative prompt** — describe artifacts to suppress:
```
blurry, distorted, floating objects, extra limbs,
flickering, inconsistent motion
```

**Strength (node 3)** — controls how much the original video influences the output:
- `1.0` — fully guided by source motion
- `0.5–0.8` — model has more freedom to deviate
- `0.0` — ignores source completely (pure text-to-video)

**Steps (node 6)** — default 20 is a reasonable trade-off. More steps = better quality, longer runtime.

---

## Using a Mask

By default the full frame is eligible for editing. To restrict changes to a specific region, connect a `MASK` tensor to node 3's `input_masks` input.

- White (1.0) = the model can edit here
- Black (0.0) = preserve the original

You can create masks with any ComfyUI mask node, or use SAM2 (`ComfyUI-segment-anything-2`) to auto-segment a subject.

---

## Using a Reference Image

If you have an image of the prop you want to add, connect it to node 3's `ref_images` input. The model will use it as a visual reference for the prop's appearance. Useful for specific objects like a branded product or a particular instrument.

---

## Adjusting for Your GPU

**Node 9 — WanVideoBlockSwap** controls VRAM usage vs speed:

| GPU VRAM | blocks_to_swap | vace_blocks_to_swap | Expected speed |
|---|---|---|---|
| 12 GB | 30 | 8 | ~94s/step |
| 16 GB | 20 | 5 | ~40s/step |
| 24 GB | 0 | 0 | ~8s/step |

Set both to 0 to disable block swapping entirely (fastest, needs 24GB+).

---

## Workflow File Reference

`workflows/vace_prop_addition.json` — 9 nodes:

| Node | Class | Role |
|---|---|---|
| 1 | VHS_LoadVideo | Load source video |
| 2 | WanVideoVAELoader | Load VAE model |
| 3 | WanVideoVACEEncode | Encode source frames as VACE conditioning |
| 4 | WanVideoTextEncodeCached | Encode text prompts |
| 5 | WanVideoModelLoader | Load diffusion transformer |
| 6 | WanVideoSampler | Run denoising (the main generation step) |
| 7 | WanVideoDecode | Decode latents to frames |
| 8 | VHS_VideoCombine | Save frames as MP4 |
| 9 | WanVideoBlockSwap | VRAM management config |

For full details on each node's inputs and outputs, see [architecture.md](architecture.md).

---

## Patches Required

The stock `ComfyUI-WanVideoWrapper` has bugs when used with the VACE Fun A14B model. Three methods in `nodes.py` must be patched manually.

File: `~/projects/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper/nodes.py`

### Patch 1 — `WanVideoVACEEncode.process`

**Problem:** Hardcodes `z_dim=16` and `spatial_stride=8` in `target_shape` and mask computation regardless of which VAE is loaded.

**Fix:** Read both values from the VAE object:
```python
_spatial_stride = getattr(vae, 'upsampling_factor', VAE_STRIDE[1])
_z_dim = getattr(vae, 'z_dim', 16)
target_shape = (_z_dim, (num_frames - 1) // VAE_STRIDE[0] + 1,
                height // _spatial_stride,
                width // _spatial_stride)
```

And pass `spatial_stride` when calling `vace_encode_masks`:
```python
spatial_stride = getattr(vae, 'upsampling_factor', VAE_STRIDE[1])
m0 = self.vace_encode_masks(input_masks, ref_images, spatial_stride=spatial_stride)
```

### Patch 2 — `vace_encode_masks`

**Problem:** Uses hardcoded `VAE_STRIDE[1]` (=8) for mask downsampling.

**Fix:** Accept `spatial_stride` as a parameter:
```python
def vace_encode_masks(self, masks, ref_images=None, spatial_stride=None):
    if spatial_stride is None:
        spatial_stride = VAE_STRIDE[1]
    # then use spatial_stride throughout instead of VAE_STRIDE[1]
```

### Patch 3 — `vace_latent`

**Problem:** Always concatenates `z0 + m0`, which produces 352 channels for VAE38 (96+256) — but the VACE conv expects exactly 96 channels.

**Fix:** Clip to 96 channels, filling from mask only as needed:
```python
def vace_latent(self, z, m):
    VACE_CONV_CHANNELS = 96
    result = []
    for zz, mm in zip(z, m):
        if zz.shape[0] >= VACE_CONV_CHANNELS:
            result.append(zz[:VACE_CONV_CHANNELS])
        else:
            needed = VACE_CONV_CHANNELS - zz.shape[0]
            result.append(torch.cat([zz, mm[:needed]], dim=0))
    return result
```

> **Warning:** `git pull` on WanVideoWrapper will overwrite these patches. Re-apply after any update.

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `Value not in list` for text encoder | ComfyUI doesn't expose `.gguf` in text_encoders folder | Use `.safetensors` T5 encoder instead |
| `Expected size 30 but got size 60` | Unpatched `vace_encode_masks` | Apply Patch 1 + 2 above |
| `expected input to have 96 channels, got 352` | Unpatched `vace_latent` | Apply Patch 3 above |
| OOM during sampling | 10.8GB model too large for VRAM headroom | Increase `blocks_to_swap` in node 9 |
| `tensor 16 must match 48` at decode | Wrong VAE loaded (VAE38 used with A14B) | Use `Wan2_1_VAE_bf16.safetensors`, not `wan2.2_vae.safetensors` |
| Workflow timeout | Sampling is slow with block swap on 12GB GPU | Normal — A14B takes ~21 min on RTX 3060 |

---

## Full Error History

See [test_report.md](test_report.md) for the complete log of all 7 errors encountered during initial setup and their fixes.
