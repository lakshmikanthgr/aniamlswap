# User Guide — WAN2.2 VACE Video Editing

## What This Does

Takes any source video and edits it based on a text prompt — adding objects, changing appearance, swapping styles — while preserving the original motion, lighting, and scene structure. The only thing you change between runs is the prompt and the source video.

**What you can do with a prompt:**
- Add a prop to a subject ("person holding a red umbrella")
- Change what a subject is wearing ("same person but in a winter jacket")
- Add a background element ("a cat sitting in the corner of the room")
- Change the style ("same scene but at night with neon lighting")
- Remove or replace objects ("person without the hat")

The workflow is not specific to any subject or prop. Whatever you describe in the positive prompt, the model will attempt to generate it while keeping the source motion intact.

---

## Prerequisites

| Component | Requirement |
|---|---|
| GPU | RTX 3060 12GB minimum (block swap required). 24GB+ recommended for speed. |
| ComfyUI | Installed with Python venv |
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

```bash
cd ~/projects/ComfyUI/custom_nodes

# WanVideoWrapper (main inference)
git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git
cd ComfyUI-WanVideoWrapper && pip install -r requirements.txt && cd ..

# VideoHelperSuite (load/save video)
git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
cd ComfyUI-VideoHelperSuite && pip install -r requirements.txt && cd ..
```

> **IMPORTANT:** After cloning WanVideoWrapper, apply the patches in [Patches Required](#patches-required). The unpatched version has bugs with the VACE Fun A14B model.

---

## Quick Start

### 1. Start ComfyUI

```bash
cd ~/projects/ComfyUI
source venv/bin/activate
python main.py --lowvram
```

### 2. Place your source video

```bash
cp /path/to/your/video.mp4 ~/projects/ComfyUI/input/source.mp4
```

Recommended: 832×480, 16 fps, ≤49 frames (~3 seconds). Other sizes work but affect VRAM.

### 3. Edit the workflow

Open `workflows/vace_prop_addition.json` and update these three things:

**Your video filename (node 1):**
```json
"video": "source.mp4"
```

**Video dimensions and frame count (node 3) — must match the actual video:**
```json
"width": 832,
"height": 480,
"num_frames": 49
```

**Your prompt (node 4) — this is the only creative input:**
```json
"positive_prompt": "describe the full scene with the change you want",
"negative_prompt": "blurry, distorted, floating objects, flickering, inconsistent motion"
```

### 4. Run

```bash
cd ~/projects/video_swap
python3 scripts/run_workflow.py
```

Output: `~/projects/ComfyUI/output/vace_output_00001.mp4`

Expected runtime on RTX 3060 12GB: ~21 minutes (94s/step × 20 steps).

---

## Writing Prompts

The positive prompt drives everything. Describe the **full scene** as you want it to appear — not just the change. The model uses this description to regenerate the video while the VACE conditioning keeps the original motion intact.

### Structure that works well

```
[subject description], [what changed or added], [motion/pose context], [style/quality terms]
```

### Examples across different use cases

**Adding a held object:**
```
positive: a person walking down the street, holding a large red balloon, 
          same natural walk cycle, photorealistic, consistent lighting
negative: blurry, floating objects, disconnected prop, flickering
```

**Changing clothing:**
```
positive: same person, now wearing a bright yellow raincoat and boots,
          identical movement and pose, photorealistic, consistent with original scene
negative: morphing, inconsistent texture, flickering outfit
```

**Adding a background element:**
```
positive: same scene, a small dog sitting near the doorway in the background,
          natural lighting, photorealistic, consistent motion in foreground
negative: distorted, extra people, flickering, inconsistent background
```

**Style change:**
```
positive: same scene rendered in a hand-drawn animation style, 
          warm colors, consistent motion
negative: photorealistic, blurry, flickering
```

### Tips

- **Always describe the original subject** in the positive prompt, not just the new element. The model needs to know what to keep.
- **Mirror wording between positive and negative** — if you said "guitar" in the positive, don't include "deformed guitar" in the negative unless needed. Only negate things that commonly appear as artifacts.
- **Keep the negative prompt generic** — `blurry, distorted, flickering, inconsistent motion` works for almost any use case without over-constraining generation.
- **Strength (node 3)** controls how tightly the output follows the source video:
  - `0.85–1.0` — strong motion preservation, less creative freedom
  - `0.5–0.8` — looser, model can reinterpret the scene more
  - `<0.5` — rarely useful; starts to ignore source motion

---

## Using a Mask (Optional)

By default, VACE can edit the entire frame. A mask restricts where changes happen:

- White `(1.0)` = the model can edit this region
- Black `(0.0)` = preserve the original pixels here

Connect a `MASK` tensor to node 3's `input_masks` input. You can:
- Draw a mask in ComfyUI's built-in mask editor
- Use SAM2 (`ComfyUI-segment-anything-2`) to auto-segment a subject by clicking on it
- Use any other ComfyUI mask node

**Example:** To add an object to a person's hand without changing the background, mask only the hand region.

---

## Using a Reference Image (Optional)

Connect an image to node 3's `ref_images` input. The model uses it as a visual reference for a specific object's appearance.

Useful when you want the added prop to match a specific design (e.g., a branded product, a specific book cover, a logo on a t-shirt) rather than letting the model invent its own version.

---

## GPU / VRAM Settings

**Node 9 — WanVideoBlockSwap** offloads transformer blocks to CPU to fit the model in VRAM:

| GPU VRAM | `blocks_to_swap` | `vace_blocks_to_swap` | Speed |
|---|---|---|---|
| 12 GB | 30 | 8 | ~94s/step |
| 16 GB | 20 | 5 | ~40s/step |
| 24 GB+ | 0 | 0 | ~8s/step |

A14B has 40 transformer blocks and 15 VACE blocks total. Higher swap = lower VRAM, slower speed.

---

## Workflow Node Reference

| Node | Class | What to change |
|---|---|---|
| 1 | VHS_LoadVideo | `video` filename, `custom_width`, `custom_height`, `frame_load_cap` |
| 2 | WanVideoVAELoader | Leave as-is (`Wan2_1_VAE_bf16.safetensors`) |
| 3 | WanVideoVACEEncode | `width`, `height`, `num_frames`, `strength` |
| 4 | WanVideoTextEncodeCached | **`positive_prompt`** and `negative_prompt` |
| 5 | WanVideoModelLoader | Leave as-is |
| 6 | WanVideoSampler | `steps`, `cfg`, `seed` if needed |
| 7 | WanVideoDecode | Leave as-is |
| 8 | VHS_VideoCombine | `filename_prefix` if you want named outputs |
| 9 | WanVideoBlockSwap | `blocks_to_swap`, `vace_blocks_to_swap` per GPU |

See [architecture.md](architecture.md) for full input/output types of every node.

---

## Patches Required

The stock `ComfyUI-WanVideoWrapper` has three bugs with the VACE Fun A14B model. Patch `nodes.py` manually after cloning.

File: `~/projects/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper/nodes.py`

### Patch 1 — `WanVideoVACEEncode.process`

Hardcodes spatial stride and z_dim instead of reading from the loaded VAE.

```python
# Replace hardcoded values with:
_spatial_stride = getattr(vae, 'upsampling_factor', VAE_STRIDE[1])
_z_dim = getattr(vae, 'z_dim', 16)
target_shape = (_z_dim, (num_frames - 1) // VAE_STRIDE[0] + 1,
                height // _spatial_stride,
                width // _spatial_stride)

# And when calling vace_encode_masks:
spatial_stride = getattr(vae, 'upsampling_factor', VAE_STRIDE[1])
m0 = self.vace_encode_masks(input_masks, ref_images, spatial_stride=spatial_stride)
```

### Patch 2 — `vace_encode_masks`

Hardcodes spatial stride = 8, breaking mask shape for VAE38.

```python
def vace_encode_masks(self, masks, ref_images=None, spatial_stride=None):
    if spatial_stride is None:
        spatial_stride = VAE_STRIDE[1]
    # use spatial_stride in place of VAE_STRIDE[1] throughout the method
```

### Patch 3 — `vace_latent`

Always concatenates z + mask → wrong channel count for VAE38.

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

> **Note:** `git pull` on WanVideoWrapper overwrites these patches. Re-apply after any update.

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `Value not in list` for text encoder | `.gguf` files not exposed in text_encoders | Use `.safetensors` T5 encoder |
| `Expected size 30 but got size 60` | Unpatched `vace_encode_masks` | Apply Patch 1 + 2 |
| `expected 96 channels, got 352` | Unpatched `vace_latent` | Apply Patch 3 |
| OOM during sampling | Model too large for VRAM | Increase `blocks_to_swap` in node 9 |
| `tensor 16 must match 48` at decode | Wrong VAE — `wan2.2_vae.safetensors` is for 5B model only | Use `Wan2_1_VAE_bf16.safetensors` |
| Very slow (~94s/step) | Block swapping to CPU | Normal on 12GB GPU — 24GB removes this overhead |
