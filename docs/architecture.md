# Architecture — WAN2.2 VACE Prop-Addition Pipeline

## Overview

The pipeline takes a source video and a text prompt, then produces an edited video that follows the original motion while reflecting whatever the prompt describes — added objects, changed clothing, style shifts, background elements, or any other visual edit. The prompt is the only creative input; everything else is configuration. It runs entirely inside ComfyUI using the WanVideoWrapper custom node set.

```
Source Video (MP4)
       │
       ▼
┌─────────────┐     ┌──────────────┐
│ VHS_LoadVideo│     │WanVideoVAELoader│
│  (node 1)   │     │   (node 2)   │
└──────┬──────┘     └──────┬───────┘
       │                   │
       ▼                   ▼
┌────────────────────────────────────┐
│         WanVideoVACEEncode         │
│              (node 3)              │
│  • VAE-encodes source frames       │
│  • Splits into inactive/reactive   │
│    latent pair (2×z_dim channels)  │
│  • Downsamples mask to latent res  │
│  • Concatenates → 96-ch context    │
└──────────────────┬─────────────────┘
                   │ WANVIDIMAGE_EMBEDS
                   │ (vace_context, target_shape, …)
                   ▼
┌──────────────────────────────────────┐
│       WanVideoSampler (node 6)       │◄── WanVideoModelLoader (node 5)
│  • Creates noise at target_shape     │◄── WanVideoBlockSwap  (node 9)
│  • Injects vace_context each step    │◄── WanVideoTextEncodeCached (node 4)
│  • Runs 20-step euler/beta denoise   │
└──────────────────┬───────────────────┘
                   │ latent samples
                   ▼
┌──────────────────────────────────────┐
│        WanVideoDecode (node 7)       │
│  • VAE-decodes latents → RGB frames  │
└──────────────────┬───────────────────┘
                   │ IMAGE
                   ▼
┌──────────────────────────────────────┐
│       VHS_VideoCombine (node 8)      │
│  • Encodes frames → h264 MP4         │
└──────────────────────────────────────┘
                   │
                   ▼
             Output MP4
```

---

## Node-by-Node Breakdown

### Node 1 — VHS_LoadVideo
**Type:** VideoHelperSuite  
**Role:** Loads source video frames as an IMAGE tensor.

| Input | Value | Notes |
|---|---|---|
| video | source.mp4 | Looked up in `ComfyUI/input/` — replace with your filename |
| force_rate | 16 | Resamples to 16 fps |
| custom_width / height | 832 × 480 | Resizes on load |
| frame_load_cap | 49 | Max frames to load |

**Output:** `IMAGE` tensor — shape `(N_frames, H, W, 3)`, float32, range [0,1]

---

### Node 2 — WanVideoVAELoader
**Type:** WanVideoWrapper  
**Role:** Loads the WAN VAE weights and auto-detects architecture from checkpoint.

| Input | Value |
|---|---|
| model_name | Wan2_1_VAE_bf16.safetensors |
| precision | bf16 |

**Output:** `WANVAE` object — either `WanVideoVAE` (16-ch, 8× spatial) or `WanVideoVAE38` (48-ch, 16× spatial), detected from `model.conv2.weight.shape[0]`.

> **Important for this project:** The VACE Fun A14B model has `out_dim=16`, so it uses the 16-channel `Wan2_1_VAE_bf16.safetensors`. The newer `wan2.2_vae.safetensors` is 48-channel and only works with the 5B model.

---

### Node 3 — WanVideoVACEEncode
**Type:** WanVideoWrapper  
**Role:** Core VACE conditioning encoder. Produces the context that tells the model what the original video looks like and where to make changes.

**Inputs:**

| Input | Type | Value | Purpose |
|---|---|---|---|
| vae | WANVAE | from node 2 | Encoder to use |
| width / height | INT | 832 × 480 | Must match node 1 |
| num_frames | INT | 49 | Must match node 1 |
| strength | FLOAT | 0.85 | VACE conditioning scale |
| vace_start_percent | FLOAT | 0.0 | Apply VACE from step 0% |
| vace_end_percent | FLOAT | 1.0 | Apply VACE until step 100% |
| input_frames | IMAGE | from node 1 | Source video to condition on |
| input_masks | MASK | (none) | Where to edit — full-frame white mask if absent |
| ref_images | IMAGE | (none) | Optional static prop reference image |

**Internal processing:**

```
input_frames
    │
    ├─► inactive latent  ─┐
    │   (frames where      │
    │    mask = 0)         ├─► z0 = cat([inactive, reactive], dim=0)
    │                      │   shape: (2 × z_dim, T_lat, H_lat, W_lat)
    └─► reactive latent  ─┘
        (original frames)

input_masks
    └─► downsampled mask  ─► m0  shape: (spatial_stride², T_lat, H_lat, W_lat)

vace_latent:
    if z0.channels >= 96:  → context = z0[:96]       # VAE38 path (unused here)
    else:                  → context = cat(z0, m0[:96-z0.ch])  # VAE16 path: 32+64=96
```

**Output:** `WANVIDIMAGE_EMBEDS` dict:
```python
{
    "vace_context":       [tensor(96, T_lat, H_lat, W_lat)],
    "vace_scale":         0.85,
    "target_shape":       (16, 13, 60, 104),   # (z_dim, T_lat, H_lat, W_lat)
    "vace_start_percent": 0.0,
    "vace_end_percent":   1.0,
    "vace_seq_len":       20280,
    "has_ref":            False,
    "num_frames":         49,
    "additional_vace_inputs": []
}
```

> `target_shape` is what the sampler uses to create the initial noise tensor. For VAE16: `(16, (49-1)//4+1, 480//8, 832//8)` = `(16, 13, 60, 104)`.

---

### Node 4 — WanVideoTextEncodeCached
**Type:** WanVideoWrapper  
**Role:** Encodes positive and negative text prompts using the UMT5-XXL text encoder.

| Input | Value |
|---|---|
| model_name | umt5-xxl-enc-fp8_e4m3fn.safetensors |
| positive_prompt | Your scene description with the desired change |
| negative_prompt | "blurry, distorted…" |
| use_disk_cache | true |

**Output:** `WANVIDTEXT_EMBEDS` — cached T5 embeddings for positive and negative prompts.

> Uses `WanVideoTextEncodeCached` (not `WanVideoTextEncode`). The cached variant auto-loads T5 internally; the non-cached variant requires a separate T5 loader node and fails without it.

---

### Node 9 — WanVideoBlockSwap
**Type:** WanVideoWrapper  
**Role:** Configuration for offloading transformer blocks to CPU during sampling. Required on GPUs with <24GB VRAM.

| Input | Value | Notes |
|---|---|---|
| blocks_to_swap | 30 | Out of 40 total transformer blocks in A14B |
| vace_blocks_to_swap | 8 | Out of 15 VACE blocks |
| offload_img_emb | true | Offload image embeddings to CPU |
| offload_txt_emb | true | Offload text embeddings to CPU |

**Output:** `BLOCKSWAPARGS` — passed into `WanVideoModelLoader` as `block_swap_args`.

> Setting `blocks_to_swap=30` reduces peak VRAM from ~20GB to ~10GB at the cost of ~94s/step (vs ~8s/step on a 24GB GPU).

---

### Node 5 — WanVideoModelLoader
**Type:** WanVideoWrapper  
**Role:** Loads the main diffusion transformer (GGUF quantized).

| Input | Value |
|---|---|
| model | LowNoise/Wan2.2-VACE-Fun-A14B-low-noise-Q4_K_M.gguf |
| base_precision | bf16 |
| load_device | offload_device (CPU) |
| attention_mode | sdpa |
| block_swap_args | from node 9 |

**Output:** `WANVIDEOMODEL` — transformer ready for sampling, with block swap configured.

---

### Node 6 — WanVideoSampler
**Type:** WanVideoWrapper  
**Role:** Runs the denoising loop. Injects VACE context at each step.

| Input | Source | Value |
|---|---|---|
| model | node 5 | Transformer |
| image_embeds | node 3 | VACE context + target_shape |
| text_embeds | node 4 | T5 embeddings |
| steps | — | 20 |
| cfg | — | 6.0 |
| shift | — | 8.0 |
| scheduler | — | euler/beta |
| denoise_strength | — | 1.0 |
| force_offload | — | true |

**Internal logic:**

1. Reads `target_shape` from `image_embeds` → creates noise tensor `(1, 16, 13, 60, 104)`
2. Checks `is_5b = transformer.out_dim == 48` → False for A14B → uses 16-ch noise
3. At each denoising step, injects `vace_context` into the VACE cross-attention layers
4. Returns denoised latent

**Output:** `LATENT` dict: `{"samples": tensor(1, 16, 13, 60, 104)}`

---

### Node 7 — WanVideoDecode
**Type:** WanVideoWrapper  
**Role:** Decodes latent tensor back to RGB frames using the VAE decoder.

| Input | Source |
|---|---|
| vae | node 2 |
| samples | node 6 |
| enable_vae_tiling | false |

**Output:** `IMAGE` tensor — `(49, 480, 832, 3)`, float32, range [0,1]

---

### Node 8 — VHS_VideoCombine
**Type:** VideoHelperSuite  
**Role:** Encodes frames to h264 MP4.

| Input | Value |
|---|---|
| frame_rate | 16 |
| format | video/h264-mp4 |
| filename_prefix | vace_output |

**Output:** MP4 saved to `ComfyUI/output/vace_output_00001.mp4`

---

## Latent Space Dimensions (VAE16, 832×480, 49 frames)

| Stage | Tensor shape | Notes |
|---|---|---|
| Input frames | (49, 480, 832, 3) | H W C, float [0,1] |
| After VAE encode | (1, 16, 13, 60, 104) | B C T H W |
| VACE context | (96, 13, 60, 104) | 2×z_dim + mask channels |
| Sampler noise/output | (1, 16, 13, 60, 104) | same as encoded |
| After VAE decode | (49, 480, 832, 3) | back to pixels |

Temporal stride: 4 (`VAE_STRIDE[0]`), so 49 frames → 13 latent frames `((49-1)//4 + 1)`.  
Spatial stride: 8 (`upsampling_factor`), so 480→60, 832→104.

---

## VACE Channel Math

The VACE conditioning conv has fixed weight shape `[5120, 96, 1, 2, 2]` — always expects 96 input channels.

| VAE type | z_dim | inactive+reactive | mask channels | total |
|---|---|---|---|---|
| VAE16 (A14B) | 16 | 2×16 = 32 | 8×8 = 64 | **96** ✓ |
| VAE38 (5B) | 48 | 2×48 = 96 | — (not needed) | **96** ✓ |

---

## File Locations

| File | Path |
|---|---|
| Main model | `~/projects/ComfyUI/models/unet/LowNoise/Wan2.2-VACE-Fun-A14B-low-noise-Q4_K_M.gguf` |
| VAE | `~/projects/ComfyUI/models/vae/Wan2_1_VAE_bf16.safetensors` |
| Text encoder | `~/projects/ComfyUI/models/text_encoders/umt5-xxl-enc-fp8_e4m3fn.safetensors` |
| Workflow JSON | `workflows/vace_prop_addition.json` |
| Patched nodes | `~/projects/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper/nodes.py` |
| Input video | `~/projects/ComfyUI/input/source.mp4` (rename to match your file) |
| Output video | `~/projects/ComfyUI/output/vace_output_00001.mp4` |
