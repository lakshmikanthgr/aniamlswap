# VACE Prop-Addition Workflow Test Report

**Date:** 2026-06-30  
**Model:** Wan2.2-VACE-Fun-A14B-low-noise-Q4_K_M.gguf (10.8GB GGUF)  
**Hardware:** RTX 3060 12GB  
**ComfyUI:** ~/projects/ComfyUI  

## Result: SUCCESS

Output file: `/home/laks/projects/ComfyUI/output/birds_prop_test_00001.mp4`  
Format: h264, 832×480, 49 frames, 16fps (~3s)

---

## Workflow Summary

**Nodes (vace_prop_addition.json):**

| ID | Node | Key Config |
|----|------|------------|
| 1 | VHS_LoadVideo | birds_source.mp4, 832×480, 49 frames |
| 2 | WanVideoVAELoader | Wan2_1_VAE_bf16.safetensors, bf16 |
| 3 | WanVideoVACEEncode | strength=0.85, full-frame mask |
| 4 | WanVideoTextEncodeCached | umt5-xxl-enc-fp8_e4m3fn.safetensors |
| 5 | WanVideoModelLoader | VACE Fun A14B GGUF, offload_device, sdpa |
| 6 | WanVideoSampler | 20 steps, cfg=6, shift=8, euler/beta |
| 7 | WanVideoDecode | tiling disabled |
| 8 | VHS_VideoCombine | h264-mp4, 16fps |
| 9 | WanVideoBlockSwap | blocks_to_swap=30, vace_blocks_to_swap=8 |

**Total sampling time:** ~1230s (~20.5 min), ~94s/step  
**Total wall time (encoding + sampling + decode):** ~1280s

---

## Errors Encountered and Fixes

### Error 1 — WanVideoDecode: missing tile_stride params
**Node:** 7 (WanVideoDecode)  
**Message:** HTTP 400, required inputs `tile_stride_x` / `tile_stride_y` missing  
**Fix:** Added `"tile_stride_x": 144, "tile_stride_y": 128` to node 7 inputs

### Error 2 — WanVideoTextEncode: no cached text embeds
**Node:** 4  
**Message:** "No cached text embeds found"  
**Fix:** Changed `class_type` from `WanVideoTextEncode` to `WanVideoTextEncodeCached`

### Error 3 — Text encoder GGUF not in list
**Node:** 4  
**Message:** "Value not in list — umt5-xxl-encoder-Q5_K_M.gguf"  
**Root cause:** ComfyUI's `text_encoders` folder type only exposes `.safetensors`, not `.gguf`  
**Fix:** Switched to `umt5-xxl-enc-fp8_e4m3fn.safetensors`

### Error 4 — WanVideoVACEEncode: tensor size mismatch (30 vs 60)
**Node:** 3  
**Message:** "Sizes of tensors must match except in dimension 0. Expected size 30 but got size 60"  
**Root cause:** `vace_encode_masks` in `nodes.py` hardcoded `VAE_STRIDE[1]=8` for spatial downsampling. When loaded with `wan2.2_vae.safetensors` (VAE38, `upsampling_factor=16`), mask height = 480/8 = 60 but latent height = 480/16 = 30.  
**Fix:** Patched `nodes.py`:
- `WanVideoVACEEncode.process`: reads `spatial_stride = getattr(vae, 'upsampling_factor', VAE_STRIDE[1])` and `z_dim = getattr(vae, 'z_dim', 16)`
- `vace_encode_masks`: accepts `spatial_stride` param instead of hardcoding 8

### Error 5 — WanVideoSampler: OOM (out of memory)
**Node:** 6  
**Message:** "Allocation on device... ran out of memory on your GPU"  
**Root cause:** 10.8GB GGUF model exceeds 12GB VRAM headroom during attention  
**Fix:** Added `WanVideoBlockSwap` node (node 9) with `blocks_to_swap=30, vace_blocks_to_swap=8`, wired into `WanVideoModelLoader` as `block_swap_args`

### Error 6 — WanVideoSampler: channel mismatch (96 channels, got 352)
**Node:** 6  
**Message:** "Given groups=1, weight of size [5120, 96, 1, 2, 2], expected input to have 96 channels, but got 352"  
**Root cause:** VACE conv expects exactly 96 input channels. With VAE38 (z_dim=48), `vace_latent` was producing: z0(96) + mask(256) = 352. For VAE38, only z0 is needed.  
**Fix:** Patched `vace_latent` in `nodes.py`: clips to 96 if `z0.shape[0] >= 96`; otherwise appends mask channels to reach 96 (VAE16 path: 32+64=96)

### Error 7 — WanVideoDecode: tensor dimension mismatch (16 vs 48)
**Node:** 7  
**Message:** "The size of tensor a (16) must match the size of tensor b (48) at non-singleton dimension 1"  
**Root cause:** VAE chosen was `wan2.2_vae.safetensors` (VAE38, 48-channel decoder). The VACE Fun A14B model has `out_dim=16` (sampler check: `is_5b = transformer.out_dim == 48` → False), so it outputs 16-channel latents. VAE38 cannot decode them.  
**Fix:** Switched VAE to `Wan2_1_VAE_bf16.safetensors` (VAE16, 16-channel)

---

## Source Patches to ComfyUI-WanVideoWrapper

File: `~/projects/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper/nodes.py`

Three methods modified:
1. `WanVideoVACEEncode.process` — dynamic spatial_stride and z_dim from VAE attrs
2. `Wace_encode_masks` — `spatial_stride` param instead of hardcoded 8
3. `vace_latent` — clips to 96 channels for VAE38 vs appending mask for VAE16

**Warning:** `git pull` on WanVideoWrapper will overwrite these patches.

---

## Notes

- The VACE Fun A14B model uses the original 16-channel VAE (not the new 48-channel VAE38 which is for 5B models)
- Block swapping (30/40 transformer blocks + 8/15 VACE blocks) is required on RTX 3060 12GB for the A14B model
- Per-step time of ~94s reflects heavy CPU↔GPU block swapping; a GPU with 24GB+ VRAM would be significantly faster
- Input video was a synthetic green-screen placeholder; real bird footage would exercise the VACE conditioning more meaningfully
- Full-frame mask was used (no SAM2 segmentation); prop addition quality would improve with a targeted mask
