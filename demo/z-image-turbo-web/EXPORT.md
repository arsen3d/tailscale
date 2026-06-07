# Building Z-Image Turbo for ONNX

How to convert [`Tongyi-MAI/Z-Image-Turbo`](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
into the three ONNX graphs this demo loads. Read this fully before you start — the
honest summary is **"feasible but involved; the DiT in a browser is the hard part."**

## TL;DR

There is **no `optimum-cli export onnx` config for Z-Image**. You export each component
yourself with `torch.onnx`. A ready-to-edit script is provided: [`export_z_image_onnx.py`](./export_z_image_onnx.py).

```bash
pip install "diffusers>=0.38.0" transformers accelerate torch onnx onnxruntime
python export_z_image_onnx.py \
  --model Tongyi-MAI/Z-Image-Turbo \
  --out ./onnx --height 512 --width 512 --seq 512 --opset 18
# -> ./onnx/text_encoder.onnx, transformer.onnx, vae_decoder.onnx (+ *.onnx_data)
```

Then point the demo's three URL fields at those files.

## The architecture you're exporting

Z-Image-Turbo is a **6B Single-Stream DiT (S3-DiT)**, not a U-Net. In diffusers it's a
`ZImagePipeline` with these components:

| Component | Class | Notes |
|---|---|---|
| `text_encoder` | Qwen3-4B (`PreTrainedModel`) | conditioning = **penultimate** hidden state (`hidden_states[-2]`), `cap_feat_dim=2560` |
| `transformer` | `ZImageTransformer2DModel` | `in_channels=16`, `dim=3840`, `n_layers=30`, `n_heads=30`, `patch_size=2`, `t_scale=1000` |
| `vae` | Flux `AutoencoderKL` | 16 latent channels, spatial scale 8 |
| `scheduler` | `FlowMatchEulerDiscreteScheduler` | flow matching; Turbo uses **~8 steps, `guidance_scale=0.0`** (CFG-free) |

## Why it isn't a one-liner

1. **The DiT takes Python lists, not tensors.** Its real signature is
   `forward(x: list[Tensor], t, cap_feats: list[Tensor], ...)` — the single-stream design
   concatenates variable-length text + image tokens and builds RoPE positions internally.
   ONNX can't trace a `list` input directly, so you wrap it to take a fixed-shape
   `[1, 16, h, w]` latent and `[1, T, 2560]` caption tensor for **one resolution at a time**.
   (Change resolution → re-export, or invest in dynamic-axes + the dynamo exporter.)
2. **Two components blow past ONNX's 2 GB protobuf limit** (6B DiT, 4B text encoder), so they
   must be saved with **external data** (`*.onnx_data` sidecars). You must serve every sidecar
   next to its `.onnx`.
3. **The text encoder is a 4B LLM.** Z-Image applies a **chat template with `enable_thinking=True`**
   before tokenizing, then takes `hidden_states[-2]`. To match outputs you must reproduce that
   templating on the JS side — or precompute embeddings (see below).

## Recommended path (easiest → hardest)

### 1. VAE decoder — do this first
The most browser-friendly piece (~few hundred MB). The script folds
`latent/scaling + shift` into the graph so JS only feeds raw latents.

```bash
python export_z_image_onnx.py --only vae --out ./onnx
```

### 2. Text encoder — prefer existing tooling
Qwen3 is a standard transformers model. Instead of hand-exporting, you can use
[🤗 Optimum / Transformers.js](https://huggingface.co/docs/transformers.js) to get a
quantized ONNX Qwen3 and run it client-side, then slice `hidden_states[-2]`. The script's
`--only text` path is a manual fallback. Either way, **quantize** (int8/int4) — 4B in fp16 is ~8 GB.

```bash
python export_z_image_onnx.py --only text --out ./onnx
```

> Shortcut: skip the in-browser text encoder entirely. Precompute `prompt_embeds` in Python,
> save them as a tensor, and load them in JS — useful for fixed prompts or a gallery demo.

### 3. The DiT transformer — the hard one
```bash
python export_z_image_onnx.py --only dit --height 512 --width 512 --seq 512
```
Edit the `DiTWrapper` in the script if the traced element shapes don't match your diffusers
version (the wrapper assumes batch=1, single image, list length 1). If tracing fails on the
list/RoPE logic, enable the **TorchDynamo exporter** (`dynamo=True` in `torch.onnx.export`),
which handles control flow and containers far better than the legacy tracer.

## Make it actually fit in a browser

A raw fp32 6B graph is ~24 GB; fp16 ~12 GB. You need quantization:

```python
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic("onnx/transformer.onnx", "onnx/transformer.int8.onnx",
                 weight_type=QuantType.QInt8)
```

- Use **int8 (or int4)** weights; consider [MatMulNBits](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html) 4-bit for the DiT/text encoder.
- Keep resolution small (256–512px) to bound the latent/attention size.
- Prefer the **WebGPU** EP; verify every operator is supported there (some fall back to WASM).
- Browser memory is finite — even int4, a 6B model is ~3 GB of weights to download and hold.
  This is why most "in-browser" DiT demos use distilled/quantized models and modest resolutions.

## Custom characters with LoRA

Diffusers supports Z-Image LoRA ([PR #12750](https://github.com/huggingface/diffusers/pull/12750)):
`load_lora_weights` / `set_adapters` / `fuse_lora`. But **ONNX is a frozen graph** — there is no
runtime adapter in ORT Web. You **bake the LoRA into the weights before export**:

```bash
python export_z_image_onnx.py --only dit \
  --lora <hf-repo-or-path> --lora-scale 0.9 --out ./onnx-mychar
# stack several: --lora "repoA,repoB"
```

The script calls `load_lora_weights → set_adapters → fuse_lora` before tracing, so the resulting
`transformer.onnx` *is* your custom character. Consequences:

- **One fused LoRA = one ONNX file.** No runtime switching; at ~6B that's a multi-GB download per
  character. Good for a small fixed cast, not an open LoRA library. (Dynamic LoRA swapping only
  works server-side in PyTorch.)
- **Order matters:** fuse **before** quantizing. `LoRA → fuse → export → quantize`.
- **Fused-QKV trap:** Z-Image stores attention as a single fused QKV matrix, but many community
  LoRAs ship separate `to_q`/`to_k`/`to_v`. Loaded without conversion they **silently don't apply**
  (you'll get the base model). Use a recent diffusers (PR #12750+) or a fused-QKV-aware loader, and
  verify the LoRA keys actually matched before exporting.
- If the LoRA also trains the text encoder, re-export `--only text` too.

## Validate before shipping

After export, sanity-check each graph in Python with `onnxruntime` (CPU) against the original
PyTorch module on the same inputs (allclose within fp tolerance). Then wire the URLs into the
demo. If outputs look like noise, the usual culprits are: wrong VAE `scaling/shift`, wrong
timestep scale (should be `sigma * t_scale`, `t_scale=1000`), or a text-embedding mismatch
(chat template / wrong hidden-state layer).

## References

- Z-Image paper — [arXiv:2511.22699](https://arxiv.org/abs/2511.22699)
- Diffusers Z-Image pipeline — https://huggingface.co/docs/diffusers/api/pipelines/z_image
- ONNX Runtime Web — https://github.com/microsoft/onnxruntime/tree/main/js/web
- ONNX export with Optimum — https://huggingface.co/docs/optimum-onnx/onnx/usage_guides/export_a_model
- ORT quantization — https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
