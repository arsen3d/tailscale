# Z-Image Turbo — DiT in your browser (ONNX Runtime Web)

A self-contained, static web demo that runs a **DiT (Diffusion Transformer)** text-to-image
pipeline entirely **in the browser** using
[ONNX Runtime Web](https://onnxruntime.ai/docs/get-started/with-javascript/web.html)
(WebGPU when available, WebAssembly otherwise).

It's inspired by Intel's blog post
[**From U-Net to DiT: Z-Image Turbo Runs in Your Browser**](https://community.intel.com/t5/Blogs/Tech-Innovation/Artificial-Intelligence-AI/From-U-Net-to-DiT-Z-Image-Turbo-Runs-in-Your-Browser/post/1743862),
and uses [`onnxruntime-web`](https://github.com/microsoft/onnxruntime/tree/main/js/web)
for in-browser inference.

> ⚠️ This is a **pipeline + UI scaffold**. It ships no model weights — you point it at your
> own ONNX exports of the text encoder, DiT denoiser, and VAE decoder. The full sampling loop,
> flow-matching scheduler, tokenizer, WebGPU/WASM selection, and rendering are all implemented.

---

## From U-Net to DiT (why this matters)

Classic Stable Diffusion used a **U-Net** denoiser with a DDPM noise schedule. Newer models
like **Z-Image Turbo** (Alibaba TongYi) replace the U-Net with a **Diffusion Transformer (DiT)**:

| | Classic SD (U-Net) | Z-Image Turbo (DiT) |
|---|---|---|
| Denoiser | Convolutional U-Net | Transformer ("S3-DiT", single-stream) |
| Objective | DDPM ε-prediction | **Flow matching** (velocity / rectified flow) |
| Steps | 20–50 | **~8** (distilled "Turbo") |
| Scaling | Harder | Parameter-efficient (6B ≈ 12B-class quality) |

This demo therefore uses a **flow-matching Euler scheduler** (straight-line path from
noise→image) and defaults to **8 steps**, matching the Turbo distillation.

## Pipeline

```
prompt ──▶ tokenizer ──▶ [text encoder.onnx] ──▶ conditioning
                                                      │
   noise ──▶ ┌─────────── × N steps ───────────┐     │
             │  [transformer.onnx]  (DiT)  ◀────┼─────┘
             │  x ← x + dt · v   (Euler)        │
             └─────────────────────────────────┘
                          │ latent
                          ▼
                   [vae_decoder.onnx] ──▶ RGB image ──▶ <canvas>
```

Each stage is a separate `ort.InferenceSession`. See [`pipeline.js`](./pipeline.js).

## Run it

It's a static site — serve the folder over HTTP (ES modules + WASM won't load from `file://`):

```bash
cd demo/z-image-turbo-web
python3 -m http.server 8000
# open http://localhost:8000
```

WebGPU requires a recent Chromium-based browser (Chrome/Edge 113+). Without it the demo
automatically falls back to the WebAssembly CPU backend (slower but works everywhere).

## Providing models

In the **Models** panel, paste URLs to three ONNX files. They can be local (served from this
same folder) or remote (CORS must allow them).

> ⚠️ There is **no** `optimum-cli export onnx --task text-to-image` config for Z-Image — the
> DiT's `forward` takes Python lists of variable-length tensors, so it doesn't trace as a
> single static graph. You export each component manually.

A ready-to-edit export script and a full walkthrough are included:

```bash
pip install "diffusers>=0.38.0" transformers accelerate torch onnx onnxruntime
python export_z_image_onnx.py --model Tongyi-MAI/Z-Image-Turbo \
  --out ./onnx --height 512 --width 512 --seq 512 --opset 18
```

See **[EXPORT.md](./EXPORT.md)** for the architecture details, per-component steps, the 2 GB
external-data caveat, quantization, and validation. It produces `text_encoder.onnx`,
`transformer.onnx`, and `vae_decoder.onnx` (plus `*.onnx_data` sidecars for the big ones) —
point the three fields at those.

### Knobs that must match your export

| Field | Meaning | Typical |
|---|---|---|
| Tokenizer | HF repo id for the text encoder's tokenizer | model-specific (e.g. a Qwen tokenizer) |
| Latent channels | DiT/VAE latent channel count | `16` |
| VAE scale | spatial downscale factor of the VAE | `8` |
| VAE scaling | latent scaling factor before decode | model-specific |

The DiT is fed inputs **by position**: `(latent, timestep, conditioning)`, with the timestep
passed as `sigma × 1000`. If your export uses different input names/order or a different
timestep convention, adjust `generate()` in [`pipeline.js`](./pipeline.js).

## Files

- `index.html` — UI
- `app.js` — UI wiring / event handlers
- `pipeline.js` — ORT Web sessions, tokenizer, flow-matching sampler, VAE decode
- `styles.css` — styling
- `export_z_image_onnx.py` — exports the 3 components to ONNX
- `EXPORT.md` — full guide to building the model for ONNX

## Notes & limitations

- Running a multi-billion-parameter DiT in a browser tab is memory- and compute-heavy;
  start with small resolutions (e.g. 256–512px) and few steps.
- WebGPU support for some operators is still maturing; if a model fails on WebGPU, retry
  with the WASM backend.
- This demo prioritizes clarity over raw speed (e.g. the Euler step runs on the CPU between
  sessions). For production, fuse the scheduler step into the graph or use typed-array views.

## Credits

- Intel — *From U-Net to DiT: Z-Image Turbo Runs in Your Browser*
- Microsoft — [ONNX Runtime Web](https://github.com/microsoft/onnxruntime/tree/main/js/web)
- Alibaba TongYi — Z-Image / Z-Image Turbo
- Hugging Face — `@huggingface/transformers` (in-browser tokenizer)
