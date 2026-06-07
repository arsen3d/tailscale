#!/usr/bin/env python3
"""
export_z_image_onnx.py

Export the three runnable components of Z-Image-Turbo to ONNX for use with
ONNX Runtime Web:

    1. text_encoder.onnx  (Qwen3-4B -> penultimate hidden state)
    2. transformer.onnx   (ZImageTransformer2DModel, the S3-DiT denoiser)
    3. vae_decoder.onnx    (Flux AutoencoderKL decoder)

There is no `optimum-cli export onnx` config for Z-Image, so each component is
exported manually with thin wrappers that expose a *static, tensor-only* call
signature (the real modules take Python lists / nested lists, which ONNX cannot
trace directly).

IMPORTANT: shapes and call conventions are pinned to diffusers v0.38.0. If you
use a different version, re-check ZImageTransformer2DModel.forward and the
pipeline's _encode_prompt before trusting the wrappers below.

Usage:
    pip install "diffusers>=0.38.0" transformers accelerate torch onnx onnxruntime
    python export_z_image_onnx.py --model Tongyi-MAI/Z-Image-Turbo --out ./onnx \
        --height 512 --width 512 --seq 512 --opset 18

Notes:
  * The DiT (~6B) and text encoder (~4B) exceed the 2 GB protobuf limit, so they
    are saved with external data (a sibling *.onnx_data file). ORT Web can load
    these, but you must serve BOTH files from the same directory.
  * For the browser you will almost certainly need to quantize (int8/int4) and/or
    keep resolution small. fp16/fp32 6B graphs are multiple GB.
"""

import argparse
import os
import torch


def export_vae_decoder(pipe, out_dir, h, w, opset):
    """Most tractable component. Folds the scaling/shift into the graph."""
    vae = pipe.vae.eval()
    scaling = vae.config.scaling_factor
    shift = getattr(vae.config, "shift_factor", 0.0) or 0.0
    latent_ch = vae.config.latent_channels
    down = 2 ** (len(vae.config.block_out_channels) - 1)  # spatial downscale, usually 8

    class VaeDecoderWrapper(torch.nn.Module):
        def __init__(self, vae, scaling, shift):
            super().__init__()
            self.vae, self.scaling, self.shift = vae, scaling, shift

        def forward(self, latents):
            latents = latents / self.scaling + self.shift
            return self.vae.decode(latents, return_dict=False)[0]  # [B,3,H,W] in [-1,1]

    wrapper = VaeDecoderWrapper(vae, scaling, shift).eval()
    dummy = torch.randn(1, latent_ch, h // down, w // down)

    path = os.path.join(out_dir, "vae_decoder.onnx")
    torch.onnx.export(
        wrapper, (dummy,), path,
        input_names=["latent"], output_names=["image"],
        dynamic_axes={"latent": {0: "B", 2: "lh", 3: "lw"},
                      "image": {0: "B", 2: "H", 3: "W"}},
        opset_version=opset, do_constant_folding=True,
    )
    print(f"  wrote {path}  (latent_channels={latent_ch}, vae_scale={down}, "
          f"scaling={scaling}, shift={shift})")


def export_text_encoder(pipe, out_dir, seq, opset):
    """
    Qwen3-4B. The pipeline uses `.hidden_states[-2]` (penultimate layer) as the
    conditioning, after applying a chat template. We export the raw LM returning
    the penultimate hidden state; you must reproduce the SAME chat-template +
    tokenization on the JS side (or precompute embeddings).
    """
    te = pipe.text_encoder.eval()

    class TextEncoderWrapper(torch.nn.Module):
        def __init__(self, te):
            super().__init__()
            self.te = te

        def forward(self, input_ids, attention_mask):
            out = self.te(input_ids=input_ids, attention_mask=attention_mask,
                          output_hidden_states=True)
            return out.hidden_states[-2]  # [B, seq, cap_feat_dim]

    wrapper = TextEncoderWrapper(te).eval()
    ids = torch.ones(1, seq, dtype=torch.int64)
    mask = torch.ones(1, seq, dtype=torch.int64)

    path = os.path.join(out_dir, "text_encoder.onnx")
    torch.onnx.export(
        wrapper, (ids, mask), path,
        input_names=["input_ids", "attention_mask"],
        output_names=["prompt_embeds"],
        dynamic_axes={"input_ids": {0: "B", 1: "T"},
                      "attention_mask": {0: "B", 1: "T"},
                      "prompt_embeds": {0: "B", 1: "T"}},
        opset_version=opset, do_constant_folding=True,
    )
    print(f"  wrote {path}  (cap_feat_dim={te.config.hidden_size})")


def export_transformer(pipe, out_dir, h, w, seq, opset):
    """
    The hard one. ZImageTransformer2DModel.forward takes:
        x:         list[Tensor]   (image latent tokens, per batch item)
        t:         Tensor          (timestep, t_scale=1000)
        cap_feats: list[Tensor]   (text features, per batch item)
    and returns a list of tensors. We wrap it for batch=1, single image, fixed
    resolution so the traced graph has a static signature.

    The pipeline does `latent_model_input.unsqueeze(2).unbind(0)`, giving list
    elements of shape [C, 1, lh, lw]; cap_feats elements are [seq, cap_feat_dim].
    """
    tr = pipe.transformer.eval()
    cfg = tr.config
    in_ch = cfg.in_channels
    cap_dim = cfg.cap_feat_dim
    patch = cfg.all_patch_size[0]
    f_patch = cfg.all_f_patch_size[0]
    down = 2 ** (len(pipe.vae.config.block_out_channels) - 1)

    class DiTWrapper(torch.nn.Module):
        def __init__(self, tr, patch, f_patch):
            super().__init__()
            self.tr, self.patch, self.f_patch = tr, patch, f_patch

        def forward(self, latent, timestep, cap_feats):
            # latent: [1, C, lh, lw] -> [C, 1, lh, lw] (add frame dim, drop batch)
            x = [latent[0].unsqueeze(1)]
            cap = [cap_feats[0]]  # [seq, cap_dim]
            out = self.tr(x, timestep, cap, return_dict=False,
                          patch_size=self.patch, f_patch_size=self.f_patch)[0]
            return out[0].unsqueeze(0)  # back to [1, C, lh, lw]

    wrapper = DiTWrapper(tr, patch, f_patch).eval()
    lh, lw = h // down, w // down
    dummy_latent = torch.randn(1, in_ch, lh, lw)
    dummy_t = torch.tensor([1000.0])
    dummy_cap = torch.randn(1, seq, cap_dim)

    path = os.path.join(out_dir, "transformer.onnx")
    torch.onnx.export(
        wrapper, (dummy_latent, dummy_t, dummy_cap), path,
        input_names=["latent", "timestep", "cap_feats"],
        output_names=["velocity"],
        dynamic_axes={"latent": {2: "lh", 3: "lw"},
                      "cap_feats": {1: "T"},
                      "velocity": {2: "lh", 3: "lw"}},
        opset_version=opset, do_constant_folding=True,
        # The list/RoPE logic traces more reliably with the dynamo exporter:
        # dynamo=True,
    )
    print(f"  wrote {path}  (in_channels={in_ch}, cap_feat_dim={cap_dim}, "
          f"patch_size={patch}, t_scale={cfg.t_scale})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Tongyi-MAI/Z-Image-Turbo")
    ap.add_argument("--out", default="./onnx")
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--only", choices=["vae", "text", "dit"], default=None,
                    help="export only one component")
    ap.add_argument("--lora", default=None,
                    help="LoRA to bake in before export. HF repo id or local path/file. "
                         "Repeat-style stacking: pass a comma-separated list.")
    ap.add_argument("--lora-weight-name", default=None,
                    help="specific .safetensors file inside the LoRA repo, if needed")
    ap.add_argument("--lora-scale", type=float, default=1.0,
                    help="LoRA strength to fuse at (default 1.0)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    from diffusers import ZImagePipeline
    print(f"Loading {args.model} (fp32 on CPU; this needs lots of RAM)…")
    pipe = ZImagePipeline.from_pretrained(args.model, torch_dtype=torch.float32)

    # --- LoRA: bake the adapter into the weights BEFORE export ---------------
    # ONNX is a frozen graph: there is no runtime adapter. The only way to get a
    # custom character into the browser pipeline is to fuse the LoRA here, then
    # export (and only then quantize). Each fused LoRA -> its own ONNX file.
    if args.lora:
        loras = [s.strip() for s in args.lora.split(",") if s.strip()]
        names = []
        for i, lp in enumerate(loras):
            name = f"lora{i}"
            kw = {"adapter_name": name}
            if args.lora_weight_name and len(loras) == 1:
                kw["weight_name"] = args.lora_weight_name
            print(f"Loading LoRA {lp!r} as {name}…")
            # NOTE: Z-Image stores attention as a single fused QKV matrix. Many
            # community LoRAs ship separate to_q/to_k/to_v and will SILENTLY not
            # apply without conversion — use a recent diffusers (PR #12750+) or a
            # fused-QKV-aware loader and verify the keys actually matched.
            pipe.load_lora_weights(lp, **kw)
            names.append(name)
        pipe.set_adapters(names, adapter_weights=[args.lora_scale] * len(names))
        print(f"Fusing LoRA(s) {names} at scale {args.lora_scale}…")
        pipe.fuse_lora()
        pipe.unload_lora_weights()  # weights are now baked into the base modules
    # ------------------------------------------------------------------------

    with torch.no_grad():
        if args.only in (None, "vae"):
            print("Exporting VAE decoder…")
            export_vae_decoder(pipe, args.out, args.height, args.width, args.opset)
        if args.only in (None, "text"):
            print("Exporting text encoder…")
            export_text_encoder(pipe, args.out, args.seq, args.opset)
        if args.only in (None, "dit"):
            print("Exporting DiT transformer…")
            export_transformer(pipe, args.out, args.height, args.width, args.seq, args.opset)

    print("Done. Remember to also copy any *.onnx_data sidecar files.")


if __name__ == "__main__":
    main()
