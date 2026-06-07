// pipeline.js
//
// A browser-native text-to-image diffusion pipeline built on ONNX Runtime Web.
//
// It implements the three-stage DiT (Diffusion Transformer) pipeline described in
// Intel's "From U-Net to DiT: Z-Image Turbo Runs in Your Browser":
//
//   1. Text encoder   : prompt  -> conditioning embeddings
//   2. DiT denoiser    : iteratively denoises a latent under a flow-matching schedule
//   3. VAE decoder    : latent  -> RGB image
//
// Each stage is a separate ONNX graph loaded as an ort.InferenceSession. The denoiser
// is run for only a handful of steps because Z-Image Turbo is an 8-step distilled model.
//
// Everything runs locally in the browser: WebGPU when available, WebAssembly otherwise.

import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.webgpu.mjs";
import { AutoTokenizer } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.3.3/dist/transformers.min.js";

// Point the WASM runtime (used for the wasm EP and for WebGPU's CPU fallback ops) at the CDN.
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";

const log = (cb, msg) => {
  if (cb) cb(msg);
  // eslint-disable-next-line no-console
  console.log("[pipeline]", msg);
};

/**
 * Pick the best available execution provider.
 * WebGPU is dramatically faster for the transformer/VAE; WASM is the universal fallback.
 */
export async function pickExecutionProvider() {
  if (navigator.gpu) {
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter) return "webgpu";
    } catch {
      /* fall through */
    }
  }
  return "wasm";
}

/**
 * A small seedable PRNG (mulberry32) plus a Box–Muller normal sampler so that a given
 * seed reproduces the same initial latent noise across runs.
 */
function makeRng(seed) {
  let a = seed >>> 0;
  const next = () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  return {
    uniform: next,
    normal() {
      // Box–Muller transform.
      let u = 0;
      let v = 0;
      while (u === 0) u = next();
      while (v === 0) v = next();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    },
  };
}

function randnTensor(shape, rng) {
  const size = shape.reduce((a, b) => a * b, 1);
  const data = new Float32Array(size);
  for (let i = 0; i < size; i++) data[i] = rng.normal();
  return new ort.Tensor("float32", data, shape);
}

/**
 * Flow-matching (rectified-flow) sigma schedule.
 *
 * DiT models like Z-Image are trained with a flow-matching objective rather than the
 * DDPM noise schedule used by classic U-Net Stable Diffusion. The latent travels along
 * a straight path from pure noise (sigma=1) to the clean sample (sigma=0); the network
 * predicts the velocity along that path and we integrate it with explicit Euler steps.
 */
function flowMatchSigmas(steps) {
  const sigmas = [];
  for (let i = 0; i < steps; i++) sigmas.push(1 - i / steps);
  sigmas.push(0); // final target
  return sigmas;
}

export class ZImagePipeline {
  /**
   * @param {object} cfg
   * @param {string} cfg.textEncoderUrl  URL to the text-encoder ONNX file
   * @param {string} cfg.transformerUrl  URL to the DiT denoiser ONNX file
   * @param {string} cfg.vaeDecoderUrl   URL to the VAE decoder ONNX file
   * @param {string} cfg.tokenizer       HF tokenizer repo id (e.g. "Qwen/Qwen2.5-1.5B")
   * @param {number} cfg.latentChannels  latent channel count (default 16)
   * @param {number} cfg.vaeScale        spatial downscale factor of the VAE (default 8)
   * @param {number} cfg.vaeScaling      latent scaling factor applied before VAE decode
   * @param {number} cfg.maxTokens       tokenizer max length (default 512)
   */
  constructor(cfg) {
    this.cfg = {
      latentChannels: 16,
      vaeScale: 8,
      vaeScaling: 1.0,
      maxTokens: 512,
      ...cfg,
    };
    this.ep = null;
    this.tokenizer = null;
    this.textEncoder = null;
    this.transformer = null;
    this.vaeDecoder = null;
  }

  async init(progress) {
    this.ep = await pickExecutionProvider();
    log(progress, `Execution provider: ${this.ep.toUpperCase()}`);

    const sessionOpts = {
      executionProviders: [this.ep],
      graphOptimizationLevel: "all",
    };

    log(progress, `Loading tokenizer: ${this.cfg.tokenizer}…`);
    this.tokenizer = await AutoTokenizer.from_pretrained(this.cfg.tokenizer);

    log(progress, "Loading text encoder…");
    this.textEncoder = await ort.InferenceSession.create(this.cfg.textEncoderUrl, sessionOpts);

    log(progress, "Loading DiT denoiser… (this is the big one)");
    this.transformer = await ort.InferenceSession.create(this.cfg.transformerUrl, sessionOpts);

    log(progress, "Loading VAE decoder…");
    this.vaeDecoder = await ort.InferenceSession.create(this.cfg.vaeDecoderUrl, sessionOpts);

    log(progress, "Pipeline ready.");
  }

  /** Tokenize a prompt and run the text encoder, returning the conditioning tensor. */
  async encodePrompt(prompt) {
    const enc = await this.tokenizer(prompt, {
      padding: "max_length",
      max_length: this.cfg.maxTokens,
      truncation: true,
      return_tensor: false,
    });

    const ids = BigInt64Array.from(enc.input_ids.map((x) => BigInt(x)));
    const mask = BigInt64Array.from(enc.attention_mask.map((x) => BigInt(x)));
    const len = enc.input_ids.length;

    const feeds = {};
    const names = this.textEncoder.inputNames;
    feeds[names[0]] = new ort.Tensor("int64", ids, [1, len]);
    if (names.length > 1) feeds[names[1]] = new ort.Tensor("int64", mask, [1, len]);

    const out = await this.textEncoder.run(feeds);
    return out[this.textEncoder.outputNames[0]];
  }

  /**
   * Generate an image.
   * @returns {{data: Uint8ClampedArray, width: number, height: number}}
   */
  async generate(opts, callbacks = {}) {
    const {
      prompt,
      steps = 8,
      seed = Math.floor(Math.random() * 2 ** 31),
      width = 512,
      height = 512,
    } = opts;
    const { onStep, onLog } = callbacks;

    const rng = makeRng(seed);

    log(onLog, `Encoding prompt (seed ${seed})…`);
    const cond = await this.encodePrompt(prompt);

    const lh = Math.floor(height / this.cfg.vaeScale);
    const lw = Math.floor(width / this.cfg.vaeScale);
    const latentShape = [1, this.cfg.latentChannels, lh, lw];

    let latents = randnTensor(latentShape, rng);
    const sigmas = flowMatchSigmas(steps);

    const tNames = this.transformer.inputNames;
    log(onLog, `DiT inputs: [${tNames.join(", ")}]`);

    for (let i = 0; i < steps; i++) {
      const sigma = sigmas[i];
      const dt = sigmas[i + 1] - sigma; // negative; we integrate toward sigma=0

      // Many flow-matching DiTs expect the timestep as t = sigma * 1000.
      const t = new ort.Tensor("float32", new Float32Array([sigma * 1000]), [1]);

      // Feed by position: latent, timestep, conditioning. Adjust names to your export.
      const feeds = {};
      feeds[tNames[0]] = latents;
      if (tNames.length > 1) feeds[tNames[1]] = t;
      if (tNames.length > 2) feeds[tNames[2]] = cond;

      const out = await this.transformer.run(feeds);
      const velocity = out[this.transformer.outputNames[0]];

      // Explicit Euler integration step: x <- x + dt * v
      const x = latents.data;
      const v = velocity.data;
      const nx = new Float32Array(x.length);
      for (let k = 0; k < x.length; k++) nx[k] = x[k] + dt * v[k];
      latents = new ort.Tensor("float32", nx, latentShape);

      log(onLog, `step ${i + 1}/${steps} (sigma ${sigma.toFixed(3)})`);
      if (onStep) onStep(i + 1, steps);
      // Yield to the event loop so the UI can paint progress.
      await new Promise((r) => setTimeout(r, 0));
    }

    // Apply the VAE latent scaling, if the export expects pre-scaled latents.
    if (this.cfg.vaeScaling !== 1.0) {
      const d = latents.data;
      const s = new Float32Array(d.length);
      for (let k = 0; k < d.length; k++) s[k] = d[k] / this.cfg.vaeScaling;
      latents = new ort.Tensor("float32", s, latentShape);
    }

    log(onLog, "Decoding latent with VAE…");
    const vaeFeeds = {};
    vaeFeeds[this.vaeDecoder.inputNames[0]] = latents;
    const decoded = await this.vaeDecoder.run(vaeFeeds);
    const image = decoded[this.vaeDecoder.outputNames[0]];

    return tensorToImageData(image, width, height);
  }
}

/**
 * Convert a [1,3,H,W] float tensor in [-1,1] (the usual VAE output range) to RGBA pixels.
 */
function tensorToImageData(tensor, fallbackW, fallbackH) {
  const [, c, h, w] = tensor.dims.length === 4 ? tensor.dims : [1, 3, fallbackH, fallbackW];
  const src = tensor.data;
  const plane = h * w;
  const rgba = new Uint8ClampedArray(plane * 4);
  for (let p = 0; p < plane; p++) {
    for (let ch = 0; ch < 3; ch++) {
      const val = src[ch * plane + p];
      // Map [-1,1] -> [0,255].
      rgba[p * 4 + ch] = Math.round((val * 0.5 + 0.5) * 255);
    }
    rgba[p * 4 + 3] = 255;
  }
  return { data: rgba, width: w, height: h, channels: c };
}
