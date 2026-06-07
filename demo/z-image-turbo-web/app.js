// app.js — UI wiring for the in-browser Z-Image Turbo DiT pipeline.

import { ZImagePipeline, pickExecutionProvider } from "./pipeline.js";

const $ = (id) => document.getElementById(id);
const logbox = $("logbox");
const appendLog = (msg) => {
  logbox.textContent += `${msg}\n`;
  logbox.scrollTop = logbox.scrollHeight;
};

let pipeline = null;

// Show the detected execution provider up front so users know what they'll get.
pickExecutionProvider().then((ep) => {
  $("epInfo").textContent =
    ep === "webgpu"
      ? "WebGPU available — GPU accelerated 🚀"
      : "WebGPU unavailable — falling back to WebAssembly (CPU)";
});

$("loadBtn").addEventListener("click", async () => {
  const cfg = {
    textEncoderUrl: $("textEncoderUrl").value.trim(),
    transformerUrl: $("transformerUrl").value.trim(),
    vaeDecoderUrl: $("vaeDecoderUrl").value.trim(),
    tokenizer: $("tokenizer").value.trim(),
    latentChannels: Number($("latentChannels").value),
    vaeScale: Number($("vaeScale").value),
    vaeScaling: Number($("vaeScaling").value),
  };

  if (!cfg.textEncoderUrl || !cfg.transformerUrl || !cfg.vaeDecoderUrl) {
    appendLog("⚠️  Please provide all three ONNX model URLs first.");
    return;
  }

  $("loadBtn").disabled = true;
  $("loadBtn").textContent = "Loading…";
  try {
    pipeline = new ZImagePipeline(cfg);
    await pipeline.init(appendLog);
    $("genBtn").disabled = false;
    $("loadBtn").textContent = "Reload pipeline";
    appendLog("✅ Ready to generate.");
  } catch (err) {
    appendLog(`❌ Load failed: ${err.message}`);
    $("loadBtn").textContent = "Load pipeline";
  } finally {
    $("loadBtn").disabled = false;
  }
});

$("genBtn").addEventListener("click", async () => {
  if (!pipeline) return;
  const steps = Number($("steps").value);
  const prog = $("prog");
  prog.hidden = false;
  prog.max = steps;
  prog.value = 0;
  $("genBtn").disabled = true;

  try {
    const result = await pipeline.generate(
      {
        prompt: $("prompt").value,
        steps,
        seed: Number($("seed").value),
        width: Number($("width").value),
        height: Number($("height").value),
      },
      {
        onLog: appendLog,
        onStep: (i, n) => {
          prog.value = i;
          prog.max = n;
        },
      }
    );

    const canvas = $("canvas");
    canvas.width = result.width;
    canvas.height = result.height;
    const ctx = canvas.getContext("2d");
    ctx.putImageData(new ImageData(result.data, result.width, result.height), 0, 0);
    appendLog("🖼️  Done.");
  } catch (err) {
    appendLog(`❌ Generation failed: ${err.message}`);
    console.error(err);
  } finally {
    $("genBtn").disabled = false;
    prog.hidden = true;
  }
});
