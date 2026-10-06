// Web Worker: runs OpenAI Whisper (ONNX, via transformers.js) entirely in the
// browser so the UI stays responsive while the model loads and transcribes.
import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

env.allowLocalModels = false;

let asr = null;
let current = null;

async function load(model) {
  if (asr && current === model) return asr;
  current = model;
  asr = null;
  asr = await pipeline("automatic-speech-recognition", model, {
    dtype: "q8",
    device: "wasm",
    progress_callback: (p) => self.postMessage({ type: "progress", data: p }),
  });
  return asr;
}

self.onmessage = async (e) => {
  const { type, model, audio } = e.data;
  try {
    if (type === "load") {
      await load(model);
      self.postMessage({ type: "ready", model });
    } else if (type === "transcribe") {
      const t0 = performance.now();
      const pipe = await load(model);
      const out = await pipe(audio);
      self.postMessage({
        type: "result",
        text: (out.text || "").trim(),
        ms: Math.round(performance.now() - t0),
      });
    }
  } catch (err) {
    self.postMessage({ type: "error", error: String(err && err.message ? err.message : err) });
  }
};
