# demo/c4-speed: one model, two runtimes

The Connect Four model from [`demo/c4`](../c4/) (v2, 7.4 M parameters), run two ways in the same page:

- **onnxruntime-web 1.30, WebAssembly**, as in the main demo (int8 file, 7.8 MB; also the fp32 file);
- **[onepass-webgpu](https://github.com/precisit/onepass-webgpu)**, a WebGPU runtime of 24 KB (8 KB gzip)
  written for one-pass models. It reads its weights straight from the unchanged ONNX files on Hugging Face:
  the fp32 file, or the same int8 file as the main demo. With the int8 file the weights stay 8-bit on the GPU
  (7.7 MB) and are unpacked inside the matrix multiply, with float math.

Play a game: the WebGPU runtime picks the AI's moves, and every move is also scored and timed by the other
engines. "Run the race" times each engine on the same 200 positions. The main demo is unchanged.

While you think, the GPU and CPU slow down to save power, and the first decision after a pause would pay
10 to 100 ms to wake them. So when you click, each engine gets one untimed warm-up run while your disc drops,
and the timed decision comes right after. Without it, single moves in a game looked several times slower
than in the race.

**The perfect solver, for comparison.** Connect Four is solved, so the page also times an exact solver,
[connect-four-ai](https://github.com/benjaminrall/connect-four-ai) (MIT), through its official WebAssembly build
(1.3 MB, with a built-in opening book). It is a search, not a model: instant when its opening book or a short
endgame answers, and up to several seconds right after the book runs out. The model takes about the same time on
every move. After each AI move the page says whether the model's move is one the solver rates best. In 200 full
games of perfect play on an idle Apple M5 Pro (onepass-webgpu `bench/results/2026-09-27-M5Pro-chrome-solver-games.json`):
the solver took under 0.1 ms per move at the median, 40 ms on average and 6.9 s at worst; onepass-webgpu took
1.2 ms at the median and 2.2 ms at worst, on the same positions.

## Checks

- **Encoder self-test:** the page re-encodes reference positions and compares the bytes with the Python tooling.
- **Runtime check:** before the WebGPU runtime plays, it must choose the same moves as onnxruntime-web (wasm,
  same fp32 file) on those positions, with scores within 1e-3. If it does not, or the browser has no WebGPU, the
  page says so and plays on WebAssembly.
- **Parity:** `?parity=<dir>` runs a whole eval set through the page's own encoder and engine code and leaves the
  result in `window.__result` (used headless by the onepass-webgpu tests). On all 17 325 positions of the
  Connect Four eval set, the f32 path chooses the same column as ONNX Runtime (CPU, fp32) every time
  (largest score difference 6.6e-5). With f16 weights (`?precision=f16`) 17 322 of 17 325 match; the other
  three are near ties. The int8 path (`?parity=<dir>&engine=gpu8`) chooses the same column as its reference, the
  same weights dequantized in Python, on all 17 325 positions (largest score difference 1.7e-5). It matches
  ONNX Runtime fp32 on 17 128 positions; onnxruntime-web's int8, which also rounds activations, on 17 004.

## Speed

Measured under the frozen [speed protocol](https://github.com/precisit/onepass-webgpu/blob/main/SPEED-PROTOCOL.md)
(500 eval positions after 20 warm-up moves, one at a time, three runs, median of medians). The records are in
[`bench/results/`](https://github.com/precisit/onepass-webgpu/tree/main/bench/results).

Apple M5 Pro, idle (load average about 1), Chrome 154 (headless, Metal adapter), runtime commit `94c8af9`:

| engine | model file | median per move | p95 | engine set-up + first move | runtime code (gzip) |
| --- | --- | ---: | ---: | ---: | ---: |
| onepass-webgpu, f32 | fp32 ONNX, 29.7 MB | **1.3 ms** | 1.4 ms | 28 ms | 37 KB (12 KB) |
| onepass-webgpu, f16 weights | the same file | **0.9 ms** | 1.1 ms | 30 ms | 37 KB (12 KB) |
| onepass-webgpu, int8 weights | int8 ONNX, 7.8 MB | **1.0 ms** | 1.1 ms | 23 ms | 37 KB (12 KB) |
| onnxruntime-web, wasm (the main demo) | int8 ONNX, 7.8 MB | 12.7 ms | 12.9 ms | 180 ms | 14.3 MB (3.7 MB) |
| onnxruntime-web, wasm | fp32 ONNX, 29.7 MB | 12.2 ms | 12.3 ms | 184 ms | 14.3 MB (3.7 MB) |

This page vendors an earlier 24 KB build of the runtime with the same kernels; the 37 KB build adds loading any
unchanged ONNX file of this model family in the browser. The page's own "race" is a quick check, not the protocol:
expect similar ratios, with more noise, on a busy machine.

## Files

- `index.html`: the page.
- `onepass-webgpu.js`: the runtime, built from the commit named in its first line.
- `onepass-c4-v2.plan.json`, `onepass-c4-v2-int8.plan.json`: the compiled plans (which ONNX initializer each weight
  comes from).
- `onepass-c4-v2-int8.probes.json`: expected int8 scores for the self-test positions (the int8 runtime check).
- `selftest-v2.json`: reference encodings, the same file as in `demo/c4`.

URL options: `?precision=f16` (f16 weights), `?model=<url>` and `?int8=<url>` (other model files).
