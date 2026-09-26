# demo/c4-speed: one model, two runtimes

The Connect Four model from [`demo/c4`](../c4/) (v2, 7.4 M parameters), run two ways in the same page:

- **onnxruntime-web 1.30, WebAssembly**, as in the main demo (int8 file, 7.8 MB; also the fp32 file);
- **[onepass-webgpu](https://github.com/precisit/onepass-webgpu)**, a WebGPU runtime of 22 KB (7 KB gzip)
  written for one-pass models. It reads its weights straight from the unchanged fp32 ONNX file on
  Hugging Face.

Play a game: the WebGPU runtime picks the AI's moves, and every move is also scored and timed on
WebAssembly. "Run the race" times each engine on the same 200 positions. The main demo is unchanged.

While you think, the GPU and CPU slow down to save power, and the first decision after a pause would pay
10 to 100 ms to wake them. So when you click, each engine gets one untimed warm-up run while your disc drops,
and the timed decision comes right after. Without it, single moves in a game looked several times slower
than in the race.

## Checks

- **Encoder self-test:** the page re-encodes reference positions and compares the bytes with the Python tooling.
- **Runtime check:** before the WebGPU runtime plays, it must choose the same moves as onnxruntime-web (wasm,
  same fp32 file) on those positions, with scores within 1e-3. If it does not, or the browser has no WebGPU, the
  page says so and plays on WebAssembly.
- **Parity:** `?parity=<dir>` runs a whole eval set through the page's own encoder and engine code and leaves the
  result in `window.__result` (used headless by the onepass-webgpu tests). On all 17 325 positions of the
  Connect Four eval set, the f32 path chooses the same column as ONNX Runtime (CPU, fp32) every time
  (largest score difference 6.6e-5). With f16 weights (`?precision=f16`) 17 322 of 17 325 match; the other
  three are near ties.

## Speed

Measured under the frozen [speed protocol](https://github.com/precisit/onepass-webgpu/blob/main/SPEED-PROTOCOL.md)
(500 eval positions after 20 warm-up moves, one at a time, three runs, median of medians). The records are in
[`bench/results/`](https://github.com/precisit/onepass-webgpu/tree/main/bench/results).

Apple M1 Max, Chrome 153 (headless, Metal adapter), on a machine shared with other jobs:

| engine | model file | median per move | p95 | set-up + first move | runtime code (gzip) |
| --- | --- | ---: | ---: | ---: | ---: |
| onepass-webgpu, f32 | fp32 ONNX, 29.7 MB | **4.1 ms** | 5.1 ms | 144 ms | **22 KB (7 KB)** |
| onepass-webgpu, f16 weights | the same file | 3.9 ms | 4.4 ms | 127 ms | 22 KB (7 KB) |
| onnxruntime-web, wasm (the main demo) | int8 ONNX, 7.8 MB | 20.4 ms | 21.4 ms | 398 ms | 14.3 MB (3.7 MB) |
| onnxruntime-web, wasm | fp32 ONNX, 29.7 MB | 19.5 ms | 21.2 ms | 385 ms | 14.3 MB (3.7 MB) |

The WebGPU runtime downloads the fp32 file (29.7 MB) for now; reading the int8 file directly is planned.
The page's own "race" is a quick check, not the protocol: expect similar ratios, with more noise.

## Files

- `index.html`: the page.
- `onepass-webgpu.js`: the runtime, built from the commit named in its first line.
- `onepass-c4-v2.plan.json`: the compiled plan (which ONNX initializer each weight comes from).
- `selftest-v2.json`: reference encodings, the same file as in `demo/c4`.

URL options: `?precision=f16` (f16 weights), `?model=<url>` and `?int8=<url>` (other model files).
