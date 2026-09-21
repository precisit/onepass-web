# Browser bench, 2026-09-21

Method: `runtime/bench.html`, headless Chromium on an M4, model fetched over the network from
`https://huggingface.co/precisit/one-pass-sv-forms/resolve/main/sv0-forms.onnx` (2 961 519 bytes),
shapes context 224 bytes, 40 option slots x 96 bytes, 60 timed decisions after one warmup call.
Cold start includes the model download.

| provider | cold start | median | p95 | decisions/s |
| --- | ---: | ---: | ---: | ---: |
| wasm | 1476.5 ms | 45.20 ms | 46.20 ms | 22.1 |
| webgpu | 640.0 ms | **10.90 ms** | 14.10 ms | **91.7** |

Same measurement earlier in the day with the model on local disk (no download in the cold start):
wasm 378 ms / 46.10 ms / 46.50 ms, webgpu 79 ms / 10.50 ms / 11.50 ms.

## Runtime size, for the same model

| file | bytes |
| --- | ---: |
| `ort.webgpu.mjs` | 677 191 |
| `ort-wasm-simd-threaded.jsep.mjs` | 46 851 |
| `ort-wasm-simd-threaded.jsep.wasm` | 28 312 028 |

Total runtime to run a 2 961 519-byte model: **29 036 070 bytes (29.0 MB)**, i.e. the engine is
roughly ten times the model. This is the argument for a purpose-built runtime; it is a size
argument, not a speed argument.

## What this means for a demo

A turn-based game needs one decision per move: 10.9 ms per decision is imperceptible, so ONNX
Runtime Web is sufficient today. A crowd of agents (many decisions per tick) is the case where a
custom runtime and batching start to matter, and that is the measurement to run before writing any
kernels.
