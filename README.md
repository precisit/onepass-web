# onepass-web

A runtime and a set of demos for running **one-pass specialists** in the browser.

A one-pass specialist answers one question per forward pass: given a piece of context and a
supplied list of options, which option is best? No tokenizer, no generation, no server. This repo
is where that contract gets a browser home: the runtime that runs a specialist, and the demos that
show one in use.

## Status

| Piece | State |
| --- | --- |
| Bench instrument (`runtime/bench.html`) | working, used for the numbers below |
| ONNX Runtime Web path (wasm, webgpu) | measured, see table |
| WebNN (the browser's route to the NPU) | **measured in Chrome Canary**: 9.7 ms median, the fastest of the five providers — but behind a flag, so demos must treat it as opportunistic. See `docs/webnn-status-2026-09-21.md` |
| Custom WGSL runtime | planned, not started |
| Game demo (`demo/c4/`) | **playable, and it plays well** — v2 (7.4 M params, int8, 7.8 MB) scores 0.91 vs a depth-4 bot and beats v1 200–0 in the arena; self-testing encoders (see `demo/c4/README.md`) |

## What we measured (headless Chromium, M4; Swedish form specialist, 706 k params, 2.82 MB ONNX, 40 option slots x 96 bytes)

| provider | cold start | median | p95 | decisions/s |
| --- | ---: | ---: | ---: | ---: |
| wasm | 1 494 ms | 45.5 ms | 47.1 ms | 22.0 |
| webgpu | 1 066 ms | 12.0 ms | 12.5 ms | 83.3 |
| webnn-cpu | 798 ms | 9.7 ms | 10.4 ms | 103.1 |
| webnn-gpu | 527 ms | 9.9 ms | 10.8 ms | 101.0 |
| webnn-npu | 601 ms | 10.0 ms | 13.8 ms | 100.0 |

All three WebNN device types land within 3 % of each other, and barely ahead of WebGPU: for a model
this small the cost is per-call overhead, not arithmetic. Chrome Canary 156, headless, macOS.

The wasm result is the informative one: 45.5 ms against ~10-12 ms for every accelerator path, which
is the opposite of the usual advice for small models. The reason is the input shape: the option block
is 40 x 96 = 3 840 option tokens plus 224 context bytes, so the graph is parallel work, and that is
what an accelerator is for. "WASM beats WebGPU for small models" is the wrong question; the input
shape decides.

Among the accelerators the spread is small - 9.7 to 12.0 ms, all within 25 % - so for this model the
device type matters much less than the fact that *some* accelerator is used. At 3 840 tokens and
706 k parameters the cost is dominated by per-call overhead, not arithmetic, which is exactly what a
purpose-built runtime would remove.

## Why a custom runtime, if ONNX Runtime already works

Because of size, not speed. The WebGPU path of ONNX Runtime ships **29.0 MB** of runtime (28.3 MB
of that is a single wasm binary) to run a **2.82 MB** model: the engine is ten times the thing it
drives. A purpose-built runtime for this fixed shape - byte embedding, a couple of encoder blocks,
a projection per option - should be a couple of hundred kilobytes, and it can dispatch once instead
of per operator.

That trade only makes sense because this repo is meant to hold *several* demos and *larger* models
later: a shared runtime amortises, and a runtime we own is a runtime we can measure. Until those
demos exist, ONNX Runtime is the honest default, and the bench numbers above say it is good enough
for a turn-based game (10 ms per move is imperceptible).

## Layout

    runtime/           the instrument, and later the kernels
      bench.html       measure cold start and per-decision latency for any one-pass ONNX model
    tools/             how to reproduce a measurement the browser will not do on its own
      run_bench.py     run the bench with a browser flag (WebNN) and print the results as JSON
      probe_webnn.py   which build and flag combination exposes WebNN on this machine
    demo/              one folder per demo, each deployable as static files
    docs/              findings that belong to the runtime rather than to a demo

## Running the bench

    cd runtime && python3 -m http.server 8765
    # then open http://localhost:8765/bench.html

Query parameters: `model` (URL of a one-pass ONNX file), `ctx` (context bytes, default 224),
`opt` (max options, default 40), `optbytes` (bytes per option, default 96), `runs` (timed
decisions, default 60). Results are printed as JSON and left on `window.__benchResults`.

A model file is *not* committed here: models live on Hugging Face and are fetched over the
network, which also keeps the page honest about what a first load actually costs.

## Licence

MIT, Precisit AB.
