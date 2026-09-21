# WebNN in the browser, measured, 2026-09-21

## The finding: the flag name was the whole problem

`chrome://flags` calls it "WebNN API", but the **registered feature name is
`WebMachineLearningNeuralNetwork`**. Passing `--enable-features=WebNN` does nothing at all and
fails silently — no warning, no `navigator.ml`, just a missing API. Everything below follows from
that one string.

    --enable-features=WebMachineLearningNeuralNetwork,WebNNCoreML

With it, **Chrome Canary 156 on macOS exposes `navigator.ml` with device types `npu`, `gpu` and
`cpu` — headless included.** No GPU-process tricks were needed; that hypothesis was wrong.

Working command (from the repo root, with a local server on the port the page expects):

    python tools/run_bench.py "http://localhost:8766/bench.html?runs=25" \
      --exe="/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary"

## The numbers (headless Canary 156, macOS, model fetched from Hugging Face)

| provider | cold start | median | p95 | decisions/s |
| --- | ---: | ---: | ---: | ---: |
| wasm | 1 494 ms | 45.5 ms | 47.1 ms | 22.0 |
| webgpu | 1 066 ms | 12.0 ms | 12.5 ms | 83.3 |
| webnn-cpu | 798 ms | 9.7 ms | 10.4 ms | 103.1 |
| webnn-gpu | 527 ms | 9.9 ms | 10.8 ms | 101.0 |
| webnn-npu | 601 ms | 10.0 ms | 13.8 ms | 100.0 |

## How to read it

The browser's NPU path works on macOS and is the fastest of the five, but only just: all three
WebNN device types land at ~10 ms, barely ahead of WebGPU's 12 ms and 4.6x ahead of wasm. For a
706 k model with a 3 840-token option block, the three accelerator targets are within 3 % of each
other, which says the time is **not** dominated by the arithmetic — it is dominated by per-call
overhead. That is the honest argument for a purpose-built runtime with one dispatch, and it also
gives that runtime a baseline to beat: 9.7 ms.

## Caveats that matter for a public demo

- Chrome/Edge **Canary only**, behind a flag. A visitor to a demo page cannot use this path, so the
  demo must select it opportunistically and fall back to webgpu/wasm.
- It is the *browser's* Core ML route to the Neural Engine, which is a different path from our own
  Core ML export (1.3 ms on the same machine). The browser stack costs roughly 8x more per call.

## A trap worth recording

`about:blank` is not a secure context, and neither WebGPU nor WebNN is exposed there. An early
probe of ours navigated to `about:blank` and reported both as missing; the platform was fine, the
probe was wrong. Probe on `http://localhost` (or https).
