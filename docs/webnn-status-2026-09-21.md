# WebNN status on this machine, 2026-09-21

## What we tried

`runtime/bench.html` now offers `webnn-cpu`, `webnn-gpu` and `webnn-npu` alongside `wasm` and
`webgpu`, and reports whether `navigator.ml` exists at all. To give WebNN a fair chance the browser
has to be started with its flag, so `tools/run_bench.py` launches Chromium with one.

| build | flags | `navigator.ml` | `navigator.gpu` | webnn rows |
| --- | --- | --- | --- | --- |
| bundled Chromium 153 (Playwright), headless | none / `WebNN` / `+blink` / `+CoreML` / `+unsafe-webgpu` | absent | present | unavailable |
| Google Chrome 151.0.7922.174, headless | same five sets | absent | present | unavailable |
| Google Chrome 151.0.7922.174, **headed** | `WebNN,WebNNCoreML,WebNNGPU` | absent | present | unavailable |

Every attempt returns the same error from the runtime, which is the honest result rather than a
page bug:

    no available backend found. ERR: [webnn] Error: WebNN is not supported in current environment

## Why

The WebNN implementation status page lists three Chromium backends: LiteRT (Windows, ChromeOS,
Android, Linux), Windows ML (Windows) and **Core ML (macOS)**. The macOS column is marked against
**Chrome Canary / Edge Canary** and the page labels the whole feature experimental. So on macOS,
WebNN reaches the Neural Engine through Core ML, but only in Canary builds today - not in the
stable channel, and not by passing a flag to it.

## What this means

- To measure WebNN here we need Chrome Canary (or Edge Canary) installed. The bench page needs no
  change: its rows will populate on their own, and `tools/run_bench.py --channel=chrome-canary`
  is the measurement command.
- Until then, the browser path we can measure on this machine is WebGPU: 12.7 ms per decision
  (headed Chrome, 25 runs) against 45.6 ms for wasm, i.e. 3.6x.
- The native path is unaffected and remains the fastest thing we have: Core ML int8 on the Neural
  Engine is ~1.3 ms per decision for the same model, and it verifies per compute path.

## A pitfall worth recording

`navigator.gpu` and `navigator.ml` are only exposed in a **secure context**. A probe that navigates
to `about:blank` reports both as absent and looks like a platform limitation when it is a probe
bug: ours did exactly that for one run. Use a real origin (`localhost` counts).

## Payload, measured by the page itself

The page reports its own transfer size from `performance.getEntriesByType("resource")`: 6 589 932
bytes across six requests for the `ort.all.mjs` bundle plus the model, against 2 961 519 bytes for
the model alone. The WebNN-capable bundle is the expensive part, which is one more data point for a
purpose-built runtime.
