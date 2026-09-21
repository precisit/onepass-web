# Browser bench, two machines, 2026-09-21

Same page (`runtime/bench.html`), same model (2 961 519-byte ONNX from Hugging Face), same flags,
Chrome Canary, headless. Two different Macs, and they do not agree on the winner.

| provider | M4 (10-core GPU) | M1 Max (32-core GPU) |
| --- | ---: | ---: |
| wasm | 45.5 ms | 124.4 ms |
| webgpu | 12.0 ms | **8.6 ms** |
| webnn-npu | **10.0 ms** | 25.5 ms |

The ranking is machine-dependent: the NPU path is fastest on the M4 and three times slower than
WebGPU on the M1 Max, where the big GPU wins. The wasm numbers differ by 2.7x as well, which is
more than the chips explain — the two runs were not measuring an identical configuration (thread
count is the usual cause), so treat the cross-machine column as a prompt to re-measure, not as a
conclusion.

## The measurement that would settle it: latency under load

Magnus's observation, and it is the right one: in a real game the **GPU is already busy drawing
frames**, while the NPU sits idle. A quiet-page median therefore answers the wrong question. What a
game needs is:

1. `webgpu` against `webnn-npu` while a GPU workload saturates the GPU (a spinning WebGL scene), and
2. the *frame time of the page* during inference — does a model call cause a visible hitch?

That experiment is not in the bench yet; it is the next thing to add, and it is the one that decides
which provider a game demo should prefer.
