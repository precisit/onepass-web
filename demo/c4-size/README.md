# demo/c4-size: a smaller AI in your browser (Size)

The story behind the ternary model: [A game-playing AI in 1.6 MB](https://precisit.com/en/blog/onepass-c4-size/)
(also [in Swedish](https://precisit.com/blog/onepass-c4-size/)).

The Connect Four model with **ternary weights**: -1, 0 or +1 times one scale per 128 weights. It plays next to the
dense model, the first model and a perfect solver, all in the browser on
[onepass-webgpu](https://github.com/precisit/onepass-webgpu) (the solver on WebAssembly).

| player | file | vs depth-4 bot | vs depth-6 bot |
| --- | ---: | ---: | ---: |
| T34, trained from scratch (longer final stage) | 1.59 MB | 0.930 | 0.910 |
| Base243, trained from scratch | 1.93 MB | 0.885 | 0.877 |
| v2, dense int8 file (the Model page) | 7.8 MB | 0.905 | 0.878 |
| v1, the first model (fp32) | 2.9 MB | 0.03 | 0.02 |
| perfect solver ([connect-four-ai](https://github.com/benjaminrall/connect-four-ai), WebAssembly) | 1.26 MB | 0.890 | 0.925 |

Game scores are fixed results of the same protocol as the Model page (200 games, 5 % random moves on both sides,
a draw counts ½). The page adds a live benchmark on 200 boards (how often each player picks a move the solver
rates best, and its time per move) and an arena between any two players.

- **Formats:** T34 is the 3:4 structured ternary format of Sherry (Tencent, ACL 2026); Base243 is base-3 packing,
  five weights per byte, as in llama.cpp's `TQ1_0`.
- **Model files:** loaded from [Hugging Face](https://huggingface.co/precisit/onepass-c4) (`onepass-c4-v3-t34.onnx`,
  `onepass-c4-v3-b243.onnx`, and v2 and v1). Here: their runtime plans (`models/*.plan.json`) and expected-score
  probes (ONNX Runtime running each file).
- **Runtime:** `onepass-webgpu.js` (a vendored build of onepass-webgpu, MIT) with the format plugin
  `ternary-formats.js` from [onepass-webgpu-ternary](https://github.com/precisit/onepass-webgpu-ternary).
- **Checks on every load:** the encoders reproduce the Python tooling's bytes on reference positions, and every
  model's scores on the GPU match the expected scores for its file before it may play.

Training, formats, every measurement and the kernels: [precisit/onepass-webgpu-ternary](https://github.com/precisit/onepass-webgpu-ternary).
