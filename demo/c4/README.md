# demo/c4 — play Connect Four against a 706 k one-pass model

Open `index.html` (or the published page) and click a column. You are X; the model is O.

Every model decision is one forward pass over 224 context bytes and 8 option slots × 24 bytes, run
by ONNX Runtime Web on WebGPU (falling back to wasm). Measured on an M4: **3.9 ms per decision** on
WebGPU, 7.8 ms on wasm — after the option block was cut from 40 × 96 to 8 × 24, which is where most
of the speed came from.

## What you are looking at

* The **bars** show the model's score for each legal column. On your turn they are the model's view of
  *your* options; on its turn, its own choice (green = the column it played).
* The **self-test** in the log compares this page's JavaScript encoder against reference hashes
  produced by the Python corpus tooling (`make_selftest.py`, same 224/8/24 limits). It replays the
  reference move history with this page's own renderer and re-encodes it, so both the renderer and the
  encoder are checked. A wrong encoder looks exactly like a weak model, which is why it is checked on
  every load rather than trusted once.

## Honest expectations

The model is weak: 78.05 % optimal moves against an exact solver, but it reads the position from the
**move history** in its context, and it loses every game to a depth-2 search bot. It does not know the
centre column is good on an empty board (it scores the centre −0.7 and column 1 +3.2). Beat it easily.
