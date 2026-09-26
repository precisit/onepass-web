# demo/c4 — play Connect Four against a one-pass model

Open `index.html` (or the published page) and click a column. Pick who moves first; pick your opponent:
the current model (v2), the previous one (v1), or a random player.

Every model decision is **one forward pass**: the board goes in as 44 bytes (`1:` or `2:` for who
opened, then 42 cells from the mover's point of view), the legal columns go in as short strings
(`column 4`), and the model returns one score per column. The page plays the highest score. There is
no search, no hand-written rule and no server — the scores under the board are the whole decision.

## The models

| | v2 (current) | v1 (previous) |
| --- | --- | --- |
| architecture | one-pass scorer, 8 layers × 256 | the same scorer, 2 layers × 128 |
| parameters | 7.4 M | 0.7 M |
| file | `onepass-c4-v2-int8.onnx`, 7.8 MB (int8) | `onepass-c4-8x24.onnx`, 2.9 MB |
| input | 44 bytes, 7 options × 8 bytes | 224 bytes, 8 options × 24 bytes |
| trained on | ~42 M positions labelled by an exact solver, every column scored | ~80 k endgame positions |

Measured with a frozen protocol (200 games per match, both players make a random move 5 % of the
time, colours alternate, starting from an empty board):

| against | v2's score | v1's score |
| --- | ---: | ---: |
| a depth-2 search bot | 0.91 | 0.03 |
| a depth-4 search bot | 0.91 | 0.03 |
| a depth-6 search bot | 0.88 | 0.02 |
| a random player | 1.00 | 0.89 |
| a perfect player | 0.48 | — |
| each other | 0.985 | 0.015 |

For scale: a *perfect* player scores 0.89 against the depth-4 bot under the same protocol (its own
random moves cost it the rest), so v2 plays at that level. On a held-out set covering every phase of
the game, 98.7 % of its moves keep the game-theoretic value of the position. It is not perfect:
against a perfect player (both sides with the same 5 % random moves) it is close to even, 0.48.

## The arena

Pick any two players and run 20 games. They swap who opens every game, and each game starts with a
couple of random moves (the "random opening" setting) — otherwise two deterministic models would play
the same two games over and over, and "20–0" would mean nothing. The tally shows a 95 % interval and
how many distinct games were actually played.

## Self-test

On every load the page re-encodes reference positions with its own JavaScript and compares the bytes
with hashes produced by the Python tooling — once for each model's encoder. A wrong encoder looks
exactly like a weak model, which is why it is checked on every load rather than trusted once.

The runtime is checked too: before an accelerator (WebGPU) is used for a model, its scores on the same
reference positions must match the wasm backend's; otherwise the page falls back to wasm and says so
in the log. This exists because the first v2 release looked like it "always played column 1": the int8
model returns all-zero scores on onnxruntime-web's WebGPU backend, and the human game used WebGPU while
every check had run on wasm. v2 now runs on wasm (~20 ms a move).
