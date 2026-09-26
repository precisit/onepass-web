"""Drive the Connect Four demo headlessly: self-tests, browser-vs-Python parity, and arena runs.

The public page and the published numbers must not be able to disagree. This script loads
`demo/c4/index.html` in headless Chromium (the same code paths the buttons use) and:

  1. checks both encoders' self-tests passed on load;
  2. optionally compares the page's choices with reference choices computed in Python
     (`--parity rows.json`: [{"board": [42 ints], "choice": col}, ...]) - through the *same provider
     path the human game uses* (webgpu first, then wasm), and through wasm; a check that only ever
     runs on wasm once missed that the int8 model returned all-zero scores on webgpu;
  3. runs arena matches (`--match "A|B"`, repeatable) with a pinned seed and prints the tallies.

Usage (serve the repo root first, e.g. `python3 -m http.server 8767`):
    python run_arena.py --url "http://localhost:8767/demo/c4/?v2=/onepass-c4-v2.onnx" \
        --match "onepass-c4 v2|onepass-c4 v1 (previous)" --games 200 --opening 2 --seed 1
"""

from __future__ import annotations

import argparse
import json

from playwright.sync_api import sync_playwright


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", required=True)
    parser.add_argument("--match", action="append", default=[])
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--opening", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--parity", help="JSON rows with board + expected choice for the v2 model")
    args = parser.parse_args()

    report: dict = {"url": args.url}
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, args=["--enable-unsafe-webgpu"])
        page = browser.new_page()
        page.on("console", lambda msg: None)
        page.goto(args.url)
        page.wait_for_function("window.__ready === true", timeout=300_000)
        report["self_test_ok"] = page.evaluate("window.__selfTestOk")
        report["human_engine_provider"] = page.evaluate(
            "() => window.__providerOf('onepass-c4 v2', ['webgpu', 'wasm'])")
        report["log"] = page.evaluate("document.getElementById('log').textContent")
        if args.parity:
            rows = json.load(open(args.parity))
            report["parity"] = {}
            for label, providers in (("human-path", ["webgpu", "wasm"]), ("wasm", ["wasm"])):
                agree = 0
                for row in rows:
                    got = page.evaluate("([b, p]) => window.__chooseFor('onepass-c4 v2', b, p)", [row["board"], providers])
                    agree += got == row["choice"]
                report["parity"][label] = {"n": len(rows), "agree": agree}
        report["matches"] = []
        for match in args.match:
            name_a, name_b = match.split("|")
            result = page.evaluate(
                "(o) => window.__runArena(o)",
                {"games": args.games, "nameA": name_a, "nameB": name_b, "opening": args.opening, "seed": args.seed},
            )
            report["matches"].append(result)
        browser.close()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
