"""Run the browser bench with WebNN enabled, and print the results as JSON.

The bench page itself cannot turn WebNN on: it is off by default behind a browser flag. This
script launches Chromium with that flag so the webnn rows can actually be measured.

Usage:
    python run_bench.py [url] [--headed] [--flag NAME]
"""

from __future__ import annotations

import json
import sys

from playwright.sync_api import sync_playwright

URL = "http://localhost:8766/bench.html?runs=40"
HEADED = False
CHANNEL: dict | None = None
EXTRA: list[str] = []
for argument in sys.argv[1:]:
    if argument == "--headed":
        HEADED = True
    elif argument.startswith("--exe="):
        CHANNEL = {"executable_path": argument.split("=", 1)[1]}
    elif argument.startswith("--channel="):
        CHANNEL = {"channel": argument.split("=", 1)[1]}
    elif argument.startswith("--flag="):
        EXTRA.append(argument.split("=", 1)[1])
    elif argument.startswith("http"):
        URL = argument

# WebNN on macOS is served by the CoreML backend, which is what reaches the Neural Engine. Both
# feature names are passed because the internal name has moved between Chrome versions.
FLAGS = [
    # The feature name matters: chrome://flags calls it "WebNN", but the registered feature is
    # WebMachineLearningNeuralNetwork. Passing --enable-features=WebNN silently does nothing.
    "--enable-features=WebMachineLearningNeuralNetwork,WebNNCoreML",
    "--ignore-gpu-blocklist",
    "--enable-unsafe-webgpu",
    *EXTRA,
]


def main() -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(args=FLAGS, headless=not HEADED, **(CHANNEL or {}))
        page = browser.new_page()
        page.goto(URL)
        page.wait_for_function("window.__benchDone === true", timeout=300_000)
        report = {
            "user_agent": page.evaluate("navigator.userAgent"),
            "webnn_api_present": page.evaluate("'ml' in navigator"),
            "webgpu_present": page.evaluate("'gpu' in navigator"),
            "results": page.evaluate("window.__benchResults"),
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        browser.close()


if __name__ == "__main__":
    main()
