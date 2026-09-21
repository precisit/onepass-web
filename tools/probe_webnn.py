"""Which combination exposes WebNN (navigator.ml) on this machine?

Tries a small matrix of browser builds and flags and prints what each one exposes. The bench page
cannot answer this: WebNN is off unless the browser is started with the right flag, and the flag
name has moved between Chrome versions.

Usage: python probe_webnn.py
"""

from __future__ import annotations

import json
import os

from playwright.sync_api import sync_playwright

HEADLESS = os.environ.get("PROBE_HEADED") != "1"
ONLY = os.environ.get("PROBE_ONLY", "")

GPU_HELPERS = ["--ignore-gpu-blocklist", "--enable-gpu", "--disable-gpu-sandbox", "--use-angle=metal"]
FLAG_SETS = {
    "none": [],
    # The feature is registered as WebMachineLearningNeuralNetwork. "WebNN" is only the name of the
    # flag in chrome://flags, which is why passing --enable-features=WebNN did nothing at all.
    "correct": ["--enable-features=WebMachineLearningNeuralNetwork"],
    "correct+coreml": ["--enable-features=WebMachineLearningNeuralNetwork,WebNNCoreML"],
    "correct+coreml+gpu": ["--enable-features=WebMachineLearningNeuralNetwork,WebNNCoreML"] + GPU_HELPERS,
}
CHANNELS = [
    ("bundled-chromium", None),
    ("chrome-canary", {"executable_path": "/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary"}),
]
# Secure-context matters: WebGPU and WebNN are not exposed on about:blank, so the probe has to run
# on a real origin. localhost counts as secure.
PROBE_URL = "http://localhost:8766/"


def probe(page) -> dict:
    return page.evaluate(
        """async () => {
            const out = {
              webnn_api: 'ml' in navigator,
              webgpu: 'gpu' in navigator,
              ml_device_types: [],
              error: null,
            };
            if (out.webnn_api) {
              for (const deviceType of ['npu', 'gpu', 'cpu']) {
                try {
                  const context = await navigator.ml.createContext({ deviceType });
                  out.ml_device_types.push(deviceType);
                  if (context && context.destroy) context.destroy();
                } catch (error) { /* this device type is not available */ }
              }
            }
            return out;
        }"""
    )


def main() -> None:
    report: dict[str, dict] = {}
    with sync_playwright() as playwright:
        for channel_name, channel in CHANNELS:
            for flag_name, flags in FLAG_SETS.items():
                if ONLY and ONLY not in channel_name:
                    continue
                label = f"{channel_name}/{flag_name}"
                try:
                    browser = playwright.chromium.launch(args=flags, headless=HEADLESS, **(channel or {}))
                    page = browser.new_page()
                    page.goto(PROBE_URL)
                    report[label] = probe(page)
                    browser.close()
                except Exception as error:  # a missing channel is a result too
                    report[label] = {"error": str(error)[:160]}
                print(f"{label}: {json.dumps(report[label], sort_keys=True)}", flush=True)
    print("\n" + json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
