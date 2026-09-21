"""Which combination exposes WebNN (navigator.ml) on this machine?

Tries a small matrix of browser builds and flags and prints what each one exposes. The bench page
cannot answer this: WebNN is off unless the browser is started with the right flag, and the flag
name has moved between Chrome versions.

Usage: python probe_webnn.py
"""

from __future__ import annotations

import json

from playwright.sync_api import sync_playwright

FLAG_SETS = {
    "none": [],
    "WebNN": ["--enable-features=WebNN"],
    "WebNN+blink": ["--enable-features=WebNN", "--enable-blink-features=WebNN"],
    "WebNN+CoreML+blink": [
        "--enable-features=WebNN,WebNNCoreML",
        "--enable-blink-features=WebNN",
    ],
    "WebNN+unsafe-webgpu+blink": [
        "--enable-features=WebNN",
        "--enable-blink-features=WebNN",
        "--enable-unsafe-webgpu",
    ],
}
CHANNELS = [("bundled-chromium", None), ("chrome", "chrome")]
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
                label = f"{channel_name}/{flag_name}"
                try:
                    browser = playwright.chromium.launch(channel=channel, args=flags, headless=True)
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
