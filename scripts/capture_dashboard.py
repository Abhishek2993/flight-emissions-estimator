"""Capture README screenshots of the running dashboard.

Usage (with the dashboard running: ``streamlit run app/dashboard.py``)::

    pip install playwright && playwright install chromium
    python scripts/capture_dashboard.py --url http://localhost:8501 --out docs/figures

Set ``CHROMIUM_PATH`` to use an existing Chromium binary instead of a downloaded one.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from playwright.sync_api import sync_playwright


def capture(url: str, out: Path, flight: str) -> None:
    """Screenshot the Flight tab (with ``flight`` selected) and the Compare tab."""
    out.mkdir(parents=True, exist_ok=True)
    executable = os.environ.get("CHROMIUM_PATH")
    with sync_playwright() as p:
        browser = (
            p.chromium.launch(executable_path=executable) if executable else p.chromium.launch()
        )
        page = browser.new_page(viewport={"width": 1500, "height": 1180}, device_scale_factor=2)
        page.goto(url, wait_until="networkidle")
        page.wait_for_selector(".js-plotly-plot", timeout=120_000)
        page.locator('[data-testid="stSelectbox"]').first.click()
        page.get_by_role("option", name=flight).first.click()
        page.wait_for_timeout(5000)
        page.screenshot(path=str(out / "dashboard_flight.png"))
        page.get_by_role("tab", name="Compare").click()
        page.wait_for_timeout(4000)
        page.screenshot(path=str(out / "dashboard_compare.png"))
        browser.close()


def main() -> None:
    """Parse arguments and capture the screenshots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8501")
    parser.add_argument("--out", type=Path, default=Path("docs/figures"))
    parser.add_argument("--flight", default="ELY1747", help="text of the flight to select")
    args = parser.parse_args()
    capture(args.url, args.out, args.flight)


if __name__ == "__main__":
    main()
