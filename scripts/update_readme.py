"""Copy the tables of the latest pipeline run into README.md.

The README contains marker pairs such as ``<!-- results:start -->`` / ``<!-- results:end -->``.
Everything between a pair is replaced by the corresponding file from the results directory,
so the numbers in the README always come from an actual run::

    python -m flight_emissions run --input data/sample --out results/ --figures-dir docs/figures
    python scripts/sensitivity.py --out results/
    python scripts/update_readme.py
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

BLOCKS = {"results": "summary.md", "sensitivity": "sensitivity.md"}


def update(readme: Path, results: Path) -> list[str]:
    """Replace every marked block whose source file exists; return the updated block names."""
    text = readme.read_text()
    updated = []
    for name, filename in BLOCKS.items():
        source = results / filename
        pattern = re.compile(rf"(<!-- {name}:start -->\n).*?(<!-- {name}:end -->)", re.S)
        if source.exists() and pattern.search(text):
            body = source.read_text().strip() + "\n"
            text = pattern.sub(lambda m, body=body: m.group(1) + body + m.group(2), text)
            updated.append(name)
    readme.write_text(text)
    return updated


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    print("updated blocks:", ", ".join(update(args.readme, args.results)) or "none")


if __name__ == "__main__":
    main()
