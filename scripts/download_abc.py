"""Resumable ABC-dataset chunk downloader.

The ABC dataset (~1M CAD models) is hosted at NYU and downloaded per-chunk as 7z
archives. The per-chunk URL list is published on the dataset site:
    https://deep-geometry.github.io/abc-dataset/    (see the "Download" section)

Rather than hardcode URLs that drift, populate `scripts/abc_urls.txt` with the
OBJ + FEAT archive URLs for 2-3 chunks (~10K models) from that page, then run:

    python scripts/download_abc.py            # downloads (resumable) into data/abc_raw/
    python scripts/download_abc.py --extract  # extracts the 7z archives

You need `wget` and `7z` (`brew install wget p7zip`). We fetch OBJ (meshes) + FEAT
(per-face surface-type ground truth) — not STEP; the pipeline is mesh-first.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URLS = ROOT / "scripts" / "abc_urls.txt"
DEST = ROOT / "data" / "abc_raw"


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    if not URLS.exists():
        print(f"Create {URLS} with one archive URL per line (OBJ + FEAT chunks).")
        print("Get them from https://deep-geometry.github.io/abc-dataset/ -> Download.")
        sys.exit(1)
    urls = [u.strip() for u in URLS.read_text().splitlines() if u.strip() and not u.startswith("#")]
    if "--extract" in sys.argv:
        for arc in sorted(DEST.glob("*.7z")):
            print(f"extracting {arc.name}")
            subprocess.run(["7z", "x", "-y", str(arc), f"-o{DEST}"], check=False)
        return
    for u in urls:
        print(f"downloading {u}")
        subprocess.run(["wget", "-c", "-P", str(DEST), u], check=False)  # -c = resume
    print(f"\ndone -> {DEST}  (run with --extract next)")


if __name__ == "__main__":
    main()
