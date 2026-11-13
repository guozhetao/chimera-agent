from __future__ import annotations

import argparse
import json
from pathlib import Path
from PIL import Image
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.figure_dir)
    report = []
    for stem in sorted({p.stem for p in root.glob("fig*.png")}):
        png = root / f"{stem}.png"
        pdf = root / f"{stem}.pdf"
        svg = root / f"{stem}.svg"
        image = Image.open(png)
        array = np.asarray(image.convert("L"))
        dpi = image.info.get("dpi", (np.nan, np.nan))
        svg_text = svg.read_text(encoding="utf-8") if svg.exists() else ""
        panel_labels = sorted(set(ch for ch in "abcdefghijk" if f">{ch}<</text>" in svg_text or f">{ch}</text>" in svg_text))
        report.append({
            "figure": stem,
            "formats": {"png": png.exists(), "pdf": pdf.exists(), "svg": svg.exists()},
            "pixel_width": image.width,
            "pixel_height": image.height,
            "dpi": dpi,
            "ink_fraction": float(np.mean(array < 250)),
            "panel_labels_detected_in_svg": panel_labels,
            "status": "PASS" if png.exists() and pdf.exists() and svg.exists() and image.width >= 2500 and np.mean(array < 250) > .03 else "FAIL",
        })
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
