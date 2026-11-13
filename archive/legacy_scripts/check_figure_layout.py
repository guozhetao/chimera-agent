from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from xml.etree import ElementTree


SVG = "{http://www.w3.org/2000/svg}"


def parse_style(style: str):
    return dict(item.split(":", 1) for item in style.split(";") if ":" in item)


def text_bbox(node):
    style = parse_style(node.attrib.get("style", ""))
    if "x" not in node.attrib or "y" not in node.attrib:
        return None
    x, y = float(node.attrib["x"]), float(node.attrib["y"])
    text = "".join(node.itertext()).strip()
    if not text:
        return None
    size = float(style.get("font-size", "6").removesuffix("px"))
    anchor = style.get("text-anchor", "start")
    width = size * 0.55 * len(text)
    height = size
    left = x - (width / 2 if anchor == "middle" else width if anchor == "end" else 0)
    bottom = y + height * 0.22
    top = bottom - height
    transform = node.attrib.get("transform", "")
    m = re.search(r"rotate\((-?\d+(?:\.\d+)?)", transform)
    angle = float(m.group(1)) if m else 0.0
    if angle:
        radians = math.radians(angle)
        corners = [(left, top), (left + width, top), (left + width, bottom), (left, bottom)]
        rotated = []
        for cx, cy in corners:
            dx, dy = cx - x, cy - y
            rotated.append((x + dx * math.cos(radians) - dy * math.sin(radians), y + dx * math.sin(radians) + dy * math.cos(radians)))
        xs = [p[0] for p in rotated]; ys = [p[1] for p in rotated]
        return (min(xs), min(ys), max(xs), max(ys), text, size)
    return (left, top, left + width, bottom, text, size)


def overlap(a, b):
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    inter = ix * iy
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(min(area_a, area_b), 1e-9)


def inspect_svg(path: Path):
    root = ElementTree.parse(path).getroot()
    width = float(root.attrib["width"].removesuffix("pt"))
    height = float(root.attrib["height"].removesuffix("pt"))
    boxes = [b for b in (text_bbox(node) for node in root.iter(f"{SVG}text")) if b]
    collisions = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            score = overlap(boxes[i], boxes[j])
            if score > 0.20:
                collisions.append({
                    "score": round(score, 3),
                    "text_1": boxes[i][4],
                    "text_2": boxes[j][4],
                })
    outside = [b[4] for b in boxes if b[0] < -1 or b[1] < -1 or b[2] > width + 1 or b[3] > height + 1]
    panels = sorted(set(text[4] for text in boxes if len(text[4]) == 1 and text[5] >= 8 and text[4] in "abcdefghijk"))
    return {
        "svg_width_pt": width,
        "svg_height_pt": height,
        "text_elements": len(boxes),
        "panel_labels": panels,
        "text_overlap_count": len(collisions),
        "text_overlaps": collisions[:25],
        "text_outside_canvas": outside,
        "status": "PASS" if not collisions and not outside else "FAIL",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = {}
    for path in sorted(Path(args.figure_dir).glob("*.svg")):
        report[path.stem] = inspect_svg(path)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
