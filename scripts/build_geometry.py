"""Rebuild the bundled raster layout from four pinned PacMon v4 JSON files."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

BLOBS = (
    "c2e030bfb7d30b14b9d060acd0b830f17df8e714",
    "73ba6caa22622dc04e4cb092901b0b3879d80a61",
    "3ad1b36b4a9de2b5523520ea2b5f895f36fb987f",
    "afb4a3a9881909bc25a730765c4df944f026339a",
)


def build(source, dest):
    tiles, templates = [], []
    shapes = {}
    start = 0
    for module, expected in enumerate(BLOBS):
        raw = (source / f"geometry_mod{module}_v4.json").read_bytes()
        digest = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        if digest != expected:
            raise ValueError(f"Module {module} differs from the pinned PacMon layout")
        config = json.loads(raw)
        pitch = config["pixel_pitch"]
        groups = {}
        for key, coords in config["geometry"].items():
            side, tile, chip, channel = map(int, key.split("-"))
            groups.setdefault((side, tile), []).append((chip, channel, *coords))
        for (side, physical), values in sorted(groups.items()):
            tile = physical - (side - 1) * 8
            iog = 2 * module + side
            points = np.array(values)
            xy0 = points[:, 2:].min(axis=0)
            grid = (points[:, 2:] - xy0) / pitch
            xy = np.rint(grid).astype(int)
            if not np.allclose(grid, xy, atol=0.001, rtol=0):
                raise ValueError("Off-grid PacMon pixel")
            w, h = xy.max(axis=0) + 1
            cell = xy[:, 0] + w * (h - 1 - xy[:, 1])
            if len(np.unique(cell)) != len(cell) or len(cell) != w * h:
                raise ValueError("Expected one channel per raster cell")
            template = np.full((256, 64), -1, "<i2")
            template[points[:, 0].astype(int), points[:, 1].astype(int)] = cell
            key = template.tobytes()
            if key not in shapes:
                shapes[key] = len(templates)
                templates.append(template)
            tiles.append(
                {
                    "iog": iog,
                    "tile": tile,
                    "start": start,
                    "width": int(w),
                    "height": int(h),
                    "x": float(xy0[0]),
                    "y": float(xy0[1]),
                    "pitch": pitch,
                    "template": shapes[key],
                }
            )
            start += int(w * h)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "tiles.json").write_text(json.dumps(tiles, separators=(",", ":")) + "\n")
    np.savez_compressed(dest / "channels.npz", templates=np.array(templates))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    build(args.source, args.destination)
