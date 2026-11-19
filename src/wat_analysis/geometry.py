"""Map tile electronics to the pinned PacMon anode raster (row zero at top)."""

import json
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent / "data"


class Geometry:
    def __init__(self):
        self.tiles = json.loads((DATA / "tiles.json").read_text())
        with np.load(DATA / "channels.npz", allow_pickle=False) as archive:
            templates = archive["templates"]
        self.size = sum(t["width"] * t["height"] for t in self.tiles)
        self.lut = np.full((9, 9, 256, 64), -1, dtype=np.int32)
        self.chips = np.empty(self.size, dtype=np.int32)
        for t in self.tiles:
            iog, tile = t["iog"], t["tile"]
            template = templates[t["template"]]
            chip, chan = np.nonzero(template >= 0)
            pixels = template[chip, chan].astype(np.int32) + t["start"]
            self.lut[iog, tile, chip, chan] = pixels
            self.chips[pixels] = ((iog - 1) * 8 + tile - 1) * 256 + chip

    def lookup(self, iog, data):
        io = data["io_channel"].astype(np.int16)
        tile = (io - 1) // 4 + 1
        chan = data["channel_id"]
        valid = (io >= 1) & (io <= 32) & (chan < 64)
        pixels = np.full(len(io), -1, dtype=np.int32)
        pixels[valid] = self.lut[iog, tile[valid], data["chip_id"][valid], chan[valid]]
        return pixels
