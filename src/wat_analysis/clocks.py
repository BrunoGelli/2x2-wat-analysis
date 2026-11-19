"""Group indexed runs into file-local or PPS-aligned detector-time views."""

from dataclasses import dataclass

import numpy as np

from .index import PERIOD


@dataclass
class Run:
    file: int
    iog: int
    shift: int
    lo: int
    hi: int  # exclusive
    data: np.ndarray

    def cut(self, lo, hi):
        bounds = self.data["tick"].searchsorted([lo - self.shift, hi - self.shift])
        return self.data[bounds[0] : bounds[1]]


class View:
    def __init__(self, files, use):
        self.runs = {"hits": [], "marks": []}
        self.coverage = []
        spans = []
        for file in sorted({f for f, _ in use}):
            index = files[file]
            for name in self.runs:
                for iog, start, size, lo, last in index.meta[name]:
                    offset = use.get((file, iog))
                    if offset is not None:
                        spans.append((file, iog, name, start, size, lo, last, offset))
        if not spans:
            raise ValueError("No indexed hits or markers in this clock view.")
        self.origin = min(lo + offset for _, _, _, _, _, lo, _, offset in spans)
        coverage = {}
        for file, iog, name, start, size, lo, last, offset in spans:
            shift = offset - self.origin
            run = Run(
                file,
                iog,
                shift,
                lo + shift,
                last + shift + 1,
                files[file].records[name][start : start + size],
            )
            self.runs[name].append(run)
            key = (file, iog)
            left, right = coverage.get(key, (run.lo, run.hi))
            coverage[key] = min(left, run.lo), max(right, run.hi)
        self.coverage = [
            {"file": f, "iog": g, "lo": lo, "hi": hi}
            for (f, g), (lo, hi) in sorted(coverage.items())
        ]


class ClockViews(dict):
    def __init__(self, files):
        super().__init__()
        self.files = files
        self.local = {}

    def __missing__(self, key):
        if key not in self.local:
            raise KeyError("Unknown clock view.")
        view = View(self.files, {self.local[key]: 0})
        return self.setdefault(key, view)


class Session:
    def __init__(self, files):
        self.files = files
        self.views = ClockViews(files)
        offsets = {}
        for i, f in enumerate(files):
            active = {b[0] for name in ("hits", "marks") for b in f.meta[name]}
            for g in sorted(active):
                epoch = f.meta["clocks"][str(g)]["epoch"]
                if epoch is not None:
                    offsets[i, g] = epoch * PERIOD
                key = f"{i}-{g}"
                self.views.local[key] = (i, g)
        if offsets:
            self.views["aligned"] = View(files, offsets)
