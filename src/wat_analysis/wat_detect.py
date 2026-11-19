"""Find spatially broad pulses in unrolled ASIC time."""

from dataclasses import dataclass

import numpy as np

from .index import HIT, PERIOD

TICKS_PER_US = PERIOD // 1_000_000
TICKS_PER_MS = PERIOD // 1000


def merge_intervals(intervals):
    """Union of half-open integer intervals, including adjacent intervals."""
    result = []
    for lo, hi in sorted(intervals):
        if hi <= lo:
            continue
        if result and lo <= result[-1][1]:
            result[-1] = (result[-1][0], max(hi, result[-1][1]))
        else:
            result.append((lo, hi))
    return result


def usable_spans(lo, hi, masks):
    out = []
    for left, right in masks:
        if right <= lo:
            continue
        if hi <= left:
            break
        if lo < left:
            out.append((lo, left))
        lo = max(lo, right)
    if lo < hi:
        out.append((lo, hi))
    return out


def union_windows(ends, width):
    if not len(ends):
        return []
    groups = np.split(ends, np.flatnonzero(np.diff(ends) > width) + 1)
    return [(int(group[0]) - width + 1, int(group[-1]) + 1) for group in groups]


class Plane:
    """One indexed file and IOG. All public ranges are in detector-view ticks."""

    def __init__(self, runs, geometry, iog, span):
        self.runs = runs
        self.geometry = geometry
        self.iog = iog
        self.lo, self.hi = span
        self.shift = runs[0].shift if runs else 0
        self.mapped = (geometry.lut[iog, 1:] >= 0).any(axis=2).ravel()
        self.receipts = None
        self.hit_mask_edges = np.empty(0, dtype=np.int64)

    def chips(self, rows):
        # The geometry resolves alternate I/O routes to the same physical chip.
        return self.geometry.chips[rows["pixel"]] % (8 * 256)

    def parts(self, lo, hi, clock="tick"):
        if clock == "tick":
            for run in self.runs:
                if run.lo < hi and lo < run.hi:
                    yield run.cut(lo, hi)
            return
        if self.receipts is None:
            self.receipts = []
            for run in self.runs:
                receipt = run.data["receipt_tick"]
                lag = receipt - run.data["tick"]
                self.receipts.append(
                    (
                        int(receipt.min()) + run.shift,
                        int(receipt.max()) + run.shift + 1,
                        int(lag.min()),
                        int(lag.max()),
                        run,
                    )
                )
        for left, right, min_lag, max_lag, run in self.receipts:
            if left < hi and lo < right:
                rows = run.cut(lo - max_lag, hi - min_lag)
                t = rows["receipt_tick"] + run.shift
                yield rows[(t >= lo) & (t < hi)]

    def rows(self, lo, hi, clock="tick"):
        parts = list(self.parts(lo, hi, clock))
        records = np.concatenate(parts) if parts else np.empty(0, HIT)
        if len(self.hit_mask_edges) and len(records):
            # A packet originating in a veto stays excluded even when it is
            # read out long after the veto closes.
            t = records["tick"] + self.shift
            inside = np.searchsorted(self.hit_mask_edges, t, side="right") % 2
            records = records[inside == 0]
        return records


def distinct_in_window(keys, ends):
    """At each hit, count keys in the trailing window, without revisiting hits.

    Each hit keeps its key active through index ends[i] (exclusive).
    Union the intervals per key before counting the intervals that cover i.
    Both inputs are in ASIC time order, so ends are nondecreasing.
    """
    n = len(keys)
    order = np.argsort(keys, kind="stable")
    starts = order.copy()
    same = keys[order[1:]] == keys[order[:-1]]
    starts[1:][same] = np.maximum(starts[1:][same], ends[order[:-1][same]])
    delta = np.bincount(starts, minlength=n + 1) - np.bincount(ends, minlength=n + 1)
    return delta[:-1].cumsum()


def qualifying_windows(ticks, chips, width, min_hits, min_chips, min_tiles, lo, hi):
    """Every qualifying width-tick window ending on an occupied tick in [lo, hi)."""
    if len(ticks) < min_hits:
        return []
    order = np.argsort(ticks, kind="stable")
    t, c = ticks[order], chips[order]
    ends = t.searchsorted(t + width)
    counts = np.arange(1, len(t) + 1) - t.searchsorted(t - width + 1)
    keep = (t >= lo) & (t < hi) & (counts >= min_hits)
    keep &= distinct_in_window(c, ends) >= min_chips
    keep &= distinct_in_window(c // 256, ends) >= min_tiles
    # At a tied tick use the last row, which includes every simultaneous hit.
    keep[:-1] &= t[:-1] != t[1:]
    return union_windows(t[keep], width)


@dataclass
class Pulse:
    start: int
    end: int
    peak: int
    peak_count: int
    packets: int
    chips: int
    tiles: int
    pixels: int


def find_pulses(plane, cfg, excluded=()):
    """Exact sliding-window cut, followed by fine-bin prompt edge estimation."""
    width, fine = cfg.window, cfg.fine
    mapped = plane.mapped
    min_chips = max(1, int(np.ceil(cfg.chip_fraction * mapped.sum())))
    n_tiles = mapped.reshape(8, 256).any(axis=1).sum()
    min_tiles = max(1, int(np.ceil(cfg.tile_fraction * n_tiles)))
    min_hits = max(cfg.min_packets, min_chips, min_tiles)

    # Any window fits in two neighboring coarse bins. This upper bound only
    # skips impossible windows, independent of phase relative to bin edges.
    coarse, weights = [], []
    for rows in plane.parts(plane.lo, plane.hi):
        bins, n = np.unique((rows["tick"] + plane.shift) // width, return_counts=True)
        coarse.append(bins)
        weights.append(n)
    if not coarse:
        return []
    bins, inv = np.unique(np.concatenate(coarse), return_inverse=True)
    total = np.zeros(len(bins), dtype=np.int64)
    np.add.at(total, inv, np.concatenate(weights))
    pairs = total.copy()
    pairs[1:] += total[:-1] * (np.diff(bins) == 1)
    seeds = merge_intervals(
        [(int(b - 1) * width, int(b + 1) * width) for b in bins[pairs >= min_hits]]
    )
    windows = []
    # Split before the exact spatial test. A reset flash cannot supply chips
    # or packets to a nearby pulse or connect two otherwise distinct events.
    eligible = [
        span for left, right in seeds for span in usable_spans(left, right, excluded)
    ]
    for left, right in eligible:
        for lo in range(max(plane.lo, left), min(plane.hi, right), PERIOD // 4):
            hi = min(lo + PERIOD // 4, right, plane.hi)
            rows = plane.rows(max(plane.lo, left, lo - width + 1), hi)
            found = qualifying_windows(
                rows["tick"] + plane.shift,
                plane.chips(rows),
                width,
                min_hits,
                min_chips,
                min_tiles,
                lo,
                hi,
            )
            windows.extend((max(left, a), b) for a, b in found)
    pulses = []
    for left, right in merge_intervals(windows):
        left = max(left, plane.lo)
        rows = plane.rows(left, right)
        t = rows["tick"] + plane.shift
        bucket = (t - left) // fine
        counts = np.bincount(bucket)
        peak = int(counts.argmax())
        prompt = np.flatnonzero(counts >= max(1, cfg.prompt_fraction * counts[peak]))
        start = left + fine * int(prompt[0])
        end = min(right, left + fine * (int(prompt[-1]) + 1))
        kept = rows[(t >= start) & (t < end)]
        chips = np.unique(plane.chips(kept))
        pulses.append(
            Pulse(
                start,
                end,
                left + peak * fine,
                int(counts[peak]),
                len(kept),
                len(chips),
                len(np.unique(chips // 256)),
                len(np.unique(kept["pixel"])),
            )
        )
    return pulses
