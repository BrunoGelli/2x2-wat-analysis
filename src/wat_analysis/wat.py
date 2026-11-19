"""Offline WAT catalog and exposure-matched recovery measurements."""

import csv
import gzip
import itertools
import json
import math
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .clocks import Session
from .geometry import Geometry
from .index import PERIOD, open_index
from .wat_detect import (
    TICKS_PER_MS,
    TICKS_PER_US,
    Plane,
    find_pulses,
    merge_intervals,
    usable_spans,
)

CLOCKS = ("tick", "receipt_tick")
CHIP_SLOTS = 8 * 256


@dataclass
class Settings:
    window: int = 3000
    fine: int = 100
    chip_fraction: float = 0.51
    tile_fraction: float = 0.51
    min_packets: int = 0
    prompt_fraction: float = 0.05
    pre: int = 200_000
    guard: int = 20_000
    post: int = 200_000
    bin: int = 2500
    train_gap: int = 100_000
    veto: tuple = ()
    veto_left: int = -500
    veto_right: int = 5000
    sync_guard: int = 300
    hold: int = 5000
    chip_probe: tuple = (10_000, 50_000)
    drift_z: float = 5
    min_expected: float = 5
    bootstrap: int = 200
    seed: int = 73
    controls: int = 8
    examples: int = 3
    fit: bool = False
    iogs: tuple = tuple(range(1, 9))


def add_arguments(p):
    p.add_argument("files", nargs="+", type=Path, help="Closed packet HDF5 files")
    p.add_argument(
        "--output", type=Path, required=True, help="New or empty output directory"
    )
    p.add_argument(
        "--iog", nargs="+", type=int, choices=range(1, 9), default=range(1, 9)
    )
    p.add_argument(
        "--window-us", type=float, default=300, help="Sliding WAT detection width (300)"
    )
    p.add_argument(
        "--fine-us", type=float, default=10, help="Bin width for prompt edges (10)"
    )
    p.add_argument(
        "--min-chip-fraction",
        type=float,
        default=0.51,
        help="Minimum fraction of all geometrically mapped chips on the anode (0.51)",
    )
    p.add_argument(
        "--min-tile-fraction",
        type=float,
        default=0.51,
        help="Minimum fraction of geometrically mapped tiles hit (0.51)",
    )
    p.add_argument(
        "--min-packets",
        type=int,
        default=0,
        help="Optional additional sliding count cut",
    )
    p.add_argument(
        "--prompt-fraction",
        type=float,
        default=0.05,
        help="First/last fine bins reaching this fraction of the peak define the prompt (0.05)",
    )
    p.add_argument(
        "--baseline-ms",
        type=float,
        nargs=2,
        default=(20, 2),
        metavar=("FAR", "NEAR"),
        help="Clean baseline before the first pulse of each train (20 2)",
    )
    p.add_argument(
        "--baseline-drift-z",
        type=float,
        default=5,
        help="Exclude trains and controls with anode pre-window half-count imbalance above this score (5; 0 disables)",
    )
    p.add_argument("--post-ms", type=float, default=20)
    p.add_argument("--bin-ms", type=float, default=0.25)
    p.add_argument(
        "--train-gap-ms",
        type=float,
        default=10,
        help="Maximum end-to-onset gap within a train",
    )
    p.add_argument(
        "--veto-trigger",
        action="append",
        default=[],
        metavar="IOG[:SUBTYPE]",
        help="Type-7 source defining a physics veto. Repeat to combine sources.",
    )
    p.add_argument(
        "--veto-us",
        type=float,
        nargs=2,
        default=(-50, 500),
        metavar=("START", "STOP"),
        help="Excluded interval relative to each selected trigger (-50 500)",
    )
    p.add_argument(
        "--sync-guard-us",
        type=float,
        default=30,
        help="Symmetric SYNC-83 exclusion before finding or stacking pulses (30)",
    )
    p.add_argument(
        "--hold-ms", type=float, default=0.5, help="Sustained return required for t90"
    )
    p.add_argument(
        "--min-expected",
        type=float,
        default=5,
        help="Minimum expected packets/bin for t90",
    )
    p.add_argument(
        "--bootstrap",
        type=int,
        default=200,
        help="Train resamples for stacked 95%% intervals; 0 disables",
    )
    p.add_argument(
        "--chip-probe-ms",
        type=float,
        nargs=2,
        default=(1, 5),
        metavar=("START", "STOP"),
        help="Compare distinct reporting chips in this post interval with equal-length pre intervals (1 5)",
    )
    p.add_argument("--seed", type=int, default=73)
    p.add_argument(
        "--controls-per-train",
        type=int,
        default=8,
        help="Quiet reference times per usable train (8)",
    )
    p.add_argument(
        "--examples",
        type=int,
        default=3,
        help="Maximum number of real pulses shown in detail",
    )
    p.add_argument(
        "--fit",
        action="store_true",
        help="Fit a signed exponential departure from baseline where supported",
    )


def settings(a):
    def ticks(value, unit):
        if not math.isfinite(value):
            raise ValueError("All time settings must be finite.")
        return round(value * unit)

    veto = []
    for spec in a.veto_trigger:
        pieces = spec.split(":")
        if len(pieces) > 2 or not all(s.isdigit() for s in pieces):
            raise ValueError("--veto-trigger expects IOG or IOG:SUBTYPE.")
        iog = int(pieces[0])
        subtype = int(pieces[1]) if len(pieces) == 2 else None
        if iog not in range(1, 9) or (
            subtype is not None and subtype not in range(256)
        ):
            raise ValueError("Invalid trigger IOG or subtype.")
        veto.append((iog, subtype))
    cfg = Settings(
        window=ticks(a.window_us, TICKS_PER_US),
        fine=ticks(a.fine_us, TICKS_PER_US),
        chip_fraction=a.min_chip_fraction,
        tile_fraction=a.min_tile_fraction,
        min_packets=a.min_packets,
        prompt_fraction=a.prompt_fraction,
        pre=ticks(a.baseline_ms[0], TICKS_PER_MS),
        guard=ticks(a.baseline_ms[1], TICKS_PER_MS),
        post=ticks(a.post_ms, TICKS_PER_MS),
        bin=ticks(a.bin_ms, TICKS_PER_MS),
        train_gap=ticks(a.train_gap_ms, TICKS_PER_MS),
        veto=tuple(dict.fromkeys(veto)),
        veto_left=ticks(a.veto_us[0], TICKS_PER_US),
        veto_right=ticks(a.veto_us[1], TICKS_PER_US),
        sync_guard=ticks(a.sync_guard_us, TICKS_PER_US),
        hold=ticks(a.hold_ms, TICKS_PER_MS),
        chip_probe=tuple(ticks(x, TICKS_PER_MS) for x in a.chip_probe_ms),
        drift_z=a.baseline_drift_z,
        min_expected=a.min_expected,
        bootstrap=a.bootstrap,
        seed=a.seed,
        examples=a.examples,
        controls=a.controls_per_train,
        fit=a.fit,
        iogs=tuple(sorted(set(a.iog))),
    )
    if not all(
        0 < x <= 1 for x in (cfg.chip_fraction, cfg.tile_fraction, cfg.prompt_fraction)
    ):
        raise ValueError("Footprint and prompt fractions must be in (0, 1].")
    if not (
        0 < cfg.fine <= cfg.window
        and cfg.pre > cfg.guard >= 0
        and 0 < cfg.bin <= cfg.post
        and cfg.post % cfg.bin == 0
        and 0 < cfg.hold <= cfg.post
    ):
        raise ValueError(
            "Check time windows. Post time must be a whole number of recovery bins."
        )
    if (
        min(
            cfg.min_packets,
            cfg.sync_guard,
            cfg.train_gap,
            cfg.examples,
            cfg.seed,
            cfg.controls,
        )
        < 0
        or not math.isfinite(cfg.drift_z)
        or cfg.drift_z < 0
        or not 0 <= cfg.chip_probe[0] < cfg.chip_probe[1]
        or cfg.veto_left >= cfg.veto_right
        or not math.isfinite(cfg.min_expected)
        or cfg.min_expected <= 0
        or cfg.bootstrap < 0
        or cfg.bootstrap == 1
    ):
        raise ValueError("Invalid count, veto, bootstrap, or recovery setting.")
    return cfg


class Table:
    """Write CSV incrementally, including the header when the selection is empty."""

    def __init__(self, directory, name, fields, compressed=False):
        path = directory / (name + (".csv.gz" if compressed else ".csv"))
        opener = gzip.open if compressed else open
        self.file = opener(path, "wt", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=fields.split())
        self.writer.writeheader()

    def row(self, **kw):
        self.writer.writerow(kw)


def marker_lanes(view):
    lanes = defaultdict(list)
    for run in view.runs["marks"]:
        for m in run.data:
            lanes[(run.iog, int(m["kind"]), int(m["subtype"]))].append(
                int(m["tick"]) + run.shift
            )
    return {k: np.asarray(sorted(v), dtype=np.int64) for k, v in lanes.items()}


def distance(t, marks):
    """Signed peak minus nearest marker, in microseconds."""
    if not len(marks):
        return None
    i = marks.searchsorted(t)
    near = min(marks[max(0, i - 1) : i + 1], key=lambda x: abs(t - x))
    return (t - int(near)) / TICKS_PER_US


def has_overlap(lo, hi, intervals):
    return any(left < hi and lo < right for left, right in intervals)


def between_files(view):
    # Keep clean parts of partially overlapping files. Neither copy supplies
    # exposure in their common interval; duplicate resolved paths are removed upstream.
    holes = defaultdict(list)
    for g in range(1, 9):
        spans = sorted(
            (s["lo"], s["hi"], s["file"]) for s in view.coverage if s["iog"] == g
        )
        active = []
        for lo, hi, file in spans:
            active = [(end, other) for end, other in active if end > lo]
            for end, other in active:
                overlap = (lo + view.origin, min(hi, end) + view.origin)
                holes[file, g].append(overlap)
                holes[other, g].append(overlap)
            active.append((hi, file))
    return holes


def control_exposure(spans, pulses, cfg):
    # Sample uniformly in the remaining live start-time exposure. Locations
    # are chosen without examining the rate at either side of a control.
    blocked = [(p.start - cfg.post + 1, p.end + cfg.pre) for p in pulses]
    allowed = []
    for lo, hi in spans:
        allowed.extend(usable_spans(lo + cfg.pre, hi - cfg.post + 1, blocked))
    return allowed, np.cumsum([b - a for a, b in allowed], dtype=np.int64)


def choose_control(allowed, lengths, rng):
    pick = int(rng.integers(lengths[-1]))
    i = int(lengths.searchsorted(pick, side="right"))
    return allowed[i][0] + pick - (int(lengths[i - 1]) if i else 0)


def matched_fifo(table, ids, rng, plane, prompt, pre, post, exposure):
    """Pair full-reporting and other reporting chips with identical tile and prompt load."""
    load = count_chips(plane, prompt)
    for field in ("local_fifo", "shared_fifo"):
        full = count_chips(plane, prompt[prompt[field] == 3]) > 0
        for tile in np.unique(np.flatnonzero(full) // 256):
            active = np.flatnonzero(load[tile * 256 : (tile + 1) * 256]) + tile * 256
            yes, no = active[full[active]], active[~full[active]]
            pairs = []
            for n in np.intersect1d(load[yes], load[no]):
                y, z = yes[load[yes] == n], no[load[no] == n]
                size = min(len(y), len(z))
                pairs.extend(
                    zip(
                        rng.choice(y, size, replace=False),
                        rng.choice(z, size, replace=False),
                    )
                )
            if pairs:
                y, z = np.array(pairs).T
                table.row(
                    **ids,
                    tile=tile + 1,
                    fifo=field,
                    chip_pairs=len(y),
                    prompt_each=load[y].sum(),
                    pre_full=pre[y].sum(),
                    post_full=post[y].sum(),
                    pre_other=pre[z].sum(),
                    post_other=post[z].sum(),
                    post_ticks=exposure,
                )


def tile_totals(chips):
    tiles = chips.reshape(8, 256).sum(axis=1)
    return np.r_[tiles.sum(), tiles]


def chip_activity(table, plane, anchor, baseline_end, cfg, ids):
    # Match window lengths. Distinct chips cannot be scaled linearly from an
    # 18 ms pre-window to a 4 ms observation.
    left, right = cfg.chip_probe
    width = right - left
    nref = (cfg.pre - cfg.guard) // width
    if not nref:
        return
    lo = baseline_end - width * nref
    before = plane.rows(lo, baseline_end)
    slots = (before["tick"] + plane.shift - lo) // width
    pairs = np.unique(slots * CHIP_SLOTS + plane.chips(before))
    expected = tile_totals(np.bincount(pairs % CHIP_SLOTS, minlength=CHIP_SLOTS)) / nref
    after = plane.rows(anchor + left, anchor + right)
    reported = np.unique(plane.chips(after))
    observed = tile_totals(np.bincount(reported, minlength=CHIP_SLOTS))
    for tile in range(9):
        table.row(
            **ids,
            tile=tile,
            reference_windows=nref,
            expected_chips=expected[tile],
            observed_chips=observed[tile],
        )


def count_chips(plane, rows):
    return np.bincount(plane.chips(rows), minlength=CHIP_SLOTS)


def baseline(plane, start, cfg):
    lo, hi = start - cfg.pre, start - cfg.guard
    pre, early = [], []
    for clock in CLOCKS:
        rows = plane.rows(lo, hi, clock)
        pre.append(count_chips(plane, rows))
        early.append(
            count_chips(plane, rows[rows[clock] + plane.shift < (lo + hi) // 2])
        )
    return np.stack(pre), np.stack(early)


def baseline_drift(before, early, cfg):
    # Conditional on the total, a stationary Poisson rate gives a binomial
    # split. Use this as a configurable imbalance score, not a fitted p-value.
    n = int(before[0].sum())
    width = cfg.pre - cfg.guard
    p = (width // 2) / width
    if not n or p == 0:
        return 0.0
    return (p * n - int(early[0].sum())) / np.sqrt(n * p * (1 - p))


def count_post(plane, anchor, n, cfg):
    # A censored bin is absent. A fully exposed bin with no packets is zero.
    observed = np.zeros((2, n, 9), dtype=np.int64)
    chip_counts = np.zeros(CHIP_SLOTS, dtype=np.int64)
    for i, clock in enumerate(CLOCKS):
        rows = plane.rows(anchor, anchor + n * cfg.bin, clock)
        chips = plane.chips(rows)
        slots = (rows[clock] + plane.shift - anchor) // cfg.bin
        binned = np.bincount(slots * 8 + chips // 256, minlength=n * 8).reshape(n, 8)
        observed[i, :, 0] = binned.sum(axis=1)
        observed[i, :, 1:] = binned
        if i == 0:
            chip_counts = np.bincount(chips, minlength=CHIP_SLOTS)
    return observed, chip_counts


@dataclass
class Observation:
    file: int
    iog: int
    train: str
    kind: str
    event: str
    load: np.ndarray
    before: np.ndarray  # clock, anode + 8 tiles
    after: np.ndarray  # clock, fully exposed time bins, anode + 8 tiles


def split_trains(pulses, gap, breaks=()):
    trains = []
    for p in pulses:
        if (
            not trains
            or p.start - trains[-1][-1].end > gap
            or has_overlap(trains[-1][-1].start, p.end, breaks)
        ):
            trains.append([])
        trains[-1].append(p)
    return trains


def take_example(plane, p, before, early, cfg, info, limit):
    pad = max(cfg.fine * 4, cfg.window // 2)
    edges = np.arange(p.start - pad, p.end + pad + cfg.fine, cfg.fine)
    adc = plane.rows(int(edges[0]), int(edges[-1]))
    fine_counts = np.histogram(adc["tick"] + plane.shift, edges)[0]
    # Explicitly aligned to onset as an independent check on prompt-edge selection.
    wide_edges = p.start + np.arange(-cfg.pre, cfg.post + cfg.bin, cfg.bin)
    wide = plane.rows(int(wide_edges[0]), int(wide_edges[-1]))
    wide_counts = np.histogram(wide["tick"] + plane.shift, wide_edges)[0]
    return dict(
        **info,
        fine_us=((edges - p.start) / TICKS_PER_US).tolist(),
        fine_counts=fine_counts.tolist(),
        wide_ms=((wide_edges - p.start) / TICKS_PER_MS).tolist(),
        wide_counts=wide_counts.tolist(),
        duration_us=(p.end - p.start) / TICKS_PER_US,
        baseline_hz=float(before[0].sum() * PERIOD / (cfg.pre - cfg.guard)),
        baseline_first_half=int(early[0].sum()),
        live_post_ms=(limit - p.end) / TICKS_PER_MS,
    )


def process_plane(plane, cfg, context, tables, rng):
    file, name, view, aligned, spans, lanes, overlaps, selection_origin = context
    iog = plane.iog
    sync = lanes.get((iog, 6, 83), ())
    triggers = {k: t for k, t in lanes.items() if k[1] == 7}
    physics, missing = [], []
    required = {iog}
    for source, subtype in cfg.veto:
        required.add(source)
        matches = [
            t
            for (g, _, s), t in triggers.items()
            if g == source and (subtype is None or s == subtype)
        ]
        if not matches:
            missing.append((source, subtype))
        for times in matches:
            physics.extend(
                (int(t) + cfg.veto_left, int(t) + cfg.veto_right) for t in times
            )
    physics = merge_intervals(physics)
    sync_masks = (
        [(int(t) - cfg.sync_guard, int(t) + cfg.sync_guard + 1) for t in sync]
        if cfg.sync_guard
        else []
    )
    holes = [
        (a - view.origin, b - view.origin)
        for g in required
        for a, b in overlaps.get((file, g), ())
    ]
    for g in required:
        resets = lanes.get((g, 6, 83), ())
        holes.extend(
            (int(a), int(b)) for a, b in itertools.pairwise(resets) if b - a > PERIOD
        )
    covered = [spans[g] for g in required if g in spans]
    lo, hi = max(a for a, _ in covered), min(b for _, b in covered)
    masks = merge_intervals(physics + sync_masks + holes)
    live = usable_spans(lo, hi, masks) if aligned and not missing else []
    pulses = find_pulses(plane, cfg, sync_masks)
    counts = Counter(
        candidates=len(pulses),
        measured_pulses=0,
        sync83_windows=len(sync_masks),
        sync83_excluded_packets=sum(
            len(d)
            for a, b in sync_masks
            for d in plane.parts(max(a, plane.lo), min(b, plane.hi))
            if a < plane.hi and b > plane.lo
        ),
    )
    plane.hit_mask_edges = np.asarray(masks, dtype=np.int64).ravel()
    allowed, lengths = control_exposure(live, pulses, cfg)
    fifo_rng = np.random.default_rng([cfg.seed, file, iog])
    observations, examples = [], []
    trains = split_trains(pulses, cfg.train_gap, masks)
    previous_end = None
    for it, train in enumerate(trains):
        train_id = f"f{file}-g{iog}-t{it + 1}"
        first, last = train[0], train[-1]
        span = next(
            ((a, b) for a, b in live if a <= first.start - cfg.pre and last.end <= b),
            None,
        )
        clean = span is not None and (
            previous_end is None or previous_end <= first.start - cfg.pre
        )
        previous_end = last.end
        drift = None
        if clean:
            before, early = baseline(plane, first.start, cfg)
            drift = baseline_drift(before, early, cfg)
            if cfg.drift_z and abs(drift) > cfg.drift_z:
                counts["baseline_drift_trains"] += 1
                clean = False
        if clean:
            early_tiles = tile_totals(early[0])
            base_tiles = np.stack([tile_totals(b) for b in before])
            train_load = np.zeros(9, dtype=np.int64)
            counts["clean_trains"] += 1
        for j, p in enumerate(train):
            event = f"{train_id}-p{j + 1}"
            status = "trigger_veto_pass" if cfg.veto else "unvetoed"
            if not aligned:
                status = "unqualified_clock"
            elif missing:
                status = "missing_veto_lane"
            elif p.start < lo or p.end > hi or has_overlap(p.start, p.end, holes):
                status = "ambiguous_coverage"
            elif has_overlap(p.start, p.end, physics):
                status = "trigger_coincident"
            elif has_overlap(p.start, p.end, sync_masks):
                status = "sync_coincident"
            counts[status] += 1
            markers = {
                f"{g}:{s}": distance(p.peak, t) for (g, _, s), t in triggers.items()
            }
            limit = p.end
            if clean:
                nxt = (
                    train[j + 1].start
                    if j + 1 < len(train)
                    else (trains[it + 1][0].start if it + 1 < len(trains) else span[1])
                )
                limit = min(p.end + cfg.post, span[1], nxt)
            n = (limit - p.end) // cfg.bin
            prompt = plane.rows(p.start, p.end) if clean else None
            lag = (
                np.percentile(
                    (prompt["receipt_tick"] - prompt["tick"]) / TICKS_PER_MS,
                    [50, 90, 99],
                )
                if clean
                else [None] * 3
            )
            counts["measured_pulses"] += int(n > 0)
            tables["events"].row(
                file=file,
                filename=name,
                iog=iog,
                train=train_id,
                event=event,
                pulse_in_train=j + 1,
                train_size=len(train),
                clock="aligned" if aligned else "file-local",
                file_tick=p.start,
                selection_tick=(view.origin + p.start - selection_origin)
                if aligned
                else None,
                absolute_tick=(view.origin + p.start) if aligned else None,
                end_file_tick=p.end,
                peak_file_tick=p.peak,
                peak_packets=p.peak_count,
                packets=p.packets,
                chips=p.chips,
                mapped_chips=int(plane.mapped.sum()),
                tiles=p.tiles,
                pixels=p.pixels,
                receipt_lag_p50_ms=lag[0],
                receipt_lag_p90_ms=lag[1],
                receipt_lag_p99_ms=lag[2],
                sync83_peak_offset_us=distance(p.peak, sync),
                trigger_peak_offsets_us=json.dumps(markers),
                status=status,
                clean_baseline=int(clean),
                baseline_drift_z=drift,
                baseline_start_file_tick=first.start - cfg.pre if clean else None,
                baseline_end_file_tick=first.start - cfg.guard if clean else None,
                post_bins=n,
            )
            if not clean:
                continue
            load_chips = count_chips(plane, prompt)
            load = tile_totals(load_chips)
            train_load += load
            observed, post_chips = count_post(plane, p.end, n, cfg)
            local = count_chips(plane, prompt[prompt["local_fifo"] == 3])
            shared = count_chips(plane, prompt[prompt["shared_fifo"] == 3])
            if limit - p.end >= cfg.chip_probe[1]:
                chip_activity(
                    tables["reporting_chips"],
                    plane,
                    p.end,
                    first.start - cfg.guard,
                    cfg,
                    {"file": file, "iog": iog, "sample": "pulse", "event": event},
                )
            if n:
                matched_fifo(
                    tables["fifo_matched"],
                    {"file": file, "iog": iog, "event": event},
                    fifo_rng,
                    plane,
                    prompt,
                    before[0],
                    post_chips,
                    n * cfg.bin,
                )
            for c in np.flatnonzero(plane.mapped):
                tables["chips"].writer.writerow(
                    {
                        "file": file,
                        "iog": iog,
                        "event": event,
                        "tile": c // 256 + 1,
                        "chip_id": c % 256,
                        "prompt": load_chips[c],
                        "local_full": local[c],
                        "shared_full": shared[c],
                        "pre": before[0, c],
                        "post": post_chips[c],
                        "post_ticks": n * cfg.bin,
                    }
                )
            for tile in range(9):
                tables["event_tiles"].row(
                    file=file,
                    iog=iog,
                    event=event,
                    tile=tile,
                    prompt=load[tile],
                    pre_early=early_tiles[tile],
                    pre_late=base_tiles[0, tile] - early_tiles[tile],
                    receipt_pre=base_tiles[1, tile],
                    post=observed[0, :, tile].sum(),
                    expected=base_tiles[0, tile] * n * cfg.bin / (cfg.pre - cfg.guard),
                    post_ticks=n * cfg.bin,
                )
            bin_writer = tables["event_bins"].file
            for k in range(n):
                # Numeric counts and internal event IDs only; write nine regions in one pass.
                bin_writer.writelines(
                    f"{file},{iog},{event},{tile},{k},{observed[0, k, tile]},{observed[1, k, tile]}\n"
                    for tile in range(9)
                )
            if n:
                obs = Observation(
                    file, iog, train_id, "pulse", event, load, base_tiles, observed
                )
                observations.append(obs)
                onset_bins = min(cfg.post, limit - p.start) // cfg.bin
                onset_obs, _ = count_post(plane, p.start, onset_bins, cfg)
                observations.append(
                    Observation(
                        file, iog, train_id, "onset", event, load, base_tiles, onset_obs
                    )
                )
                if j == len(train) - 1:
                    observations.append(
                        Observation(
                            file,
                            iog,
                            train_id,
                            "train_end",
                            event,
                            train_load,
                            base_tiles,
                            observed,
                        )
                    )
                if cfg.examples:
                    examples.append(
                        (
                            p.packets,
                            p,
                            before,
                            early,
                            limit,
                            {
                                "file": file,
                                "filename": name,
                                "iog": iog,
                                "event": event,
                                "file_tick": p.start,
                                "packets": p.packets,
                                "chips": p.chips,
                                "tiles": p.tiles,
                            },
                        )
                    )
        if clean and len(allowed):
            for ref in range(cfg.controls):
                t = choose_control(allowed, lengths, rng)
                ctrl, half = baseline(plane, t, cfg)
                if cfg.drift_z and abs(baseline_drift(ctrl, half, cfg)) > cfg.drift_z:
                    counts["baseline_drift_controls"] += 1
                    continue
                if cfg.post >= cfg.chip_probe[1]:
                    chip_activity(
                        tables["reporting_chips"],
                        plane,
                        t,
                        t - cfg.guard,
                        cfg,
                        {
                            "file": file,
                            "iog": iog,
                            "sample": "control",
                            "event": f"{train_id}-c{ref + 1}",
                        },
                    )
                observed, _ = count_post(plane, t, cfg.post // cfg.bin, cfg)
                observations.append(
                    Observation(
                        file,
                        iog,
                        train_id,
                        "control",
                        f"{train_id}-c{ref + 1}",
                        np.zeros(9, dtype=np.int64),
                        np.stack([tile_totals(b) for b in ctrl]),
                        observed,
                    )
                )
                tables["controls"].row(
                    file=file, iog=iog, train=train_id, reference=ref + 1, file_tick=t
                )
                counts["controls"] += 1
    # Keep a bounded selection from across the load range for plotting.
    examples.sort(key=lambda x: x[0])
    selected = np.unique(
        np.linspace(0, len(examples) - 1, min(len(examples), cfg.examples), dtype=int)
    )
    examples = [
        take_example(plane, p, b, e, cfg, info, lim)
        for idx in selected
        for _, p, b, e, lim, info in [examples[idx]]
    ]
    return observations, examples, dict(counts)


def run(files, cache, output, cfg):
    from .wat_report import summarize, write_pdf

    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Choose a new or empty output directory.")
    output.parent.mkdir(parents=True, exist_ok=True)
    geom = Geometry()
    paths = dict.fromkeys(Path(path).expanduser().resolve() for path in files)
    indices = [open_index(path, cache, geom) for path in paths]
    selection = Session(indices).views.get("aligned")
    overlaps = between_files(selection) if selection is not None else {}
    origin = selection.origin if selection is not None else None
    metadata = {
        "settings": asdict(cfg),
        "ticks_per_second": PERIOD,
        "selection_origin": origin,
        "files": [],
    }
    observations, examples = [], []
    rng = np.random.default_rng(cfg.seed)
    # A failed analysis leaves no partially published results.
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".wat-") as tmp:
        directory = Path(tmp) / "results"
        directory.mkdir()
        tables = {}
        try:
            for key, fields in (
                (
                    "events",
                    "file filename iog train event pulse_in_train train_size clock file_tick selection_tick absolute_tick end_file_tick peak_file_tick peak_packets packets chips mapped_chips tiles pixels receipt_lag_p50_ms receipt_lag_p90_ms receipt_lag_p99_ms sync83_peak_offset_us trigger_peak_offsets_us status clean_baseline baseline_drift_z baseline_start_file_tick baseline_end_file_tick post_bins",
                ),
                (
                    "event_tiles",
                    "file iog event tile prompt pre_early pre_late receipt_pre post expected post_ticks",
                ),
                (
                    "chips",
                    "file iog event tile chip_id prompt local_full shared_full pre post post_ticks",
                ),
                ("controls", "file iog train reference file_tick"),
                (
                    "reporting_chips",
                    "file iog sample event tile reference_windows expected_chips observed_chips",
                ),
                (
                    "fifo_matched",
                    "file iog event tile fifo chip_pairs prompt_each pre_full post_full pre_other post_other post_ticks",
                ),
                ("event_bins", "file iog event tile bin asic receipt"),
            ):
                tables[key] = Table(
                    directory,
                    key,
                    fields,
                    compressed=key in ("chips", "event_bins"),
                )
            for file, index in enumerate(indices):
                session = Session([index])
                info = {
                    "name": index.path.name,
                    "path": str(index.path),
                    "rows": index.meta["rows"],
                    "clocks": index.meta["clocks"],
                    "planes": {},
                    "trigger_lanes": {},
                }
                metadata["files"].append(info)
                aligned = session.views.get("aligned")
                views = [("aligned", aligned)] if aligned is not None else []
                measured_iogs = (
                    {s["iog"] for s in aligned.coverage}
                    if aligned is not None
                    else set()
                )
                for key, (_, g) in session.views.local.items():
                    if g not in measured_iogs and g in cfg.iogs:
                        views.append((key, session.views[key]))
                for key, view in views:
                    lanes = marker_lanes(view)
                    spans = {s["iog"]: (s["lo"], s["hi"]) for s in view.coverage}
                    # Late pre-reset hits are valid for display but do not establish
                    # continuous acquisition before the first recorded reset.
                    for g, (left, right) in spans.items():
                        sync = lanes.get((g, 6, 83))
                        if sync is not None:
                            spans[g] = (max(left, int(sync[0])), right)
                    info["trigger_lanes"][key] = {
                        f"{g}:{s}": len(t) for (g, k, s), t in lanes.items() if k == 7
                    }
                    for g in cfg.iogs:
                        if g not in spans:
                            continue
                        runs = [r for r in view.runs["hits"] if r.iog == g]
                        plane = Plane(runs, geom, g, spans[g])
                        context = (
                            file,
                            index.path.name,
                            view,
                            key == "aligned",
                            spans,
                            lanes,
                            overlaps,
                            origin,
                        )
                        obs, ex, counts = process_plane(
                            plane, cfg, context, tables, rng
                        )
                        info["planes"][g] = dict(
                            clock=key, origin=view.origin, coverage=spans[g], **counts
                        )
                        observations.extend(obs)
                        examples.extend(ex)
                        print(
                            f"{index.path.name}  IOG {g}  {counts['candidates']} candidates, "
                            f"{counts['measured_pulses']} usable pulses",
                            flush=True,
                        )
        finally:
            for table in tables.values():
                table.file.close()
        curves = summarize(directory, observations, cfg, len(indices))
        write_pdf(directory, metadata, curves, examples, cfg)
        (directory / "analysis.json").write_text(json.dumps(metadata, indent=2) + "\n")
        if output.is_dir():
            output.rmdir()  # Only an empty directory can be replaced on every platform.
        directory.rename(output)
    print(f"Report: {output / 'wat-recovery.pdf'}", flush=True)
