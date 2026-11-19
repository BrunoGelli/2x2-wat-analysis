"""Build an immutable per-file index. Each run is sorted *after* clock unrolling."""

import hashlib
import json
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

PERIOD = 10_000_000
INDEX_VERSION = 2
CHUNK_ROWS = 131_072
HIT = np.dtype(
    [
        ("tick", "<i8"),
        ("receipt_tick", "<i8"),
        ("row", "<u8"),
        ("pixel", "<i4"),
        ("adc", "u1"),
        ("local_fifo", "u1"),
        ("shared_fifo", "u1"),
        ("flags", "u1"),
    ]
)
MARK = np.dtype([("tick", "<i8"), ("row", "<u8"), ("kind", "u1"), ("subtype", "u1")])
REQUIRED = {
    "io_group",
    "io_channel",
    "chip_id",
    "channel_id",
    "packet_type",
    "timestamp",
    "receipt_timestamp",
    "valid_parity",
    "dataword",
    "local_fifo",
    "shared_fifo",
    "trigger_type",
    "first_packet",
    "downstream_marker",
}


@dataclass
class Clock:
    offset: int = 0
    last_step: int = PERIOD
    label: int = 0
    ready: bool = False
    votes: Counter = field(default_factory=Counter)
    counts: Counter = field(default_factory=Counter)

    def decode(self, iog, rows, row_ids, geometry):
        kind, subtype = rows["packet_type"], rows["trigger_type"]
        raw = rows["timestamp"].astype(np.int64)
        sync = np.flatnonzero((kind == 6) & (subtype == 83))
        # A reset word carries the length of the previous interval, not a hit time.
        steps = ((raw[sync] + PERIOD // 2) // PERIOD) * PERIOD
        good = (steps > 0) & (np.abs(raw[sync] - steps) <= PERIOD // 20)
        self.counts["invalid_sync"] += int((~good).sum())
        sync, steps = sync[good], steps[good]
        offsets = np.r_[self.offset, self.offset + np.cumsum(steps, dtype=np.int64)]
        previous = np.r_[self.last_step, steps]
        headers = np.flatnonzero(kind == 4)
        labels = np.r_[self.label, raw[headers]]
        at_sync = labels[np.searchsorted(headers, sync)]
        for label, offset in zip(at_sync, offsets[1:]):
            if label > 0:
                self.votes[int(label) - int(offset) // PERIOD] += 1
            else:
                self.counts["unlabelled_sync"] += 1
        self.label = int(labels[-1])

        data_i = np.flatnonzero(kind == 0)
        d = rows[data_i]
        phase = raw[data_i]
        receipt = d["receipt_timestamp"].astype(np.int64)
        epoch = np.searchsorted(sync, data_i)
        ready = self.ready | (epoch > 0)
        tick = offsets[epoch] + phase % PERIOD - (receipt < phase) * previous[epoch]
        timing_ok = (
            (phase >= 0)
            & (phase < 2**31)
            & (tick >= 0)
            & (receipt < 32 * PERIOD)
            & (offsets[epoch] + receipt >= tick)
        )
        diag = (
            d["fifo_diagnostics_enabled"] != 0
            if "fifo_diagnostics_enabled" in d.dtype.names
            else np.zeros(len(d), bool)
        )
        normal = (d["valid_parity"] != 0) & ~diag
        pixel = geometry.lookup(iog, d)
        selected = normal & ready & timing_ok & (pixel >= 0)
        self.counts.update(
            charge=len(d),
            mapped=int(selected.sum()),
            bad_parity=int((d["valid_parity"] == 0).sum()),
            diagnostic=int((diag & (d["valid_parity"] != 0)).sum()),
            before_sync=int((normal & ~ready).sum()),
            invalid_time=int((normal & ready & ~timing_ok).sum()),
            unmapped=int((normal & ready & timing_ok & (pixel < 0)).sum()),
            resets=len(sync),
        )
        hits = np.empty(int(selected.sum()), HIT)
        hits["tick"], hits["row"], hits["pixel"] = (
            tick[selected],
            row_ids[data_i[selected]],
            pixel[selected],
        )
        chosen = d[selected]
        # PACMAN receipt belongs to the recorded reset epoch, including for
        # late hits whose ASIC tick was placed in the preceding interval.
        hits["receipt_tick"] = (offsets[epoch] + receipt)[selected]
        hits["adc"] = chosen["dataword"]
        hits["local_fifo"], hits["shared_fifo"] = (
            chosen["local_fifo"],
            chosen["shared_fifo"],
        )
        hits["flags"] = (
            (chosen["trigger_type"] & 3)
            | ((chosen["first_packet"] & 1) << 2)
            | ((chosen["downstream_marker"] & 1) << 3)
        )

        other = np.flatnonzero((kind == 7) | ((kind == 6) & (subtype != 83)))
        order = np.searchsorted(sync, other)
        eligible = (
            (self.ready | (order > 0)) & (raw[other] >= 0) & (raw[other] < PERIOD)
        )
        self.counts["unplaced_markers"] += int((~eligible).sum()) + int((~good).sum())
        other, order = other[eligible], order[eligible]
        idx = np.r_[sync, other]
        markers = np.empty(len(idx), MARK)
        markers["tick"] = np.r_[offsets[1:], offsets[order] + raw[other]]
        markers["row"], markers["kind"], markers["subtype"] = (
            row_ids[idx],
            kind[idx],
            subtype[idx],
        )
        if len(steps):
            self.ready, self.offset, self.last_step = (
                True,
                int(offsets[-1]),
                int(steps[-1]),
            )
        return hits, markers

    def summary(self):
        qualified = (
            len(self.votes) == 1
            and sum(self.votes.values()) >= 2
            and not self.counts["invalid_sync"]
        )
        epoch = next(iter(self.votes)) if qualified else None
        return {"counts": dict(self.counts), "epoch": epoch, "votes": dict(self.votes)}


def signature(path):
    stat = path.stat()
    return [str(path), stat.st_size, stat.st_mtime_ns, INDEX_VERSION]


def packet_dataset(f):
    ds = f.get("packets")
    if not isinstance(ds, h5py.Dataset) or ds.ndim != 1:
        raise ValueError("Select a packetized HDF5 file with a flat /packets dataset.")
    fields = set(ds.dtype.names or ())
    if REQUIRED - fields:
        raise ValueError(
            "Missing LArPix packet fields: " + ", ".join(sorted(REQUIRED - fields))
        )
    header = f.get("_header")
    version = (header.attrs if header is not None else f.attrs).get("version", "")
    if isinstance(version, bytes):
        version = version.decode()
    if str(version) not in ("", "2.3", "2.4") or ds.dtype["dataword"].itemsize != 1:
        raise ValueError(
            f"Expected packet HDF5 2.3 or 2.4 with ASIC v2 data (found {version})."
        )
    return ds


def write_run(output, blocks, records, iog):
    if not len(records):
        return
    records = records[np.argsort(records["tick"], kind="stable")]
    start = output.tell() // records.dtype.itemsize
    records.tofile(output)
    blocks.append(
        [iog, start, len(records), int(records["tick"][0]), int(records["tick"][-1])]
    )


def build_index(path, directory, geometry, sig, progress, chunk):
    clocks = {g: Clock() for g in range(1, 9)}
    blocks = {"hits": [], "marks": []}
    totals = Counter()
    with (
        h5py.File(path, "r") as f,
        (directory / "hits.bin").open("wb") as hits,
        (directory / "marks.bin").open("wb") as marks,
    ):
        ds = packet_dataset(f)
        size = len(ds)
        for start in range(0, size, chunk):
            batch = ds[start : start + chunk]
            totals.update(
                {
                    int(k): int(v)
                    for k, v in zip(
                        *np.unique(batch["packet_type"], return_counts=True)
                    )
                }
            )
            for iog in np.unique(batch["io_group"]):
                group = int(iog)
                if group not in clocks:
                    continue
                idx = np.flatnonzero(batch["io_group"] == group)
                data, markers = clocks[group].decode(
                    group, batch[idx], idx + start, geometry
                )
                write_run(hits, blocks["hits"], data, group)
                write_run(marks, blocks["marks"], markers, group)
            progress(min(start + chunk, size), size)
        cfg = f.get("daq_configs")
        config_name = str(cfg.attrs.get("dirname", "")) if cfg is not None else ""
        fields = list(ds.dtype.names)
    if sig != signature(path):
        raise ValueError(
            "The input changed while indexing. Select a closed packet file."
        )
    metadata = dict(
        signature=sig,
        rows=size,
        fields=fields,
        config=config_name,
        types=dict(totals),
        clocks={g: c.summary() for g, c in clocks.items()},
        **blocks,
    )
    (directory / "index.json").write_text(json.dumps(metadata, separators=(",", ":")))


class FileIndex:
    def __init__(self, path, directory):
        self.path = path
        self.meta = json.loads((directory / "index.json").read_text())
        self.directory = directory
        self.records = {}
        for name, dtype in [("hits", HIT), ("marks", MARK)]:
            file = directory / (name + ".bin")
            expected = sum(run[2] for run in self.meta[name]) * dtype.itemsize
            if file.stat().st_size != expected:
                raise ValueError(
                    f"Incomplete packet index at {directory}. Remove that cache entry and reopen."
                )
            self.records[name] = (
                np.memmap(file, mode="r", dtype=dtype)
                if expected
                else np.empty(0, dtype)
            )

    def rows(self, ids):
        if signature(self.path) != self.meta["signature"]:
            raise ValueError(
                "The input file changed. Reopen the selection to rebuild its index."
            )
        with h5py.File(self.path, "r") as f:
            ds = f["packets"]
            ids = np.asarray(ids, dtype=np.int64)
            ordered, inverse = np.unique(ids, return_inverse=True)
            if not len(ordered):
                return []
            if ordered[-1] - ordered[0] + 1 == len(ordered):
                raw = ds[int(ordered[0]) : int(ordered[-1]) + 1]
            else:
                raw = ds[ordered]
            return [
                dict(row=int(i), **{k: int(r[k]) for k in ds.dtype.names})
                for i, r in zip(ids, raw[inverse])
            ]


def open_index(
    path, cache, geometry, progress=lambda done, total: None, chunk=CHUNK_ROWS
):
    path, cache = Path(path).resolve(), Path(cache)
    sig = signature(path)
    key = hashlib.sha256(json.dumps(sig).encode()).hexdigest()[:32]
    target = cache / key
    if not target.exists():
        cache.mkdir(parents=True, exist_ok=True)
        temp = Path(tempfile.mkdtemp(prefix=key + ".", dir=cache))
        try:
            build_index(path, temp, geometry, sig, progress, chunk)
            try:
                temp.rename(target)
            except OSError:
                if not (target / "index.json").exists():
                    raise
        finally:
            if temp.exists():
                shutil.rmtree(temp)
    return FileIndex(path, target)
