import csv
import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from conftest import DTYPE

from wat_analysis.clocks import Run
from wat_analysis.index import HIT, PERIOD, open_index
from wat_analysis.wat import Settings, run, split_trains
from wat_analysis.wat_detect import (
    Plane,
    find_pulses,
    merge_intervals,
    qualifying_windows,
)
from wat_analysis.wat_report import fit_recovery, recovery, stack


def test_exact_sliding_footprint_against_brute_force():
    rng = np.random.default_rng(7)
    for _ in range(50):
        t = rng.integers(0, 150, size=100)
        c = rng.integers(0, 3, size=100) * 256 + rng.integers(0, 4, size=100)
        wanted = []
        for end in np.unique(t):
            use = (t > end - 12) & (t <= end)
            if (
                use.sum() >= 8
                and len(np.unique(c[use])) >= 6
                and len(np.unique(c[use] // 256)) >= 3
            ):
                wanted.append((int(end - 11), int(end + 1)))
        assert merge_intervals(
            qualifying_windows(t, c, 12, 8, 6, 3, 0, 150)
        ) == merge_intervals(wanted)


def test_reset_cannot_supply_footprint_or_receipt(geom):
    # The flash and its neighbor would only pass by borrowing chips across the reset.
    rows = np.zeros(8, HIT)
    for i, tile in enumerate(range(1, 9)):
        rows["pixel"][i] = geom.lut[1, tile][geom.lut[1, tile] >= 0][0]
    rows["tick"] = [99, 99, 99, 99, 102, 102, 102, 102]
    rows["receipt_tick"] = rows["tick"] + 100
    p = Plane([Run(0, 1, 0, 99, 103, rows)], geom, 1, (0, 300))
    cfg = Settings(window=12, fine=2, chip_fraction=0.01, tile_fraction=1)
    assert len(find_pulses(p, cfg)) == 1
    assert find_pulses(p, cfg, [(98, 101)]) == []
    p.hit_mask_edges = np.array([98, 101])
    assert len(p.rows(190, 210, "receipt_tick")) == 4


def test_receipt_crosses_reset_in_original_epoch(packet_file, tmp_path, geom):
    index = open_index(packet_file, tmp_path / "cache", geom, chunk=3)
    rows = index.records["hits"]
    late = rows[rows["adc"] == 24]
    assert len(late) == 2 and np.all(late["receipt_tick"] - late["tick"] == 3)


def test_quiet_neighbor_and_censored_exposure():
    c = Settings(
        post=4000, bin=1000, pre=5000, guard=0, bootstrap=0, hold=1000, min_expected=1
    )
    before = np.array([10, 10])
    after = np.array([[0, 1, 2, 2], [3, 2, 2, 2]])
    s = stack(
        [("t1", after, before, 80), ("t1", after[:, :1], before, 40)],
        c,
        np.random.default_rng(1),
    )
    assert s["support"].tolist() == [2, 1, 1, 1]
    assert s["independent"].tolist() == [1, 1, 1, 1]
    np.testing.assert_array_equal(s["ratio"][0], [0, 0.5, 1, 1])
    assert recovery(s["ratio"][0], s["expected"][0], c) == (0.2, "returned")
    assert recovery(s["ratio"][1], s["expected"][1], c)[1] == "no_sustained_dip"
    assert recovery(np.array([5.0, 0.2, 0.4, 1.1]), np.ones(4) * 20, c) == (
        0.3,
        "returned",
    )
    assert recovery(
        np.array([4.0, 2.0, 1.05, 1.01]), np.ones(4) * 20, c, band=True
    ) == (0.2, "returned")
    assert s["complete"] == 1 and s["deficit"][0] == 3
    assert split_trains(
        [SimpleNamespace(start=2, end=5), SimpleNamespace(start=8, end=10)], 6, [(6, 7)]
    ) == [[SimpleNamespace(start=2, end=5)], [SimpleNamespace(start=8, end=10)]]


def test_optional_fit_resolves_excess_and_dip_only_when_constrained():
    dt = 0.25
    t = np.arange(40) * dt
    e = np.full(len(t), 3000.0)
    k = np.round(e * (1 - 0.9 * np.exp(-t / 2) * (-np.expm1(-dt / 2)) * 2 / dt))
    fit = fit_recovery(k, e, dt, 100000)
    assert fit["fit_status"] == "fit" and abs(fit["tau_ms"] - 2) < 0.2
    assert fit_recovery(e, e, dt, 100000)["fit_status"] == "no_supported_departure"
    excess = fit_recovery(2 * e - k, e, dt, 100000)
    assert (
        excess["direction"] == "excess"
        and fit["direction"] == "dip"
        and abs(excess["tau_ms"] - 2) < 0.2
    )


def acquisition(
    path, geom, sec, veto_phase=400_000, conflicting=False, bursts=(2, 350_000, 400_000)
):
    rows = []

    def record(g, kind, ts, tile=1, sub=0, receipt=None):
        r = np.zeros(1, DTYPE)
        lut = geom.lut[g, tile]
        chip, chan = np.unravel_index(np.argmax(lut >= 0), lut.shape)
        for name, val in {
            "io_group": g,
            "io_channel": (tile - 1) * 4 + 1,
            "chip_id": chip,
            "channel_id": chan,
            "packet_type": kind,
            "timestamp": ts,
            "trigger_type": sub,
            "valid_parity": 1,
            "receipt_timestamp": ts if receipt is None else receipt,
        }.items():
            r[name] = val
        rows.append(r)

    for g in (1, 5):
        record(g, 4, sec)
        record(g, 6, PERIOD, sub=83)
        if g == 1:
            for phase in range(500, 510_000, 1000):
                if not (350_000 < phase < 353_000):
                    record(g, 0, phase, 1)
                record(g, 0, phase, 2)
            for burst in bursts:
                for tile in range(1, 9):
                    record(g, 0, burst, tile)
        else:
            record(g, 7, veto_phase, sub=2)
        record(g, 4, sec + 1 + int(conflicting and g == 5))
        record(g, 6, PERIOD, sub=83)
    # One reset epoch per block; the index must sort hit times after decoding.
    with h5py.File(path, "w") as h:
        h.create_dataset("packets", data=np.concatenate(rows), chunks=True)
        h.create_group("_header").attrs["version"] = "2.4"
    return path


def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def test_batch_multifile_real_entry_point(tmp_path, geom, monkeypatch):
    from wat_analysis.cli import main

    paths = [
        acquisition(tmp_path / f"run{i}.h5", geom, 1_700_000_000 + 3 * i)
        for i in range(2)
    ]
    out = tmp_path / "out"
    args = [
        "wat-analysis",
        *map(str, paths),
        str(paths[0]),
        "--output",
        str(out),
        "--cache-dir",
        str(tmp_path / "cache"),
        "--window-us",
        "2",
        "--fine-us",
        ".1",
        "--min-chip-fraction",
        ".009",
        "--min-tile-fraction",
        "1",
        "--baseline-ms",
        "1",
        "0",
        "--bin-ms",
        ".1",
        "--post-ms",
        ".9",
        "--hold-ms",
        ".1",
        "--sync-guard-us",
        "5",
        "--veto-trigger",
        "5:2",
        "--veto-us",
        "-1",
        "1",
        "--train-gap-ms",
        "1",
        "--chip-probe-ms",
        ".1",
        ".3",
        "--controls-per-train",
        "1",
        "--bootstrap",
        "0",
        "--iog",
        "1",
        "--examples",
        "0",
    ]
    monkeypatch.setattr("sys.argv", args)
    main()
    assert (out / "wat-recovery.pdf").read_bytes().startswith(b"%PDF-")
    events = rows(out / "events.csv")
    assert len(events) == 4
    assert [x["status"] for x in events] == [
        "trigger_veto_pass",
        "trigger_coincident",
    ] * 2
    assert [int(x["post_bins"]) for x in events] == [9, 0] * 2
    assert events[0]["file_tick"] == events[2]["file_tick"]
    assert events[0]["selection_tick"] != events[2]["selection_tick"]
    meta = json.loads((out / "analysis.json").read_text())
    assert (
        len(meta["files"]) == 2
        and meta["files"][0]["planes"]["1"]["sync83_excluded_packets"] == 8
    )
    event = rows(out / "event_tiles.csv")
    assert [(int(x["tile"]), int(x["post"])) for x in event[:3]] == [
        (0, 15),
        (1, 6),
        (2, 9),
    ]
    assert {r["scope"] for r in rows(out / "summary.csv")} == {"all", "file0", "file1"}
    chips = [r for r in rows(out / "reporting_chips.csv") if r["sample"] == "pulse"]
    assert [
        (float(r["expected_chips"]), int(r["observed_chips"])) for r in chips[:3]
    ] == [(2, 1), (1, 0), (1, 1)]
    sentinel = out / "keep"
    sentinel.write_text("keep")
    with pytest.raises(SystemExit):
        main()
    assert sentinel.read_text() == "keep"


def test_unavailable_beam_clock_and_overlapping_files(tmp_path, geom):
    p = acquisition(tmp_path / "a.h5", geom, 1700000000, conflicting=True)
    q = acquisition(tmp_path / "b.h5", geom, 1700000000)
    out = tmp_path / "out"
    cfg = Settings(
        window=20,
        fine=1,
        chip_fraction=0.009,
        tile_fraction=1,
        veto=((5, 2),),
        iogs=(1, 5),
        examples=0,
        bootstrap=0,
    )
    run([p, q], tmp_path / "cache", out, cfg)
    cats = {r["status"] for r in rows(out / "events.csv")}
    assert cats == {"missing_veto_lane", "ambiguous_coverage"}
    assert not rows(out / "summary.csv")


def test_train_inherits_baseline_and_censors_at_next_pulse(tmp_path, geom):
    p = acquisition(
        tmp_path / "train.h5", geom, 1_700_000_000, bursts=(2, 350_000, 352_500)
    )
    cfg = Settings(
        window=20,
        fine=1,
        chip_fraction=0.009,
        tile_fraction=1,
        pre=10_000,
        guard=0,
        post=4000,
        bin=1000,
        hold=1000,
        train_gap=5000,
        controls=0,
        examples=0,
        bootstrap=0,
        iogs=(1,),
    )
    out = tmp_path / "out"
    run([p], tmp_path / "cache", out, cfg)
    a, b = rows(out / "events.csv")
    assert (
        a["train"] == b["train"]
        and a["baseline_start_file_tick"] == b["baseline_start_file_tick"]
    )
    assert [int(a["post_bins"]), int(b["post_bins"])] == [2, 4]
    ends = [
        r
        for r in rows(out / "summary.csv")
        if r["sample"] == "train_end" and r["tile"] == "0" and r["clock"] == "asic"
    ]
    assert [int(a["packets"]), int(b["packets"])] == [
        8,
        9,
    ]  # One ordinary hit shares the second prompt.
    assert len(ends) == 1 and int(ends[0]["load_min"]) == 17
