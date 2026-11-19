# Reproducing a WAT study

The Run 0070045 example is a 2.44 GB packet HDF5 recording. Use an existing local copy if available. To download the example into the current directory, run

```bash
curl -fLO 'https://portal.nersc.gov/project/dune/data/2x2/nearline_run3/packet/Operations/20261111-LowIntensityBeam_LRSselfTrigger_Physics/packet-0070045-2026_11_12_04_06_30_CST.h5'
```

## Three measurements on the same file

```bash
recording=packet-0070045-2026_11_12_04_06_30_CST.h5

wat-analysis "$recording" --veto-trigger 5:2 --fit \
  --output results/run45

wat-analysis "$recording" --veto-trigger 5:2 \
  --min-packets 10000 --min-chip-fraction 0.75 --min-tile-fraction 0.75 \
  --fine-us 5 --prompt-fraction 0.01 --controls-per-train 1 --bootstrap 0 --examples 0 \
  --output results/run45-strict

wat-analysis "$recording" --veto-trigger 5:2 \
  --post-ms 200 --bin-ms 2 --hold-ms 6 --examples 0 \
  --output results/run45-long
```

The first command reproduces the 20 ms study. The second tests sensitivity to the spatial footprint and the definition of the prompt end. The third follows the activity for 200 ms. The default stack uses all fully exposed bins at each delay. For a comparison that follows exactly the same pulses throughout the long window, select events with `post_bins == 100` (200 ms / 2 ms) and use those event IDs at every delay.

For consecutive files, pass their paths in the same invocation. In a shell that expands globs, use `wat-analysis /path/to/packet-0070045-*.h5 --output results/run45-set --veto-trigger 5:2`. On Windows, pass the individual file paths when the shell does not expand wildcards.

## Read a stack

This Python snippet prints the anode-level ASIC profile from 1 to 5 ms in the default study. It weights each bin by its expected count.

```python
import csv
import gzip
from collections import defaultdict

totals = defaultdict(lambda: [0, 0.0])
with gzip.open("results/run45/profiles.csv.gz", "rt", newline="") as stream:
    for row in csv.DictReader(stream):
        if (row["scope"], row["sample"], row["tile"], row["load"], row["clock"]) != (
            "all", "pulse", "0", "all", "asic"
        ):
            continue
        if float(row["lo_ms"]) >= 1 and float(row["hi_ms"]) <= 5:
            counts = totals[int(row["iog"])]
            counts[0] += int(row["observed"])
            counts[1] += float(row["expected"])

for iog, (observed, expected) in sorted(totals.items()):
    ratio = observed / expected if expected else None
    print(iog, observed, expected, ratio)
```

As a worked example, a tile with 90 hits in an 18 ms pre-window has an expected rate of 5 packets/ms. A fully exposed 0.25 ms post bin then carries an expectation of 1.25 packets. Four such observations with six total recorded packets give a normalized rate of 6/5 = 1.2. If a fifth pulse is interrupted before that bin, it adds neither counts nor exposure. The quiet-control curve provides a check on whether the baseline describes comparable periods of the run.

## Common questions

| Observation | What to inspect |
| --- | --- |
| Broad pulses but few usable pulses | Compare `status`, `clean_baseline`, `baseline_drift_z`, and `post_bins` in `events.csv`. A candidate can pass the trigger veto and still lack clean baseline or follow-up. |
| A requested veto lane is missing | Check `trigger_lanes` and clock qualification in `analysis.json`. Verify the IOG and subtype for this run before changing the selection. |
| A tile has a blank ratio or fit | Check its expected baseline count, fit status, and number of contributing trains. Sparse data can leave a ratio undefined or a timescale unconstrained. |
| Receipt delay rises while hit rate is high | Compare ASIC-time and receipt-time curves, prompt delay quantiles, distinct reporting chips, and the within-pulse load groups. These observables distinguish delayed delivery from changes in recorded activity. |
| A result changes with the follow-up window | Check support versus delay and repeat on a fixed set of events with full exposure. An isolated 200 ms sample preferentially selects well-separated pulses. |
