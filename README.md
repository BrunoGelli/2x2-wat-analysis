# 2×2 WAT Analysis

After a whole-anode trigger (WAT), do fewer chips record charge, or do packets from the burst simply take longer to reach PACMAN? `wat-analysis` measures both effects in closed ND-LAr 2×2 LArPix packet files. It finds broad pulses on each anode, measures every tile against its own pre-train baseline, and follows the same prompt packets from ASIC timestamp to PACMAN receipt.

This is a standalone offline Python package. It includes the HDF5 reader, clock reconstruction, geometry, and PDF report generator. The separate [2×2 Packet Explorer](https://github.com/BrunoGelli/2x2-packet-explorer) is useful for hand-scanning a particular file and tick.

## Install and run

Use Python 3.10 or later. From a terminal on a workstation or NERSC compute node, run

```bash
git clone https://github.com/BrunoGelli/2x2-wat-analysis.git
cd 2x2-wat-analysis
python -m venv .venv
source .venv/bin/activate
python -m pip install .
wat-analysis --help
```

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1`. The module entry point `python -m wat_analysis` accepts the same arguments.

For example, run this command in the directory containing the Run 0070045 recording. It writes a PDF and machine-readable tables to a new or empty output directory.

```bash
wat-analysis packet-0070045-2026_11_12_04_06_30_CST.h5 \
  --output results/run45 --veto-trigger 5:2 --fit
```

The `5:2` choice means type-7 markers on IOG 5 with subtype 2. Check the trigger routing for your run. Omitting the option produces a spatial candidate study labelled `unvetoed`. The [Run 0070045 walkthrough](docs/run45.md) includes the source recording, a stricter selection, a 200 ms follow-up, and commands for inspecting a result.

## What the analysis measures

| Question | Measurement |
| --- | --- |
| Where and when did the pulse occur? | An exact sliding 300 µs window looks for hits on at least 51% of mapped chips and tiles on one anode. The default ±30 µs SYNC-83 exclusion applies before pulse finding. |
| Did ordinary readout activity change? | Each tile gets the same anode pulse time. Its expected count comes from −20 to −2 ms before the first pulse of the train. Stacked rates are compared with quiet reference times from the same file and IOG. |
| Could one busy chip dominate the rate? | A distinct-chip measurement counts each reporting chip once per interval. Load groups compare tiles within a pulse and compare pulses of different sizes. |
| Did prompt packets arrive late? | The per-pulse median, 90th and 99th percentile measure receipt time minus ASIC hit time for the same prompt packets. |
| Can we assign a recovery time? | The report measures sustained return to 90% after a dip, or return to a band around baseline after an excess or dip. Optional exponential fits carry explicit support and fit-quality statuses. |

A recorded rate above baseline can coexist with lost physics hits if residual noise produces extra packets. The measurement quantifies recorded activity. A physics-efficiency estimate requires an independently identifiable signal.

## Reading a result

1. Open **`wat-recovery.pdf`** for candidate counts, anode profiles, tile heatmaps, prompt examples, and paired receipt delays.
2. Check **`analysis.json`** for the input files, qualified clocks, trigger lanes, settings, and per-IOG eligibility counts. In **`profiles.csv.gz`**, check how many trains still contribute at the delay you care about.
3. Use **`events.csv`** to revisit a pulse. Its file name, IOG, and `file_tick` identify the prompt onset on the aligned clock of that file. Join the event ID to **`event_tiles.csv`** and **`event_bins.csv.gz`** for individual counts.

Multiple input paths produce both per-file and pooled measurements. Inputs stay read-only. The index cache defaults to `$SCRATCH/2x2-wat-analysis` on NERSC when `NERSC_HOST` and `SCRATCH` are set, and otherwise to `2x2-wat-analysis` under `$XDG_CACHE_HOME` or `~/.cache`. Use `--cache-dir` to choose a different location. Keep the cache when comparing cuts on the same files.

| Guide | Contents |
| --- | --- |
| [Analysis method](docs/methods.md) | Spatial selection, train grouping, normalization, controls, censoring, FIFO matching, fits, and output inventory |
| [Packet clocks and geometry](docs/timing.md) | Input schema, SYNC unrolling, late hits, the two clocks, physical chip identity, file boundaries, and overlaps |
| [Run 0070045 walkthrough](docs/run45.md) | Reproducible commands, a worked normalization example, and troubleshooting |
| [Source and geometry provenance](NOTICE.md) | Upstream timing conventions, pinned detector layouts, attribution, and geometry rebuild instructions |

## Development

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

Tests build small HDF5 recordings with known pulse times and tile behavior. They exercise the installed CLI, a reset-crossing receipt, a tile that keeps recording while its neighbor falls quiet, multi-file overlaps, train censoring, and recovery fits of both signs.
