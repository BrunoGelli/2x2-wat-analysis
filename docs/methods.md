# Whole-anode pulse recovery

`wat-analysis` searches for pulses with a broad anode footprint and measures the subsequent packet rate on each tile. It also measures how long packets from the same pulse take to reach PACMAN. Run it in a terminal or a NERSC batch allocation. The PDF and data tables work offline. See the [quick start](../README.md) and [timing guide](timing.md).

```bash
python -m pip install .
wat-analysis /path/to/packet-*.h5 --output wat-run45 --veto-trigger 5:2 --fit
```

The example veto uses type-7 subtype 2 on IOG 5. Select the lane appropriate to the run and set its timing with `--veto-us START STOP`. The default interval is [−50, +500) µs relative to the marker. Supply repeated `--veto-trigger IOG[:SUBTYPE]` options to combine lanes. Other trigger lanes remain available as tags. With no selected veto the measurements are labelled `unvetoed`. An absent or unaligned requested lane makes its file ineligible for a trigger-vetoed measurement. A veto can only identify activity associated with a recorded marker.

### Finding a pulse

- The search runs independently on every selected anode in ASIC time. By default an exact sliding 300 µs window must contain hits on at least 51% of the physically mapped chips and 51% of the mapped tiles. On an 800-chip, eight-tile plane these cuts require 408 chips across at least five tiles. Each physical chip counts once, including chips reached through alternate I/O routes. The recorded pixel footprint is also exported.
- `--min-chip-fraction`, `--min-tile-fraction`, `--window-us`, and an optional `--min-packets` cut control the selection. A quiet or disconnected chip stays in the geometry denominator. Each qualifying window is evaluated at native 100 ns resolution, regardless of the phase of a pulse relative to bin edges.
- SYNC-83 has a default ±30 µs guard on the measured IOG. Guarded hits cannot seed or enlarge a WAT, or join two pulses across a reset. The guard also excludes baseline and recovery exposure. A masked ASIC hit remains excluded from the receipt-time analysis if it arrives late.
- Overlapping qualifying windows form one candidate. Within that interval the first and last 10 µs bins reaching 5% of its peak bin define the prompt. Both the fine width and the fraction are adjustable. The main recovery clock starts at the right edge of the last prompt bin. The onset-aligned stack and raw examples expose sensitivity to that edge. Closely spaced bursts can be unresolved by the detection window.

### Baseline and exposure

Pulses separated by at most 10 ms from end to onset share a train. Every pulse in a train inherits its tile's baseline from [−20, −2) ms before the **first** pulse. The entire baseline-to-onset interval must have timing coverage clear of earlier candidates and vetoes. The two halves of the pre-window are compared on the anode. The default cut excludes an absolute conditional binomial imbalance score above 5 on either side. `--baseline-drift-z 0` disables it. It is an activity screen, and slow changes common to both halves can pass. The halves are retained by tile for inspection.

Each pulse contributes post-pulse bins up to the next pulse on that anode, a mask, a file boundary, or the requested follow-up. Only fully exposed bins count. A separate train-end stack compares the final pulse with the pre-train baseline. Each clean train supplies up to eight random control times from the same file and anode. Both their baseline and full follow-up must be free of candidates and vetoes, and their pre-window passes the same drift cut. Control placement never uses the post-window rate.

For a tile and delay bin, the normalized rate is the sum of observed counts divided by the sum of expected counts. The expected count for an event is its pre-train count times the ratio of post-bin exposure to pre-window exposure. Both clocks have their own measured baseline. Zero-count bins with exposure are included. A zero expected count for the entire stack leaves its ratio undefined. The tables give the number of pulses, number of trains, exposure, and both counts at each delay. Pointwise uncertainty bands resample whole trains within each anode. At least five contributing trains and finite bootstrap ratios are required.

`t90` measures a sustained return to 90% of baseline after a resolved dip. The rate must stay below 90% for the hold time and then at or above 90% for the hold time within supported bins. Defaults are 0.5 ms and at least five **expected** packets per bin. `t90_ms` is measured from the alignment reference and can be unresolved. `band_return_ms` also accepts an initial excess and gives the first sustained return to [0.9, 1.1] after a sustained departure on either side. The signed net deficit is expected minus observed packets. Its full-window mean uses only events with complete follow-up. Residual noise and delayed readout can mask losses of ordinary hits, so this statistic alone cannot determine physics efficiency.

### Load, FIFO and receipt timing

The load comparison groups pulses into empirical thirds by recorded prompt count, separately for each anode or tile. A second comparison ranks the eight tiles within each individual pulse. It pools rates for the low-, middle- and high-load tiles in that same disturbance. Equal loads stay together, so group sizes can differ. Distinct trains are the resampling units in both comparisons.

`--fit` additionally fits a bin-integrated signed model, `R(t) = 1 + A exp(−t/τ)`, to the ASIC profiles and individual anode/tile pulses. It uses Poisson counts conditional on the measured baseline, profiles amplitude from −1 upward (negative for a dip, positive for excess), and scans τ. The fit requires at least 20 baseline and expected post counts, four exposed bins, improvement over a flat baseline, a bounded conditional 95% profile interval and a deviance per degree of freedom of at most 3. These are descriptive, model-dependent fits. The rate measurement admits deficits, excesses, and unchanged rates regardless of fit status.

The index places PACMAN receipt in its recorded SYNC epoch. Late ASIC timestamps can belong to the preceding epoch. The two rates use the same ordinary, mapped, parity-valid charge population. The delay quantiles refer to the **same packets** selected in the prompt ASIC window, including receipts occurring later than the recovery horizon. Near a file boundary, packets absent from the file cannot contribute to a delay estimate.

### Inputs and output tables

Each input uses the standalone HDF5 index and per-IOG timing model. Identical resolved paths count once. Files keep independent clock state and source identity. Baselines and follow-up stop at file boundaries, even for consecutive files. The common time of overlapping files is masked on the affected IOGs. A missing nominal reset interval also interrupts exposure. A file-local clock can supply a spatial catalog entry but cannot enter the cross-IOG measurement. Detector coverage is inferred from the available timing records. Subsecond acquisition outages with no independent live indicator can be indistinguishable from silence.

The output directory must be new or empty. Each file has a stable numeric ID in `analysis.json`. For a single file, the pooled scope is also the file result. For multiple files the profiles and summaries include `file0`, `file1`, … and `all`.

| File | Contents |
| --- | --- |
| `wat-recovery.pdf` | File selection, pooled anode and tile measurements, load comparisons, support versus delay, prompt receipt delays, and examples spanning the recorded loads |
| `events.csv` | Candidate footprint, prompt edges, nearest marker offsets, eligibility, train membership, delay quantiles. Use `file_tick` with that file open alone, or `selection_tick` with the full aligned selection. |
| `event_tiles.csv`, `event_bins.csv.gz` | Pre-window half counts, prompt load, post exposure and deficit inputs by event and tile. The second table gives both clocks' counts in each exposed post bin. Tile 0 is the anode sum. |
| `profiles.csv.gz`, `summary.csv` | Exposure-matched stacks, bands, counts, support, sustained t90 and baseline-band return, signed deficit, fit status and load ranges |
| `reporting_chips.csv` | Distinct chips recording at least one ASIC hit 1–5 ms after the pulse, by tile and anode, with quiet controls. The reference averages equal-width, nonoverlapping intervals ending at the pre-window stop. `--chip-probe-ms` changes the interval. |
| `chips.csv.gz`, `fifo_matched.csv` | All mapped chips in each clean pulse, including quiet chips. The matched table compares chips on the same tile in the same pulse with **identical prompt counts**, pairing full-reporting chips with other prompt-reporting chips, separately for local and shared FIFO. Silent chips have no observed FIFO state. |
| `controls.csv`, `event_fits.csv` | Reproducible control reference ticks and optional individual fit parameters/status |
| `analysis.json` | Settings in integer detector ticks, input paths, clock qualification, trigger lanes, guard and eligibility counts by file and IOG |

Use `wat-analysis --help` for all options. Use `--cache-dir` to reuse this project's packet index across analysis runs.
