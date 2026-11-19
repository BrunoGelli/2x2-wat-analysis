# From packet rows to detector time

A recording contains a flat `/packets` HDF5 dataset in LArPix packet format 2.3 or 2.4. The reader checks the required field names and one-byte ADC dataword. Files without a version label can pass the same schema checks. One detector tick is 100 ns, so 10 ticks make 1 µs and 10,000 ticks make 1 ms.

## Two timestamps for each hit

The reader processes bounded blocks in source-row order and carries independent state for every IOG across blocks.

1. A type-6, subtype-83 SYNC reports the duration of the previous reset interval. The reader rounds it to the nearest multiple of 10,000,000 ticks and accepts a positive interval within 500,000 ticks of that multiple. A valid reset advances the IOG clock.
2. For an ordinary type-0 packet, ASIC time is the current offset plus the recorded ASIC phase modulo 10,000,000. When the raw receipt value is smaller than the raw ASIC timestamp, the reader subtracts the previous reset interval from the ASIC time. PACMAN receipt time uses the current offset plus the raw receipt value. The two timestamps can therefore lie on opposite sides of a reset.
3. The hit must have a valid time, valid parity, a mapped physical channel, and ordinary timestamp encoding. Rows before the first valid SYNC and FIFO-diagnostic timestamp words are excluded from the rate measurement. Each block is sorted by hit time **after** the reset correction.

For example, an ASIC hit one tick before reset that PACMAN receives two ticks after reset has a three-tick delay (0.3 µs). Subtracting the two raw timestamp fields directly across that reset would give a misleading value.

## Aligning anodes and files

The type-4 software timestamp preceding each reset supplies a whole-second label. A file/IOG qualifies for detector-wide alignment when at least two labelled resets agree on a single epoch and the stream has no invalid resets. A file-local tick remains useful for cataloguing pulses when qualification fails. Cross-IOG trigger matching requires a common clock.

The index keys each input by resolved path, file size, modification time, and schema version. Distinct files have independent reset histories. Each file's qualified coverage starts no earlier than its first recorded valid reset for this measurement. Baselines and recovery windows end at coverage boundaries. Earlier-leaving events contribute to early bins only. Spans shared by overlapping files on a required IOG are masked, and repeated references to one resolved path count once.

## Geometry and packet populations

The PacMon lookup maps IOG, I/O channel, chip ID, and channel ID to a physical pixel. Each group of four I/O channels serves one tile. A chip's physical identity includes its IOG and tile, so alternate routes to that chip contribute one vote to spatial participation. The footprint denominator includes all mapped chips, including quiet or disabled chips.

The SYNC guard splits the searchable time intervals before the exact footprint test. A masked hit also stays excluded from receipt-time profiles if it arrives after the guard. Trigger vetoes use the configured recorded lane and time window. The program records the available lanes so the run's trigger routing can be checked.

See the [method and output definitions](methods.md) and [geometry attribution](../NOTICE.md).
