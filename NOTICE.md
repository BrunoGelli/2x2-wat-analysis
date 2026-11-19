# Source and geometry provenance

The WAT batch analysis originated in a draft on the [2×2 Packet Explorer repository](https://github.com/BrunoGelli/2x2-packet-explorer/pull/1). The initial standalone implementation derives from [commit c1369f8](https://github.com/BrunoGelli/2x2-packet-explorer/commit/c1369f8d0c1102eb55336a5ed3802f2608479be0). This project maintains its own packet reader, geometry loader, clock views, and tests. Its installation needs no Packet Explorer checkout.

The detector clock follows the ordered per-IOG SYNC-83 and receipt-boundary conventions of the [2×2 movie maker](https://github.com/BrunoGelli/2x2-raw-event-display-movie-maker/blob/62290e2652d42abf1c9441abed62b103202696d9/raw_movie/_upstream/timing.py).

The bundled geometry in `src/wat_analysis/data/tiles.json` and `channels.npz` derives from four PacMon v4 JSON layouts at commit `2cf0e2c7db056dd205efb7f41616c1795fa9ea67`. Copyright © 2023 FERMI NATIONAL ACCELERATOR LABORATORY for the benefit of the DUNE Collaboration. PacMon distributes its layouts under the Apache License, Version 2.0. A copy accompanies the geometry. [Upstream layouts](https://github.com/BrunoGelli/2x2Pacmon/tree/2cf0e2c7db056dd205efb7f41616c1795fa9ea67/layout) · [Bundled license](src/wat_analysis/data/LICENSE-PacMon.txt)

To rebuild the derived geometry from that PacMon checkout, run

```bash
python scripts/build_geometry.py /path/to/2x2Pacmon/layout src/wat_analysis/data
```

The builder verifies the four original Git blob hashes before generating the tile and channel maps.
