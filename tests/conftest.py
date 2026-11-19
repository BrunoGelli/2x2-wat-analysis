import h5py
import numpy as np
import pytest

from wat_analysis.geometry import Geometry
from wat_analysis.index import PERIOD

DTYPE = np.dtype(
    [
        (n, t)
        for n, t in (
            ("io_group", "u1"),
            ("io_channel", "u1"),
            ("chip_id", "u1"),
            ("channel_id", "u1"),
            ("packet_type", "u1"),
            ("timestamp", "<u8"),
            ("receipt_timestamp", "<u4"),
            ("valid_parity", "u1"),
            ("trigger_type", "u1"),
            ("dataword", "u1"),
            ("local_fifo", "u1"),
            ("shared_fifo", "u1"),
            ("fifo_diagnostics_enabled", "u1"),
            ("first_packet", "u1"),
            ("downstream_marker", "u1"),
        )
    ]
)


@pytest.fixture(scope="module")
def geom():
    return Geometry()


def sample(path, geom, second=1794029067, conflicting=False):
    tile = geom.lut[1, 1]
    chip, chan = next(zip(*np.nonzero(tile >= 0)))
    rows = []

    def packet(kind, ts, g=1, **kw):
        r = np.zeros(1, DTYPE)
        r["io_group"] = g
        r["io_channel"] = 1
        r["chip_id"] = chip
        r["channel_id"] = chan
        r["packet_type"] = kind
        r["timestamp"] = ts
        r["valid_parity"] = 1
        for key, value in kw.items():
            r[key] = value
        rows.append(r)

    for g in (1, 2):
        packet(4, second, g)
        packet(0, 15, g, receipt_timestamp=25)  # unqualified prefix
        packet(6, PERIOD + 2, g, trigger_type=83)
        packet(0, 3, g, receipt_timestamp=8, dataword=8, local_fifo=3)
        packet(0, 3, g, receipt_timestamp=10, dataword=16, local_fifo=0)
        packet(
            0, 2, g, receipt_timestamp=15, dataword=200, local_fifo=1
        )  # late in row order
        packet(7, 5, g, trigger_type=2)
        packet(6, 6, g, trigger_type=72)
        packet(0, 6, g, receipt_timestamp=10, valid_parity=0)
        packet(0, 6, g, receipt_timestamp=10, fifo_diagnostics_enabled=1)
        packet(4, second + 1 + (g == 2 and conflicting), g)
        packet(6, PERIOD, g, trigger_type=83)
        packet(0, PERIOD - 1, g, receipt_timestamp=2, dataword=24)
    with h5py.File(path, "w") as f:
        f.create_group("_header").attrs["version"] = "2.4"
        f.create_dataset("packets", data=np.concatenate(rows), chunks=True)
    return path


@pytest.fixture
def packet_file(tmp_path, geom):
    return sample(tmp_path / "example.h5", geom)
