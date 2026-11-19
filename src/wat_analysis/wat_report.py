"""Exposure-weighted WAT stacks, optional fits, and standalone diagnostics."""

import csv
from collections import Counter, defaultdict

import numpy as np

from .wat import Table
from .wat_detect import TICKS_PER_MS


def divide(count, expected):
    return np.divide(
        count, expected, out=np.full(np.shape(expected), np.nan), where=expected > 0
    )


def recovery(ratio, expected, cfg, band=False):
    valid = np.isfinite(ratio) & (expected >= cfg.min_expected)
    hold = (cfg.hold + cfg.bin - 1) // cfg.bin

    def sustained(mask):
        return np.flatnonzero(
            np.convolve(mask.astype(int), np.ones(hold, int), "valid") == hold
        )

    if not len(sustained(valid)):
        return None, "insufficient_exposure"
    departures = sustained(valid & (ratio < 0.9))
    returned = ratio >= 0.9
    if band:
        departures = np.union1d(departures, sustained(valid & (ratio > 1.1)))
        returned &= ratio <= 1.1
    if not len(departures):
        return None, "no_sustained_departure" if band else "no_sustained_dip"
    first = int(departures[0])
    returns = sustained(valid & returned)
    for j in returns[returns >= first + hold]:
        if valid[first : j + hold].all():
            return float(j * cfg.bin / TICKS_PER_MS), "returned"
        break
    return None, "unresolved_in_observed_window"


def fit_recovery(k, expected, dt_ms, baseline_counts):
    """Conditional Poisson profile with bin-integrated 1+A exp(-t/tau).

    A joint confidence band on physical deadtime would require an independent
    efficiency probe. These optional fits describe the observed packet rate.
    """
    result = {
        "fit_status": "insufficient_counts",
        "direction": None,
        "amplitude": None,
        "tau_ms": None,
        "tau_lo_ms": None,
        "tau_hi_ms": None,
        "deviance_dof": None,
    }
    usable = expected > 0
    if baseline_counts < 20 or expected.sum() < 20 or usable.sum() < 4:
        return result
    t = np.flatnonzero(usable) * dt_ms
    k, expected = k[usable], expected[usable]
    tau = np.geomspace(dt_ms / 10, (t[-1] + dt_ms) * 10, 72)[:, None]
    f = np.exp(-t / tau) * (-np.expm1(-dt_ms / tau)) * tau / dt_ms
    # For R=1+A*f the physical lower limit is -1. At the upper bracket,
    # sum(k*f/(1+A*f)) <= sum(k)/A <= sum(expected*f), so the optimum is bracketed.
    low = -np.ones(len(tau))
    high = np.maximum(1, k.sum() / np.sum(expected * f, axis=1))
    for _ in range(36):
        a = (low + high) / 2
        derivative = np.sum(f * (expected - k / (1 + a[:, None] * f)), axis=1)
        low = np.where(derivative < 0, a, low)
        high = np.where(derivative < 0, high, a)
    amplitude = (low + high) / 2
    frac = amplitude[:, None] * f
    delta = np.sum(expected * frac - k * np.log1p(frac), axis=1)
    i = int(delta.argmin())
    if delta[i] > -1.92:
        return dict(result, fit_status="no_supported_departure")
    profile = np.flatnonzero(2 * (delta - delta[i]) <= 3.84)
    if profile[0] == 0 or profile[-1] == len(tau) - 1:
        return dict(result, fit_status="tau_unconstrained")
    mu = expected * (1 + frac[i])
    positive = k > 0
    deviance = 2 * (
        np.sum(mu - k) + np.sum(k[positive] * np.log(k[positive] / mu[positive]))
    )
    deviance /= len(k) - 2
    if deviance > 3:
        return dict(result, fit_status="poor_exponential_shape", deviance_dof=deviance)
    return {
        "fit_status": "fit",
        "direction": "excess" if amplitude[i] > 0 else "dip",
        "amplitude": amplitude[i],
        "tau_ms": tau[i, 0],
        "tau_lo_ms": tau[profile[0], 0],
        "tau_hi_ms": tau[profile[-1], 0],
        "deviance_dof": deviance,
    }


def strata(values):
    a, b = np.quantile(values, [1 / 3, 2 / 3])
    # Keep equal observed loads together, including zero-load tiles.
    return {
        "low": values <= a,
        "middle": (values > a) & (values <= b),
        "high": values > b,
    }


def stack(members, cfg, rng):
    """members contain (train, after[clock,time], before[clock], load)."""
    nbin = cfg.post // cfg.bin
    train_id = {name: i for i, name in enumerate(dict.fromkeys(m[0] for m in members))}
    by_train = np.zeros((len(train_id), 2, 2, nbin))
    exposed = np.zeros((len(train_id), nbin), dtype=int)
    complete_deficit = []
    scale = cfg.bin / (cfg.pre - cfg.guard)
    for train, after, before, _ in members:
        i, n = train_id[train], after.shape[1]
        by_train[i, 0, :, :n] += after
        by_train[i, 1, :, :n] += before[:, None] * scale
        exposed[i, :n] += 1
        if n == nbin:
            complete_deficit.append(nbin * before * scale - after.sum(axis=1))
    obs, exp = by_train.sum(axis=0)
    ratio = divide(obs, exp)
    num_trains = (exposed > 0).sum(axis=0)
    bounds = np.full((2, 2, nbin), np.nan)
    if cfg.bootstrap and len(train_id) >= 5:
        ntrain = len(train_id)
        weights = rng.multinomial(
            ntrain, np.full(ntrain, 1 / ntrain), size=cfg.bootstrap
        )
        draws = (weights @ by_train.reshape(ntrain, -1)).reshape(-1, 2, 2, nbin)
        boot = divide(draws[:, 0], draws[:, 1])
        for clock in range(2):
            good = (num_trains >= 5) & np.isfinite(boot[:, clock, :]).all(axis=0)
            if good.any():
                bounds[:, clock, good] = np.percentile(
                    boot[:, clock, good], [2.5, 97.5], axis=0
                )
    return {
        "ratio": ratio,
        "bounds": bounds,
        "observed": obs,
        "expected": exp,
        "support": exposed.sum(axis=0),
        "independent": num_trains,
        "complete": len(complete_deficit),
        "n": len(members),
        "trains": len(train_id),
        "deficit": (
            np.mean(complete_deficit, axis=0) if complete_deficit else [None, None]
        ),
        "loads": [m[3] for m in members],
    }


def groups(obs, tile):
    members = [
        (x.train, x.after[:, :, tile], x.before[:, tile], int(x.load[tile]))
        for x in obs
    ]
    yield "all", members
    if obs[0].kind == "pulse" and len(members) >= 3:
        values = np.array([m[3] for m in members])
        if np.ptp(values):
            for label, mask in strata(values).items():
                selected = [m for m, yes in zip(members, mask) if yes]
                if selected:
                    yield label, selected


def within_pulse(obs):
    by_rank = defaultdict(list)
    for x in obs:
        if not np.ptp(x.load[1:]):
            continue
        for rank, mask in strata(x.load[1:]).items():
            if mask.any():
                after = x.after[:, :, 1:][:, :, mask].sum(axis=2)
                before = x.before[:, 1:][:, mask].sum(axis=1)
                by_rank[rank].append(
                    (x.train, after, before, int(x.load[1:][mask].sum()))
                )
    return by_rank.items()


def summarize(directory, observations, cfg, nfiles):
    curves = {}
    rng = np.random.default_rng(cfg.seed)
    dt = cfg.bin / TICKS_PER_MS
    meta = "scope iog sample tile load clock"
    fit_fields = (
        "fit_status direction amplitude tau_ms tau_lo_ms tau_hi_ms deviance_dof"
    )
    totals = Table(
        directory,
        "summary",
        meta + " events trains load_min load_max complete_events "
        "mean_complete_net_deficit t90_ms t90_status band_return_ms band_status "
        + fit_fields,
    )
    data = Table(
        directory,
        "profiles",
        meta
        + " lo_ms hi_ms events trains exposure_ms observed expected ratio low95 high95",
        True,
    )
    fits = (
        Table(directory, "event_fits", "file iog train event tile prompt " + fit_fields)
        if cfg.fit
        else None
    )
    fit_points = defaultdict(list)
    try:
        if fits:
            for x in observations:
                if x.kind != "pulse":
                    continue
                for tile in range(9):
                    k = x.after[0, :, tile]
                    exp = np.full(
                        len(k), x.before[0, tile] * cfg.bin / (cfg.pre - cfg.guard)
                    )
                    fit = fit_recovery(k, exp, dt, x.before[0, tile])
                    fits.row(
                        file=x.file,
                        iog=x.iog,
                        train=x.train,
                        event=x.event,
                        tile=tile,
                        prompt=x.load[tile],
                        **fit,
                    )
                    if fit["fit_status"] == "fit":
                        fit_points[x.iog].append(
                            (tile, x.load[tile], fit["tau_ms"], fit["direction"])
                        )
        sets = defaultdict(list)
        for x in observations:
            sets["all", x.iog, x.kind].append(x)
            if nfiles > 1:
                sets[f"file{x.file}", x.iog, x.kind].append(x)
        for (scope, iog, kind), samples in sets.items():
            selections = [
                (tile, load, members)
                for tile in range(9)
                for load, members in groups(samples, tile)
            ]
            if kind == "pulse":
                selections.extend(
                    ("within", key, val) for key, val in within_pulse(samples)
                )
            for tile, load, members in selections:
                s = stack(members, cfg, rng)
                # Keep only the pooled curves used in the PDF. Write all scopes to CSV.
                if scope == "all" and (load == "all" or tile in (0, "within")):
                    curves[iog, kind, tile, load] = s
                for c, clock in enumerate(("asic", "receipt")):
                    key = {
                        "scope": scope,
                        "iog": iog,
                        "sample": kind,
                        "tile": tile,
                        "load": load,
                        "clock": clock,
                    }
                    ratio = s["ratio"][c]
                    t90, status = recovery(ratio, s["expected"][c], cfg)
                    band_time, band_status = recovery(
                        ratio, s["expected"][c], cfg, band=True
                    )
                    fit = {"fit_status": "disabled"}
                    if cfg.fit and c == 0 and kind == "pulse" and tile != "within":
                        fit = fit_recovery(
                            s["observed"][c],
                            s["expected"][c],
                            dt,
                            s["expected"][c, 0] * (cfg.pre - cfg.guard) / cfg.bin,
                        )
                    totals.row(
                        **key,
                        events=s["n"],
                        trains=s["trains"],
                        load_min=min(s["loads"]),
                        load_max=max(s["loads"]),
                        complete_events=s["complete"],
                        mean_complete_net_deficit=s["deficit"][c],
                        t90_ms=t90,
                        t90_status=status,
                        band_return_ms=band_time,
                        band_status=band_status,
                        **fit,
                    )
                    for j, (n, nt) in enumerate(zip(s["support"], s["independent"])):
                        if not n:
                            break
                        ci = s["bounds"][:, c, j]
                        data.row(
                            **key,
                            lo_ms=j * dt,
                            hi_ms=(j + 1) * dt,
                            events=n,
                            trains=nt,
                            exposure_ms=n * dt,
                            observed=int(s["observed"][c, j]),
                            expected=s["expected"][c, j],
                            ratio=ratio[j] if np.isfinite(ratio[j]) else None,
                            low95=ci[0] if np.isfinite(ci[0]) else None,
                            high95=ci[1] if np.isfinite(ci[1]) else None,
                        )
    finally:
        totals.file.close()
        data.file.close()
        if fits:
            fits.file.close()
    return curves, fit_points


def write_pdf(directory, metadata, result, examples, cfg):
    from matplotlib import colormaps
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.figure import Figure

    curves, fits = result
    files = metadata["files"]
    dt = cfg.bin / TICKS_PER_MS
    t = (np.arange(cfg.post // cfg.bin) + 0.5) * dt
    teal, orange, slate, purple = "#077f82", "#c34e20", "#475a73", "#754799"
    dark = "#183448"

    def page(title, subtitle):
        fig = Figure(figsize=(11.7, 8.3), facecolor="white")
        fig.text(0.07, 0.94, title, fontsize=22, weight="bold", color=dark)
        fig.text(0.07, 0.897, subtitle, fontsize=10, color=slate)
        return fig

    def panel(ax, title, ylabel):
        ax.set_title(
            title, loc="left", fontsize=12, fontweight="bold", color=dark, pad=8
        )
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)

    def line(ax, s, label, color, clock=0, ci=False, style="-"):
        if s is None:
            return
        y = s["ratio"][clock]
        ax.plot(t, y, style, label=label, color=color, lw=1.6)
        if ci:
            ax.fill_between(t, *s["bounds"][:, clock], color=color, alpha=0.15, lw=0)

    def ratio_axis(ax, xlabel="Time after prompt end (ms)"):
        ax.axhline(1, color="#555555", lw=0.8, ls=":")
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_xlim(0, cfg.post / TICKS_PER_MS)
        ax.set_yscale("symlog", linthresh=1, base=10)
        ax.set_ylim(bottom=-0.02)
        # Matplotlib's default symlog locator omits sub-baseline detail.
        top = ax.get_ylim()[1]
        ticks = [
            y for y in (0, 0.5, 1, 2, 5, 10, 50, 100, 1000, 10000, 100000) if y <= top
        ]
        ax.set_yticks(ticks, [f"{y:g}" for y in ticks])
        if ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=False, fontsize=8)

    def note(fig, content):
        fig.text(0.07, 0.035, content, fontsize=8.5, color=slate)

    with PdfPages(directory / "wat-recovery.pdf") as pdf:
        fig = page(
            "Whole-anode pulse recovery",
            files[0]["name"]
            if len(files) == 1
            else f"{len(files)} packet files  ·  PPS-aligned detector time",
        )
        fig.text(0.07, 0.82, "Pulse selection", fontsize=13, weight="bold", color=dark)
        mask = ", ".join(str(g) if s is None else f"{g}/{s}" for g, s in cfg.veto)
        selection = (
            f"Sliding window {cfg.window / 10:g} µs   ·   ≥{cfg.chip_fraction:.0%} of mapped chips   ·   "
            f"≥{cfg.tile_fraction:.0%} of mapped tiles   ·   ≥{cfg.min_packets} packets\n"
            f"SYNC-83 guard ±{cfg.sync_guard / 10:g} µs   ·   "
            + (
                f"Type-7 veto on IOG/subtype {mask}, [{cfg.veto_left / 10:g}, {cfg.veto_right / 10:g}) µs"
                if mask
                else "No type-7 veto configured"
            )
        )
        fig.text(0.07, 0.746, selection, fontsize=11, linespacing=1.5)
        ax = fig.add_axes([0.065, 0.341, 0.87, 0.36])
        ax.axis("off")
        body = []
        first_bin = min(len(t) - 1, round(1 / dt))
        last_bin = min(len(t), max(first_bin + 1, round(5 / dt)))
        for iog in cfg.iogs:
            total = Counter()
            for f in files:
                total.update(
                    {
                        k: v
                        for k, v in f["planes"].get(iog, {}).items()
                        if isinstance(v, int)
                    }
                )

            s = curves.get((iog, "pulse", 0, "all"))
            c = curves.get((iog, "control", 0, "all"))

            def early_ratio(curve):
                if not curve:
                    return "—"
                e = curve["expected"][0, first_bin:last_bin].sum()
                return (
                    f"{curve['observed'][0, first_bin:last_bin].sum() / e:.2f}"
                    if e
                    else "—"
                )

            body.append(
                [
                    iog,
                    total["candidates"],
                    total["trigger_coincident"],
                    total["measured_pulses"],
                    total["clean_trains"],
                    total["controls"],
                    early_ratio(s),
                    early_ratio(c),
                ]
            )
        cols = [
            "IOG",
            "Broad pulses",
            "Trigger veto",
            "Usable pulses",
            "Clean trains",
            "Controls",
            f"ASIC R\n{first_bin * dt:g}–{last_bin * dt:g} ms",
            "Control R",
        ]
        table = ax.table(
            cellText=body,
            colLabels=cols,
            cellLoc="center",
            loc="upper left",
            bbox=[0, 0, 1, 1],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        for (r, _), cell in table.get_celld().items():
            cell.set_edgecolor("white")
            cell.set_facecolor("#dfefef" if r == 0 else "#f1f5f6" if r % 2 else "white")
            if r == 0:
                cell.set_text_props(weight="bold", color=dark, fontsize=9)
        fig.text(
            0.07,
            0.282,
            "How to read the measurement",
            fontsize=13,
            weight="bold",
            color=dark,
        )
        fig.text(
            0.07,
            0.143,
            f"R compares observed packets with the count expected from the same tiles before the first pulse of each train.\n"
            f"Baseline [−{cfg.pre / TICKS_PER_MS:g}, −{cfg.guard / TICKS_PER_MS:g}) ms. Trains join pulses separated by ≤{cfg.train_gap / TICKS_PER_MS:g} ms.\n"
            "Post-pulse exposure stops at the next pulse, veto, or file boundary. Quiet controls use randomly placed reference times.\n"
            "Signed deficits estimate missing baseline activity. Trigger-vetoed candidates can still include untagged physics.",
            fontsize=10,
            linespacing=1.6,
        )
        note(
            fig,
            "Tile 0 in the tables denotes the whole anode. Exact event references, bin exposure, and fit status accompany the figures.",
        )
        pdf.savefig(fig)
        for iog in cfg.iogs:
            s = curves.get((iog, "pulse", 0, "all"))
            if not s:
                continue
            control = curves.get((iog, "control", 0, "all"))
            fig = page(
                f"IOG {iog}  ·  the anode and its eight tiles",
                f"{s['n']} measured pulses from {s['trains']} trains   ·   "
                f"{s['complete']} pulses have the full {cfg.post / TICKS_PER_MS:g} ms of follow-up",
            )
            axes = fig.subplots(2, 2)
            fig.subplots_adjust(
                left=0.075, right=0.95, top=0.81, bottom=0.115, hspace=0.51, wspace=0.28
            )
            a, b, c, d = axes.ravel()
            panel(a, "Recovery in ASIC hit time", "Rate / baseline")
            line(a, s, "Pulses", teal, ci=True)
            line(a, control, "Quiet controls", slate, ci=True)
            train = curves.get((iog, "train_end", 0, "all"))
            if train and train["n"] < s["n"]:
                line(a, train, "After final pulse", purple, style="--")
            ratio_axis(a)
            panel(b, "Hit time and readout time", "Rate / baseline")
            line(b, s, "ASIC, prompt end", teal)
            line(b, s, "PACMAN receipt, prompt end", orange, clock=1)
            line(
                b,
                curves.get((iog, "onset", 0, "all")),
                "ASIC, prompt onset",
                slate,
                style="--",
            )
            ratio_axis(b, "Time after each alignment reference (ms)")
            heat = np.stack(
                [curves[iog, "pulse", tile, "all"]["ratio"][0] for tile in range(1, 9)]
            )
            panel(c, "Tile recovery in ASIC hit time", "Tile")
            cmap = colormaps["RdBu_r"].with_extremes(bad="#d2d5d6")
            finite = heat[np.isfinite(heat)]
            upper = (
                max(1, int(np.ceil(np.log2(max(2, np.percentile(finite, 95))))))
                if len(finite)
                else 1
            )
            im = c.imshow(
                np.ma.masked_invalid(np.log2(np.maximum(heat, 0.125))),
                aspect="auto",
                origin="lower",
                interpolation="nearest",
                extent=[0, cfg.post / TICKS_PER_MS, 0.5, 8.5],
                cmap=cmap,
                norm=TwoSlopeNorm(vmin=-3, vcenter=0, vmax=upper),
            )
            c.grid(False)
            c.set_yticks(range(1, 9))
            c.set_xlabel("Time after prompt end (ms)", fontsize=9)
            cticks = sorted({-3, -1, 0, 1, upper // 2, upper})
            cb = fig.colorbar(im, ax=c, pad=0.02, ticks=cticks, extend="both")
            cb.ax.set_yticklabels([f"{2.0**k:g}×" for k in cticks], fontsize=8)
            panel(d, "Local load within the same pulse", "Rate / baseline")
            for key, color in [("low", teal), ("middle", slate), ("high", orange)]:
                line(
                    d,
                    curves.get((iog, "pulse", "within", key)),
                    f"{key.capitalize()}-load tiles",
                    color,
                )
            ratio_axis(d)
            note(
                fig,
                f"Shading shows pointwise 95% train-bootstrap intervals when available. "
                f"Contributing pulses, first to last bin: {s['support'][0]} to {s['support'][-1]}. Gray tiles have no estimable ratio.",
            )
            pdf.savefig(fig)
            # File-wide quantiles answer a different question from within-pulse ranks.
            fig = page(
                f"IOG {iog}  ·  pulse load and recovery",
                "Load is the number of recorded prompt ASIC packets. Each load group uses its own pre-train baseline.",
            )
            a, b = fig.subplots(1, 2)
            fig.subplots_adjust(
                left=0.075, right=0.94, top=0.78, bottom=0.23, wspace=0.27
            )
            panel(a, "Anode recovery by prompt load", "Rate / baseline")
            for key, color in [("low", teal), ("middle", slate), ("high", orange)]:
                group = curves.get((iog, "pulse", 0, key))
                if group:
                    label = f"{key.capitalize()}: {min(group['loads']):,}–{max(group['loads']):,}"
                    line(a, group, label, color, ci=True)
            ratio_axis(a)
            panel(b, "Events contributing at each delay", "Pulses")
            b.step(t, s["support"], where="mid", color=teal, label="All pulses")
            for key, color in [("low", "#2a9491"), ("middle", slate), ("high", orange)]:
                group = curves.get((iog, "pulse", 0, key))
                if group:
                    b.step(
                        t,
                        group["support"],
                        where="mid",
                        color=color,
                        label=key.capitalize(),
                    )
            b.set_xlabel("Time after prompt end (ms)")
            b.set_xlim(0, cfg.post / TICKS_PER_MS)
            b.set_ylim(bottom=0)
            b.legend(frameon=False, fontsize=9)
            note(
                fig,
                "Load boundaries are empirical thirds for this selection. Ties stay together. The support plot shows censoring at later delays.",
            )
            pdf.savefig(fig)
        with (directory / "reporting_chips.csv").open() as stream:
            active = defaultdict(lambda: np.zeros(2))
            for row in csv.DictReader(stream):
                key = (int(row["iog"]), row["sample"], int(row["tile"]))
                active[key] += [
                    float(row["observed_chips"]),
                    float(row["expected_chips"]),
                ]
        if active:
            start, stop = np.array(cfg.chip_probe) / TICKS_PER_MS
            fig = page(
                "How many chips are still reporting?",
                f"Distinct chips with a recorded hit {start:g}–{stop:g} ms after prompt end, compared with equal-duration intervals before the train.",
            )
            a, b = fig.subplots(1, 2)
            fig.subplots_adjust(
                left=0.085, right=0.945, top=0.79, bottom=0.22, wspace=0.3
            )
            panel(a, "The whole anode", "Reporting chips / baseline")
            panel(b, "Individual tiles", "Reporting chips / baseline")
            for g in cfg.iogs:
                ratios = []
                for tile in range(9):
                    obs, exp = active[g, "pulse", tile]
                    ratios.append(obs / exp if exp else np.nan)
                if np.isfinite(ratios[0]):
                    a.bar(
                        g,
                        ratios[0],
                        width=0.55,
                        color=teal,
                        label="After pulse" if not a.patches else None,
                    )
                    ctl, ref = active[g, "control", 0]
                    if ref:
                        a.scatter(
                            g,
                            ctl / ref,
                            marker="D",
                            color=orange,
                            zorder=4,
                            label="Quiet control" if len(a.collections) == 0 else None,
                        )
                    b.plot(
                        range(1, 9), ratios[1:], marker="o", lw=1.3, label=f"IOG {g}"
                    )
            for ax in (a, b):
                ax.axhline(1, color=slate, ls=":", lw=1)
                if ax.get_legend_handles_labels()[0]:
                    ax.legend(frameon=False, fontsize=8)
                ax.set_ylim(bottom=0)
            a.set_xticks(cfg.iogs)
            b.set_xticks(range(1, 9))
            a.set_xlabel("IO group")
            b.set_xlabel("Physical tile")
            note(
                fig,
                "Each chip counts once per interval regardless of packet multiplicity. Only pulses with full exposure through the end of the interval contribute.",
            )
            pdf.savefig(fig)
        with (directory / "events.csv").open() as stream:
            events = [r for r in csv.DictReader(stream) if int(r["post_bins"])]
        if events:
            fig = page(
                "When prompt packets reach PACMAN",
                "Each point is one anode pulse. Delay uses the two unrolled timestamps of the same recorded packet.",
            )
            axes = fig.subplots(1, 2)
            fig.subplots_adjust(
                left=0.085, right=0.95, top=0.79, bottom=0.22, wspace=0.25
            )
            for ax, col, title in zip(
                axes,
                ("receipt_lag_p50_ms", "receipt_lag_p99_ms"),
                ("Median packet delay", "99th-percentile packet delay"),
            ):
                panel(ax, title, "Receipt time − ASIC hit time (ms)")
                for iog in cfg.iogs:
                    rows = [r for r in events if int(r["iog"]) == iog]
                    if rows:
                        ax.scatter(
                            [int(r["packets"]) for r in rows],
                            [float(r[col]) for r in rows],
                            s=16,
                            alpha=0.65,
                            label=f"IOG {iog}",
                        )
                ax.set_xscale("log")
                ax.set_yscale("symlog", linthresh=1)
                ax.set_xlabel("Packets in prompt ASIC window")
                ax.legend(frameon=False, fontsize=8)
            note(
                fig,
                "Receipt delay measures the path from ASIC timestamp to PACMAN. The packet stream alone cannot locate all delay or measure lost physics hits.",
            )
            pdf.savefig(fig)
        if examples:
            examples.sort(key=lambda x: x["packets"])
            ix = np.linspace(
                0, len(examples) - 1, min(cfg.examples, len(examples)), dtype=int
            )
            for idx in np.unique(ix):
                x = examples[idx]
                fig = page(
                    f"Example pulse  ·  IOG {x['iog']}",
                    f"{x['filename']}   ·   File-relative PPS-aligned tick {x['file_tick']:,}.",
                )
                a, b = fig.subplots(2, 1)
                fig.subplots_adjust(
                    left=0.10, right=0.94, top=0.81, bottom=0.12, hspace=0.53
                )
                panel(
                    a,
                    f"Prompt structure   ·   {x['packets']:,} packets, {x['chips']} chips, {x['tiles']} tiles",
                    f"Packets / {cfg.fine / 10:g} µs",
                )
                a.stairs(
                    x["fine_counts"], x["fine_us"], color=teal, fill=True, alpha=0.85
                )
                a.axvspan(0, x["duration_us"], color=orange, alpha=0.12)
                a.set_xlabel("Time from pulse onset (µs)")
                panel(b, "Recorded ASIC activity around onset", f"Packets / {dt:g} ms")
                b.stairs(x["wide_counts"], x["wide_ms"], color=slate)
                b.axhline(
                    x["baseline_hz"] * dt / 1000,
                    color=teal,
                    ls="--",
                    label="Pre-train baseline",
                )
                b.axvline(0, color=orange, lw=1)
                stop = x["duration_us"] / 1000 + x["live_post_ms"]
                if stop < x["wide_ms"][-1]:
                    b.axvspan(
                        stop,
                        x["wide_ms"][-1],
                        color="#bbb",
                        alpha=0.3,
                        label="Censored",
                    )
                b.set_xlabel("Time from pulse onset (ms)")
                b.set_yscale("symlog", linthresh=1)
                b.legend(frameon=False)
                note(
                    fig,
                    f"{x['event']}   ·   Fine-bin prompt duration {x['duration_us']:g} µs. "
                    "The lower trace includes all recorded activity in view, including adjacent pulses.",
                )
                pdf.savefig(fig)
        if cfg.fit:
            fig = page(
                "Individual relaxation fits",
                "ASIC-time fits that resolve both sides of the conditional 95% profile interval.",
            )
            ax = fig.add_axes([0.12, 0.22, 0.75, 0.56])
            panel(ax, "Recovery time versus recorded prompt load", "Fitted τ (ms)")
            plotted = False
            for g, v in sorted(fits.items()):
                for direction, marker in (("dip", "v"), ("excess", "^")):
                    for anode in (False, True):
                        points = [
                            (load, tau)
                            for tile, load, tau, sign in v
                            if (tile == 0) == anode and sign == direction and load > 0
                        ]
                        if points:
                            region = "anode" if anode else "tile"
                            ax.scatter(
                                *zip(*points),
                                s=34 if anode else 15,
                                marker=marker,
                                alpha=0.6,
                                label=f"IOG {g} {region} · {direction}",
                            )
                            plotted = True
            if plotted:
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.legend(frameon=False, fontsize=8)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No individual exponential fits passed the support and shape checks.",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
            ax.set_xlabel("Prompt packets")
            note(
                fig,
                "The fit table retains unsuccessful attempts and their status. The stacked rate measurement includes every eligible pulse.",
            )
            pdf.savefig(fig)
