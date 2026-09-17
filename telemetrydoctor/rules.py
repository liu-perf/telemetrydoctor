"""The ten rules, each of which is a mistake somebody actually shipped.

Every rule here has a provenance line in `docs/rules.md` pointing at where it was
hit: a hand-written analysis note, a real terminal session, or an experiment in
`examples/verify_on_device.py` on one RTX 5060 Ti. None of them were invented by
reading the NVML documentation and imagining what could go wrong -- which matters,
because the ones that go wrong in practice are a small and unobvious subset of the
ones that could.

Statuses, in the vocabulary the other six tools use:

    ok          the file does not have this problem
    info        a fact about the file the reader needs before quoting any number
    warn        an aggregate computed the obvious way would mislead
    violation   an aggregate computed the obvious way would be wrong by a known factor
    unknown     the file does not contain what is needed to decide -- NOT a pass
"""
from . import crosscheck as _cc
from . import interval as _iv
from . import segment as _seg
from .contracts import CONTRACTS, FORBIDDEN, MEANINGLESS, contract_for

OK = "ok"
INFO = "info"
WARN = "warn"
VIOLATION = "violation"
UNKNOWN = "unknown"

SEVERITY = {OK: 0, INFO: 1, UNKNOWN: 2, WARN: 3, VIOLATION: 4}

# Above this the nominal interval is wrong enough that any per-second figure built
# on it is wrong by more than measurement noise. The field case was 37.6%.
INTERVAL_WARN_PCT = 5.0
# The threshold the field script used before it had to be changed per workload.
LEGACY_FIXED_THRESHOLD = 90.0


class Finding:
    __slots__ = ("rule", "location", "status", "message")

    def __init__(self, rule, location, status, message):
        self.rule = rule
        self.location = location
        self.status = status
        self.message = message

    def as_dict(self):
        return {"rule": self.rule, "location": self.location,
                "status": self.status, "message": self.message}

    def __str__(self):
        return "{}: [{}] {}".format(self.location, self.status, self.message)


class Context:
    """Everything the rules read, computed once."""

    __slots__ = ("series", "interval", "threshold", "segments", "cross", "column")

    def __init__(self, series, column="sm_pct", explicit_threshold=None):
        self.series = series
        self.column = column
        self.interval = _iv.measure(series)
        self.segments, self.threshold = _seg.segments(
            series, column, explicit_threshold)
        self.cross = _cc.compare(series, self.segments, self.threshold, column)


# --------------------------------------------------------------------- the rules
def tl001_interval(ctx):
    iv = ctx.interval
    if iv.basis == "unknown":
        return Finding(
            "TL001", "interval", UNKNOWN,
            "no timestamps in this file and no --interval given, so the sampling "
            "interval cannot be measured. Every per-second figure derived from this "
            "file rests on the nominal value, and the nominal value is the one thing "
            "known to be wrong: a real 8-card pynvml loop asked for 1.0 s and produced "
            "1.376 s/row. Re-record with timestamps (nvidia-smi dmon -o DT).")
    if iv.basis == "nominal_only":
        return Finding(
            "TL001", "interval", UNKNOWN,
            "no timestamps; using the nominal {:.3f} s you supplied. Nothing in the "
            "file confirms it -- sampler overhead grows with the number of cards and "
            "is not visible from the loop's source.".format(iv.nominal_s))

    over = iv.overshoot_pct
    detail = ("measured {:.3f} s/row over {} gaps (median {:.3f}, min {:.3f}, max {:.3f})"
              .format(iv.measured_s, iv.n_gaps, iv.median_s, iv.min_s, iv.max_s))
    if iv.nominal_s is None:
        return Finding("TL001", "interval", INFO,
                       detail + "; no nominal value given to compare against, so this "
                       "measured figure is the one to divide by")
    if over is not None and abs(over) > INTERVAL_WARN_PCT:
        return Finding("TL001", "interval", WARN,
                       "{}, against a nominal {:.3f} s -- {:+.1f}%. Any figure computed "
                       "as value/nominal is off by that much, in the direction that "
                       "flatters the machine.".format(detail, iv.nominal_s, over))
    return Finding("TL001", "interval", OK,
                   "{}, within {:.1f}% of the nominal {:.3f} s"
                   .format(detail, abs(over or 0.0), iv.nominal_s))


def tl002_phase_mixing(ctx):
    segs = ctx.segments
    if not segs:
        return Finding("TL002", "phases", UNKNOWN,
                       "no phases could be cut (see TL007), so whether a whole-file "
                       "mean mixes phases cannot be decided")
    total = ctx.series.n_samples
    counted = {}
    for s in segs:
        counted[s.kind] = counted.get(s.kind, 0) + s.n_samples
    active = counted.get(_seg.ACTIVE, 0)
    non_active = total - active
    shape = ", ".join("{} {} samples".format(v, k) for k, v in sorted(counted.items()))
    labels = sorted({s.label for s in segs if s.kind == _seg.ACTIVE})

    if non_active <= 0 and len(labels) <= 1:
        return Finding("TL002", "phases", OK,
                       "one configuration and nothing else in the file ({}), so a "
                       "whole-file mean and a phase mean are the same number"
                       .format(shape))
    if counted.get(_seg.LOAD):
        why = ("A mean over the whole file blends them, and the load stretch is the "
               "dangerous one: it is a multi-gigabyte host-to-device transfer with the "
               "SMs idle, so averaging it into training traffic is what produced a "
               "fabricated '8 cards, 47 GB/s'.")
    elif len(labels) > 1:
        why = ("A mean over the whole file blends {} different configurations plus the "
               "gaps between them, and lands on a number that describes none of them."
               .format(len(labels)))
    else:
        why = ("A mean over the whole file blends the work with the idle stretches "
               "around it and lands between the two.")
    return Finding(
        "TL002", "phases", WARN,
        "{} of {} samples ({:.0f}%) are not steady work -- {}. Configurations found: "
        "{}. {}".format(non_active, total, 100.0 * non_active / max(1, total), shape,
                        " ".join(labels) or "none", why))


def tl003_integration(ctx):
    iv = ctx.interval
    interval_s = iv.effective_s
    bad = []
    for col in ctx.series.columns:
        c = contract_for(col)
        if c and c.verdict("integrate") == FORBIDDEN and isinstance(c.window_ms, int):
            duty = c.duty_cycle(interval_s) if interval_s else None
            bad.append((col, c, duty))
    if not bad:
        return Finding("TL003", "integrate", OK,
                       "no column in this file summarises a fixed hardware window, so "
                       "there is nothing here that breaks under mean x elapsed")
    if interval_s is None:
        cols = ", ".join(c for c, _, _ in bad)
        return Finding("TL003", "integrate", UNKNOWN,
                       "{} summarise fixed hardware windows, so mean x elapsed is "
                       "scaled by an unmeasured duty cycle -- and with no interval the "
                       "size of that scaling is unknown too".format(cols))
    worst = max(bad, key=lambda b: (1.0 / b[2]) if b[2] else 0.0)
    col, c, duty = worst
    return Finding(
        "TL003", "integrate", VIOLATION,
        "{} observe{} {} ms of each {:.3f} s interval -- {:.2f}% of elapsed time. "
        "Multiplying a mean by elapsed time to get a total understates it by about "
        "{:.0f}x. The mean itself is fine as a mean rate; it is the product that is "
        "not a total.".format(
            ", ".join(b[0] for b in bad), "" if len(bad) > 1 else "s",
            c.window_ms, interval_s, 100.0 * duty, 1.0 / duty))


def tl004_peaks(ctx):
    cols = [c for c in ctx.series.columns
            if contract_for(c) and contract_for(c).verdict("peak") == MEANINGLESS]
    if not cols:
        return Finding("TL004", "peak", OK,
                       "no column here has a peak that reports sampling luck")
    interval_s = ctx.interval.effective_s
    c = contract_for(cols[0])
    duty = c.duty_cycle(interval_s) if interval_s else None
    extra = (" At {:.2f}% duty the maximum is drawn from {:.2f}% of the traffic, so it "
             "is a statement about which samples got lucky.".format(
                 100.0 * duty, 100.0 * duty) if duty else "")
    return Finding(
        "TL004", "peak", WARN,
        "{} carry a windowed reading, so their maxima are not peak rates.{} Report the "
        "mean, or record with a tool that integrates in hardware."
        .format(", ".join(cols), extra))


def tl005_scope(ctx):
    series = ctx.series
    if "sm_pct" not in series.columns:
        return Finding("TL005", "scope", OK, "no percentage column in this file")
    n_gpus = len(series.gpus)
    if n_gpus < 2:
        return Finding("TL005", "scope", OK,
                       "single card, so busy-card mean and all-card mean are the same "
                       "number and cannot be confused")
    active = [s for s in ctx.segments if s.kind == _seg.ACTIVE]
    if not active:
        return Finding("TL005", "scope", UNKNOWN,
                       "no active phase found, so the two scopes cannot be compared")

    worst = None
    from .aggregate import ALL_DEVICE, BUSY
    from .aggregate import mean as _mean
    for s in active:
        busy = _mean(series, s, "sm_pct", scope=BUSY)
        allv = _mean(series, s, "sm_pct", scope=ALL_DEVICE)
        if busy.refused or allv.refused:
            continue
        gap = busy.value - allv.value
        if worst is None or gap > worst[0]:
            worst = (gap, s, busy.value, allv.value)
    if worst is None:
        return Finding("TL005", "scope", UNKNOWN, "could not compute both scopes")

    gap, s, busy_v, all_v = worst
    if gap < 5.0:
        return Finding("TL005", "scope", OK,
                       "every card participates in every configuration here, so the two "
                       "scopes agree to within {:.1f} points".format(gap))
    return Finding(
        "TL005", "scope", WARN,
        "in {} the busy-card mean is {:.1f}% and the all-card mean is {:.1f}% -- "
        "{:.1f} points apart, because {} of {} cards were idle. Both are correct "
        "answers to different questions and a report that prints only the second one "
        "invites the reader to conclude the cards were not working. That has happened."
        .format(s.label, busy_v, all_v, gap, n_gpus - len(s.busy_gpus), n_gpus))


def tl006_noise_floor(ctx):
    idle = [s for s in ctx.segments if s.kind == _seg.IDLE]
    if not idle:
        return Finding(
            "TL006", "noise_floor", UNKNOWN,
            "no idle stretch in this file, so the noise floor cannot be measured from "
            "it. Without one there is no control group: a single-card configuration "
            "whose tx sits at the floor is the only evidence that the tx column is "
            "measuring communication at all.")
    from .aggregate import BUSY
    from .aggregate import mean as _mean
    parts = []
    for col in ("pcie_tx_gbs", "pcie_rx_gbs", "power_w", "sm_pct"):
        if col not in ctx.series.columns:
            continue
        vals = []
        for s in idle:
            v = _mean(ctx.series, s, col, scope=BUSY if s.busy_gpus else "all_device_mean")
            if not v.refused:
                vals.append(v.value)
        if vals:
            parts.append("{}={:.5f} {}".format(col, sum(vals) / len(vals),
                                               contract_for(col).unit))
    n = sum(s.n_samples for s in idle)
    return Finding(
        "TL006", "noise_floor", INFO,
        "measured from {} idle samples in {} stretch(es): {}. Measure this per file, "
        "not once: two files from the same 8-card node an hour apart had idle tx of "
        "0.04538 and 0.00643 GB/s -- a factor of 7 -- so a floor carried over from "
        "another run is an assumption dressed as a measurement."
        .format(n, len(idle), ", ".join(parts) or "no comparable columns"))


def tl007_threshold(ctx):
    th = ctx.threshold
    if th.basis == "explicit":
        return Finding(
            "TL007", "threshold", WARN,
            "busy threshold {:.1f} was supplied on the command line, so it was not "
            "taken from this file. Three real workloads on 8-card nodes plateaued at "
            "48.4%, 69.9% and 94.1%; a constant that fits one of those selects nothing "
            "on the others.".format(th.value))
    if not th.usable:
        sep = "{:.2f}".format(th.separability) if th.separability is not None else "n/a"
        return Finding(
            "TL007", "threshold", UNKNOWN,
            "no usable busy/idle split: separability {} is under the {:.2f} floor, so "
            "the distribution of {} is effectively one class. This file cannot be "
            "phase-segmented, and every phase-scoped number below is withheld rather "
            "than computed against a threshold cut down the middle of one mode."
            .format(sep, _seg.MIN_SEPARABILITY, ctx.column))

    vals = ctx.series.column(ctx.column)
    n_busy = sum(1 for v in vals if v >= th.value)
    missed = sum(1 for v in vals if th.value <= v < LEGACY_FIXED_THRESHOLD)
    tail = ""
    if missed and n_busy:
        share = 100.0 * missed / n_busy
        tail = (" A fixed {:.0f} threshold would discard {} of the {} card-samples this "
                "file says were working -- {}."
                .format(LEGACY_FIXED_THRESHOLD, missed, n_busy,
                        "every one of them" if missed == n_busy
                        else "{:.0f}% of them".format(share)))

    def level(x):
        return "n/a" if x is None else "{:.1f}".format(x)

    return Finding(
        "TL007", "threshold", INFO,
        "busy threshold {:.1f} computed from this file (Otsu, separability {:.2f}); "
        "busy plateau {}, idle level {}.{}"
        .format(th.value, th.separability, level(th.high_mean), level(th.low_mean),
                tail))


def tl008_semantics(ctx):
    if "sm_pct" not in ctx.series.columns:
        return Finding("TL008", "semantics", OK, "no utilisation column in this file")
    return Finding(
        "TL008", "semantics", INFO,
        "sm_pct is the fraction of the driver's sample window in which at least one "
        "kernel was resident. It is not SM occupancy and not tensor-core activity; for "
        "those, DCGM prof counters or Nsight. Measured on one RTX 5060 Ti: a 64x64 "
        "matmul submitted back-to-back and a 4096x4096 matmul at quarter duty both read "
        "~28% while delivering 0.046 and 45.4 TFLOPS -- 978x apart. So two jobs cannot "
        "be ranked by this column, which is why one 8-card node showing 48.4% on "
        "detection and 94.1% on segmentation says nothing about which used the hardware "
        "better.")


def tl009_crosscheck(ctx):
    cc = ctx.cross
    if cc.basis != "compared":
        return Finding("TL009", "crosscheck", UNKNOWN,
                       "the second method could not run (no usable threshold, no active "
                       "phase, or no busy plateau), so nothing independent confirms the "
                       "segmentation and no configuration total should be quoted yet")
    if cc.agree:
        return Finding(
            "TL009", "crosscheck", OK,
            "two independent segmentations agree on every configuration: per-card "
            "identity (which cards are above {:.1f}) and pooled level alone (all-card "
            "mean / {:.1f} plateau x {} cards). The second never looks at an individual "
            "card, so agreement is not the first method checking itself."
            .format(ctx.threshold.value, cc.plateau, cc.n_gpus))
    label, frac = cc.worst
    return Finding(
        "TL009", "crosscheck", VIOLATION,
        "the two segmentations disagree: in {} only {:.0f}% of samples agree on how "
        "many cards were busy (floor {:.0f}%). Either the threshold is wrong, the "
        "plateau is not flat, or cards are neither working nor idle. No per-"
        "configuration figure from this file should be reported until that is resolved."
        .format(label, 100.0 * frac, 100.0 * _cc.MIN_AGREEING_FRACTION))


def tl010_oversampling(ctx):
    interval_s = ctx.interval.effective_s
    if "sm_pct" not in ctx.series.columns:
        return Finding("TL010", "oversampling", OK, "no windowed column to check")
    active = [s for s in ctx.segments if s.kind == _seg.ACTIVE]
    if not active:
        return Finding("TL010", "oversampling", UNKNOWN,
                       "no active phase to count repeats over; an idle card reports "
                       "exactly 0.0 every time, so counting repeats across idle "
                       "stretches would measure the idling, not the sampling")
    rep = _iv.repeats(ctx.series, "sm_pct", interval_s, segments=active)
    if rep.n_rows < 4:
        return Finding("TL010", "oversampling", UNKNOWN, "too few rows to tell")
    frac = rep.repeat_fraction
    if interval_s is None:
        return Finding("TL010", "oversampling", UNKNOWN,
                       "{:.0f}% of rows repeat the row before, but with no measured "
                       "interval it cannot be said whether that is oversampling or a "
                       "genuinely steady signal".format(100.0 * frac))
    if frac < _iv.REPEAT_WARN_FRACTION:
        return Finding("TL010", "oversampling", OK,
                       "{:.0f}% of sm_pct card-samples in the active phases are "
                       "identical to the one before, at {:.3f} s sampling; the rate is "
                       "not outrunning the driver's window"
                       .format(100.0 * frac, interval_s))
    return Finding(
        "TL010", "oversampling", WARN,
        "{:.0f}% of sm_pct card-samples in the active phases are identical to the one "
        "before, at {:.3f} s sampling. "
        "NVML documents the utilisation window as 1 s to 1/6 s, so a faster poll "
        "returns the same reading repeatedly -- {} card-samples, but far fewer "
        "independent observations. Quoting n={} in an error bar overstates the evidence."
        .format(100.0 * frac, interval_s, rep.n_rows, rep.n_rows))


RULES = (
    ("TL001", tl001_interval),
    ("TL002", tl002_phase_mixing),
    ("TL003", tl003_integration),
    ("TL004", tl004_peaks),
    ("TL005", tl005_scope),
    ("TL006", tl006_noise_floor),
    ("TL007", tl007_threshold),
    ("TL008", tl008_semantics),
    ("TL009", tl009_crosscheck),
    ("TL010", tl010_oversampling),
)


def audit(series, column="sm_pct", explicit_threshold=None, only=None):
    """Run the rules. -> (list of Finding, Context)."""
    ctx = Context(series, column, explicit_threshold)
    wanted = set(only) if only else None
    findings = []
    for rule_id, fn in RULES:
        if wanted and rule_id not in wanted:
            continue
        findings.append(fn(ctx))
    return findings, ctx


def worst_status(findings):
    if not findings:
        return OK
    return max((f.status for f in findings), key=lambda s: SEVERITY.get(s, 0))


def describe_contracts():
    """One line per column in the contract table, for `telemetrydoctor contracts`."""
    out = []
    for name in sorted(CONTRACTS):
        c = CONTRACTS[name]
        window = ("{}-{} ms".format(*c.window_ms) if isinstance(c.window_ms, tuple)
                  else ("{} ms".format(c.window_ms) if c.window_ms else "instant"))
        ops = " ".join("{}={}".format(op, c.verdict(op))
                       for op in ("mean", "peak", "integrate", "sum_across_gpus"))
        out.append(Finding("contract", name, INFO,
                           "{} [{}], window {}; {}".format(c.unit, c.kind, window, ops)))
    return out
