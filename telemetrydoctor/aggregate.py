"""Compute the aggregates the contracts allow, and refuse the ones they don't.

The refusals are the product. Anybody can average a column; the useful thing is a
tool that will not average the wrong column and will say why in one line.

Two things it does that the hand-written awk it replaces could not:

  * scope is explicit. `busy_mean` averages only the cards that were working in
    that segment; `all_device_mean` averages every card in the file. On an 8-card
    node running a 1-card configuration those are 99.0% and 12.5%, and both are
    correct answers to different questions. The field report printed only the
    second one and a reader concluded the card was idle. So both are computed, and
    the name of each says which it is.
  * an integration that the contract forbids comes back with the size of the error
    attached, not just a refusal. `mean x elapsed` on a 20 ms-window PCIe counter
    sampled every 1.376 s is wrong by 69x, and 69x is more persuasive than "not
    permitted".
"""
from .contracts import (
    FORBIDDEN,
    MEANINGLESS,
    OK,
    UNKNOWN,
    WITHIN_PHASE,
    contract_for,
    integration_error_factor,
)

# Scope names. Chosen so that a reader of the output cannot mistake one for the other.
BUSY = "busy_mean"
ALL_DEVICE = "all_device_mean"
PER_SAMPLE_SUM = "per_sample_sum_mean"


class Value:
    """One aggregate, with the contract's opinion of it attached."""

    __slots__ = ("column", "op", "scope", "value", "unit", "status", "n", "note")

    def __init__(self, column, op, scope, value, unit, status, n, note=""):
        self.column = column
        self.op = op
        self.scope = scope
        self.value = value
        self.unit = unit
        self.status = status        # ok | info | meaningless | forbidden | unknown
        self.n = n
        self.note = note

    @property
    def refused(self):
        return self.value is None

    def as_dict(self):
        return {"column": self.column, "op": self.op, "scope": self.scope,
                "value": self.value, "unit": self.unit, "status": self.status,
                "n": self.n, "note": self.note}

    def __repr__(self):
        return "Value({}.{}.{}={} [{}])".format(
            self.column, self.op, self.scope, self.value, self.status)


def _rows(series, segment, gpus=None):
    for r in series.readings:
        if segment is not None and not (segment.i0 <= r.index <= segment.i1):
            continue
        if gpus is not None and r.gpu not in gpus:
            continue
        yield r


def _values(series, segment, column, gpus=None):
    return [r.values[column] for r in _rows(series, segment, gpus)
            if column in r.values]


def mean(series, segment, column, scope=BUSY):
    """Average `column` over `segment`, with scope stated in the result."""
    contract = contract_for(column)
    if contract is None:
        return Value(column, "mean", scope, None, "?", UNKNOWN, 0,
                     "no contract on file for this column; refusing to aggregate a "
                     "column whose meaning has not been established")

    verdict = contract.verdict("mean")
    gpus = set(segment.busy_gpus) if (scope == BUSY and segment is not None) else None
    vals = _values(series, segment, column, gpus)
    if not vals:
        return Value(column, "mean", scope, None, contract.unit, UNKNOWN, 0,
                     "no samples of this column in this segment")

    value = sum(vals) / len(vals)
    if verdict == WITHIN_PHASE and segment is None:
        return Value(column, "mean", scope, None, contract.unit, FORBIDDEN, len(vals),
                     "mean is only defined within one phase for this column, and this "
                     "call covers the whole file -- a mean over load plus idle plus "
                     "steady describes none of them")
    status = OK if verdict in (OK, WITHIN_PHASE) else verdict
    note = "" if status == OK else contract.why
    return Value(column, "mean", scope, value, contract.unit, status, len(vals), note)


def peak(series, segment, column, scope=BUSY):
    contract = contract_for(column)
    if contract is None:
        return Value(column, "peak", scope, None, "?", UNKNOWN, 0,
                     "no contract on file for this column")
    verdict = contract.verdict("peak")
    gpus = set(segment.busy_gpus) if (scope == BUSY and segment is not None) else None
    vals = _values(series, segment, column, gpus)
    if not vals:
        return Value(column, "peak", scope, None, contract.unit, UNKNOWN, 0,
                     "no samples of this column in this segment")
    if verdict == MEANINGLESS:
        return Value(column, "peak", scope, None, contract.unit, MEANINGLESS, len(vals),
                     "the maximum of a {} ms window sampled every sample period reports "
                     "whether a sample landed on a burst, not how large the bursts were"
                     .format(contract.window_ms))
    return Value(column, "peak", scope, max(vals), contract.unit,
                 OK if verdict == OK else verdict, len(vals),
                 "" if verdict == OK else contract.why)


def per_sample_sum(series, segment, column):
    """Sum across the busy cards within each sample, then average those sums.

    The right order for a rate: 8 cards each moving 0.5 GB/s is 4 GB/s of traffic at
    that instant. Doing it the other way -- average per card, then multiply by the
    card count -- gives the same answer only when every card is symmetric, and the
    per-card detail in the field data shows ring all-reduce is not symmetric at any
    instant.
    """
    contract = contract_for(column)
    if contract is None:
        return Value(column, "sum", PER_SAMPLE_SUM, None, "?", UNKNOWN, 0,
                     "no contract on file for this column")
    if contract.verdict("sum_across_gpus") == FORBIDDEN:
        return Value(column, "sum", PER_SAMPLE_SUM, None, contract.unit, FORBIDDEN, 0,
                     contract.why)

    gpus = set(segment.busy_gpus) if segment is not None else None
    by_index = {}
    for r in _rows(series, segment, gpus):
        if column in r.values:
            by_index[r.index] = by_index.get(r.index, 0.0) + r.values[column]
    if not by_index:
        return Value(column, "sum", PER_SAMPLE_SUM, None, contract.unit, UNKNOWN, 0,
                     "no samples of this column in this segment")
    sums = list(by_index.values())
    return Value(column, "sum", PER_SAMPLE_SUM, sum(sums) / len(sums), contract.unit,
                 OK, len(sums))


def integrate(series, segment, column, interval_s):
    """`mean x elapsed`. Legal for power; for the windowed rates, refuse with a number."""
    contract = contract_for(column)
    if contract is None:
        return Value(column, "integrate", BUSY, None, "?", UNKNOWN, 0,
                     "no contract on file for this column")

    verdict = contract.verdict("integrate")
    m = mean(series, segment, column, scope=BUSY)
    if m.refused:
        return Value(column, "integrate", BUSY, None, "?", m.status, m.n, m.note)

    n_samples = len({r.index for r in _rows(series, segment, None)})
    elapsed = (segment.duration_s() if segment is not None else series.span_s())
    if elapsed is None and interval_s:
        elapsed = n_samples * interval_s

    if verdict == FORBIDDEN:
        factor = integration_error_factor(contract, interval_s) if interval_s else None
        detail = ("the counter observes {} ms of each {:.3f} s interval, so it is looking "
                  "{:.2f}% of the time and this product understates the true total by "
                  "about {:.0f}x".format(contract.window_ms, interval_s,
                                         100.0 / factor, factor)
                  if factor else
                  "this column summarises a fixed hardware window, so multiplying its "
                  "mean by elapsed time scales the answer by the unmeasured duty cycle; "
                  "the interval is unknown here so the size of the error is unknown too")
        return Value(column, "integrate", BUSY, None, contract.unit + "*s", FORBIDDEN,
                     m.n, detail)

    if elapsed is None:
        return Value(column, "integrate", BUSY, None, contract.unit + "*s", UNKNOWN, m.n,
                     "no timestamps and no interval, so elapsed time is unknown")
    return Value(column, "integrate", BUSY, m.value * elapsed, contract.unit + "*s",
                 OK, m.n)


def energy_j(series, segment, interval_s=None):
    """Total energy across the busy cards over a segment, in joules.

    Deliberately built on `per_sample_sum` and not on the per-card mean. The first
    version integrated the per-card mean and labelled the result
    `per_sample_sum_mean`, which on an 8-card configuration reported one card's
    energy under a name that promised all eight -- out by exactly the card count.
    A mislabelled scope is the failure this module exists to prevent, so it is worth
    recording that it happened here first.
    """
    total = per_sample_sum(series, segment, "power_w")
    if total.refused:
        return Value("power_w", "energy", PER_SAMPLE_SUM, None, "J", total.status,
                     total.n, total.note)
    elapsed = (segment.duration_s() if segment is not None else series.span_s())
    if elapsed is None and interval_s:
        elapsed = len({r.index for r in _rows(series, segment, None)}) * interval_s
    if elapsed is None:
        return Value("power_w", "energy", PER_SAMPLE_SUM, None, "J", UNKNOWN, total.n,
                     "no timestamps and no interval, so elapsed time is unknown")
    n_cards = len(segment.busy_gpus) if segment is not None and segment.busy_gpus \
        else len(series.gpus)
    return Value("power_w", "energy", PER_SAMPLE_SUM, total.value * elapsed, "J", OK,
                 total.n,
                 "W x s = J across the {} busy card(s); power is an instantaneous "
                 "reading of a continuous quantity, which is what makes this one legal "
                 "while the PCIe rates are not".format(n_cards))


def summarise(series, segment, interval_s=None, columns=None):
    """Every aggregate worth reporting for one segment, refusals included.

    Order matters for reading: the two scopes of sm_pct sit next to each other so a
    reader sees 99.0% and 12.5% together and cannot quote one without the other.
    """
    cols = columns or [c for c in series.columns if contract_for(c)]
    out = []
    if "sm_pct" in cols:
        out.append(mean(series, segment, "sm_pct", scope=BUSY))
        out.append(mean(series, segment, "sm_pct", scope=ALL_DEVICE))
    for col in cols:
        if col == "sm_pct":
            continue
        contract = contract_for(col)
        out.append(mean(series, segment, col, scope=BUSY))
        if contract.verdict("sum_across_gpus") != FORBIDDEN and len(series.gpus) > 1:
            out.append(per_sample_sum(series, segment, col))
        p = peak(series, segment, col)
        if p.status != OK or contract.kind == "instantaneous":
            out.append(p)
    if "power_w" in cols:
        out.append(energy_j(series, segment, interval_s))
    if "pcie_tx_gbs" in cols:
        out.append(integrate(series, segment, "pcie_tx_gbs", interval_s))
    return out
