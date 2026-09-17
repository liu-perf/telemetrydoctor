"""How often the file was really sampled, and whether that rate bought anything.

Two separate questions that both get answered wrong by reading the sampling loop's
source code instead of the file it produced.

**What the interval actually was.** A loop that ends in `sleep(1.0)` does not
produce one row per second; it produces one row per second plus however long the
body took. On one real 8-card node reading four NVML counters per card, the body
cost 0.37 s and the file came out at 1.376 s/row -- 37.6% over nominal. Any
per-second figure computed with the nominal value is 37.6% wrong, in the direction
that flatters the machine. On one card through `nvidia-smi -lms 250` the same
measurement gives +3.0%: small, still not zero, and the difference between the two
is the point -- the overshoot is a function of how much work the sampler does, so
it cannot be known without measuring.

**Whether the rate was useful.** NVML documents the utilisation sample window as
"between 1 second and 1/6 second". Poll at 250 ms and consecutive rows repeat: on
this machine, 252 rows carried 64 changes, so 75% of the rows were copies of the
row before. Those repeats are not independent samples. Treating them as such
shrinks every confidence interval by a factor that has nothing to do with the
measurement -- the classic way to make a noisy number look settled.
"""
from .contracts import UTILIZATION_WINDOW_MS

# Under this, calling something "n independent samples" is misleading enough to flag.
REPEAT_WARN_FRACTION = 0.25


class IntervalReport:
    __slots__ = ("measured_s", "nominal_s", "n_gaps", "spread_s", "basis",
                 "median_s", "min_s", "max_s")

    def __init__(self, measured_s, nominal_s, n_gaps, spread_s, basis,
                 median_s=None, min_s=None, max_s=None):
        self.measured_s = measured_s
        self.nominal_s = nominal_s
        self.n_gaps = n_gaps
        self.spread_s = spread_s
        self.basis = basis              # 'measured' | 'nominal_only' | 'unknown'
        self.median_s = median_s
        self.min_s = min_s
        self.max_s = max_s

    @property
    def overshoot_pct(self):
        if self.measured_s is None or not self.nominal_s:
            return None
        return (self.measured_s / self.nominal_s - 1.0) * 100.0

    @property
    def effective_s(self):
        """The number to actually divide by. None when nothing was measurable."""
        return self.measured_s if self.measured_s is not None else self.nominal_s


def measure(series, max_gap_s=30.0):
    """Measure the real sampling interval from the timestamps in the file.

    `max_gap_s` drops the gaps that are not sampling intervals at all -- a pause
    between two configurations in a matrix run, or the sampler being stopped and
    restarted. Including those would inflate the mean and, worse, would make the
    inflated mean look like a property of the sampler.
    """
    if not series.has_timestamps:
        return IntervalReport(None, series.nominal_interval_s, 0, None,
                              "nominal_only" if series.nominal_interval_s else "unknown")

    ts = series.times()
    gaps = [b - a for a, b in zip(ts, ts[1:]) if 0 < b - a <= max_gap_s]
    if not gaps:
        return IntervalReport(None, series.nominal_interval_s, 0, None,
                              "nominal_only" if series.nominal_interval_s else "unknown")

    gaps_sorted = sorted(gaps)
    mid = len(gaps_sorted) // 2
    median = (gaps_sorted[mid] if len(gaps_sorted) % 2
              else (gaps_sorted[mid - 1] + gaps_sorted[mid]) / 2.0)
    mean = sum(gaps) / len(gaps)
    return IntervalReport(mean, series.nominal_interval_s, len(gaps),
                          gaps_sorted[-1] - gaps_sorted[0], "measured",
                          median, gaps_sorted[0], gaps_sorted[-1])


class RepeatReport:
    __slots__ = ("column", "n_rows", "n_changes", "window_ms", "interval_s")

    def __init__(self, column, n_rows, n_changes, window_ms, interval_s):
        self.column = column
        self.n_rows = n_rows
        self.n_changes = n_changes
        self.window_ms = window_ms
        self.interval_s = interval_s

    @property
    def repeat_fraction(self):
        denom = max(1, self.n_rows - 1)
        return 1.0 - (self.n_changes / denom)

    @property
    def independent_rows(self):
        """Rows that could carry new information, given the driver's window.

        Not a correction to apply -- an order-of-magnitude statement about how many
        of the rows can possibly be independent.
        """
        if not self.interval_s or not self.window_ms:
            return None
        lo, hi = (self.window_ms if isinstance(self.window_ms, tuple)
                  else (self.window_ms, self.window_ms))
        return self.n_rows * min(1.0, self.interval_s / (hi / 1000.0))


def repeats(series, column, interval_s=None, segments=None):
    """How many card-samples of `column` differ from the one before.

    `segments` restricts the count to those stretches, and callers should pass the
    active ones. Counting over an idle stretch measures nothing: an idle card
    reports exactly 0.0 every time, so a file that is half idle scores ~50%
    "repeats" no matter how it was sampled. The first version of this did that and
    reported oversampling on a file sampled at 1.376 s, which is four times slower
    than the fastest window NVML documents -- a false positive produced by the
    exact kind of scope error the rest of the library is about.
    """
    from .contracts import contract_for

    contract = contract_for(column)
    window = contract.window_ms if contract else None
    spans = [(s.i0, s.i1, set(s.busy_gpus)) for s in segments] if segments else None

    n_rows = n_changes = 0
    for gpu in series.gpus:
        vals = []
        for r in sorted((r for r in series.readings if r.gpu == gpu),
                        key=lambda r: r.index):
            if column not in r.values:
                continue
            if spans is not None and not any(
                    i0 <= r.index <= i1 and (not busy or gpu in busy)
                    for i0, i1, busy in spans):
                continue
            vals.append(r.values[column])
        if len(vals) < 2:
            continue
        n_rows += len(vals)
        n_changes += sum(1 for a, b in zip(vals, vals[1:]) if a != b)
    return RepeatReport(column, n_rows, n_changes, window, interval_s)


def oversampled(interval_s):
    """-> True when the poll rate is faster than the slowest documented window.

    Uses the slow end of NVML's documented range on purpose. At the fast end
    (1/6 s) a 250 ms poll is fine; at the slow end (1 s) it is 4x too fast. Since
    the driver does not say which applies, the honest reading is that any interval
    under the slow end can be returning repeats -- and whether it did is a question
    for `repeats()`, which counts them instead of predicting them.
    """
    if not interval_s:
        return None
    return interval_s < (UTILIZATION_WINDOW_MS[1] / 1000.0)
