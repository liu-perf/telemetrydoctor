"""Cut a telemetry file into phases, without a hard-coded threshold.

Why this module is not four lines of `if sm >= 90`.

The field procedure this replaces says: bin the samples by which cards have
`sm >= 90`, and treat that set as the configuration label. It works, and the
analysis built on it was right. But 90 is a constant that was chosen by looking at
one workload, and the same three-machine dataset contains the counter-example:

    detection training (yolox)          plateau sm ~= 48.4%
    inference decode (Qwen3-VL)         plateau sm ~= 69.9%   (never reaches 90)
    segmentation training (KNet-Swin-L) plateau sm ~= 94.1%

Three real workloads on 8-card nodes, three plateaux, and `sm >= 90` selects zero
rows on two of them. The field script had to be re-run with `sm >= 20` for the
inference file -- a second constant, chosen the same way, for the same reason.

So the threshold is computed from the file. `otsu()` is the standard one-dimensional
two-class split: try every candidate cut, keep the one that maximises the variance
*between* the two groups. It needs no libraries and no parameters.

The part that matters more than the threshold is the refusal. Otsu will happily cut
a unimodal distribution straight down the middle and report a number. So the split
is only accepted when the two classes are actually separated -- `separability`, the
between-class share of total variance, has to clear a floor. Under that floor the
answer is `unknown` and no phases are emitted. A file where "busy" and "idle" are
not distinguishable is a file that cannot be phase-segmented, and saying so is the
only correct output.
"""
IDLE = "idle"
LOAD = "load"
ACTIVE = "active"

# Between-class share of total variance. Below this the two "classes" are one class.
#
# 0.90, and the value is measured rather than chosen. Otsu always returns *a* cut, so
# the floor has to sit above what a distribution with no real split scores:
#
#     uniform over one band, nothing to cut   separability 0.77
#     two plateaux with idle between them     separability 0.99 - 1.00
#
# The first version of this used 0.70, which let the deliberately unimodal fixture
# through and produced a confident threshold of 49.17 on a file whose whole point is
# that it cannot be segmented. tests/test_segment.py pins both numbers, so if either
# moves the floor gets revisited instead of quietly stopping working.
MIN_SEPARABILITY = 0.90
# A run shorter than this many samples is a transition, not a phase.
MIN_PHASE_SAMPLES = 3
# Signature of a weight load rather than work: SMs under this, receive above it.
# The 14% ceiling is the field note's own observation ("加载期间 SM 只有 0-14%"),
# rounded up; the 1 GB/s floor is three orders of magnitude above a measured idle
# floor of ~0.006 GB/s, so it cannot be tripped by driver polling noise.
LOAD_SM_CEILING = 20.0
LOAD_RX_FLOOR = 1.0


class Threshold:
    __slots__ = ("value", "basis", "separability", "n_values", "low_mean", "high_mean",
                 "high_median")

    def __init__(self, value, basis, separability, n_values, low_mean, high_mean,
                 high_median=None):
        self.value = value
        self.basis = basis                  # 'otsu' | 'explicit' | 'unknown'
        self.separability = separability
        self.n_values = n_values
        self.low_mean = low_mean
        self.high_mean = high_mean
        # The median of the busy readings, not their mean. This is the figure that
        # represents "what a card that is really working reads", and it is what
        # `crosscheck` divides by. The mean is dragged down by any card sitting at
        # part load -- and dragged down by exactly the amount that makes the second
        # segmentation agree with the first, which made the cross-check an algebraic
        # identity rather than a check. See crosscheck.py.
        self.high_median = high_median

    @property
    def usable(self):
        return self.basis == "explicit" or (
            self.basis == "otsu" and self.separability is not None
            and self.separability >= MIN_SEPARABILITY)

    def __repr__(self):
        return "Threshold({}, basis={}, sep={})".format(
            self.value, self.basis, self.separability)


class Segment:
    __slots__ = ("kind", "i0", "i1", "busy_gpus", "t0", "t1", "n_samples")

    def __init__(self, kind, i0, i1, busy_gpus, t0, t1, n_samples):
        self.kind = kind
        self.i0 = i0
        self.i1 = i1
        self.busy_gpus = tuple(sorted(busy_gpus))
        self.t0 = t0
        self.t1 = t1
        self.n_samples = n_samples

    @property
    def label(self):
        """The configuration label: which cards were busy. '{}' for an idle stretch."""
        return "{" + ",".join(str(g) for g in self.busy_gpus) + "}"

    def duration_s(self):
        if self.t0 is None or self.t1 is None:
            return None
        return self.t1 - self.t0

    def __repr__(self):
        return "Segment({} {} n={})".format(self.kind, self.label, self.n_samples)


def otsu(values, bins=256):
    """One-dimensional two-class threshold. -> (threshold, separability).

    separability is the between-class share of total variance, in [0, 1]. Returns
    (None, None) when there is nothing to split.
    """
    vals = [v for v in values if v is not None]
    if len(vals) < 4:
        return None, None
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        return None, None

    n = len(vals)
    mean_all = sum(vals) / n
    total_var = sum((v - mean_all) ** 2 for v in vals) / n
    if total_var <= 0:
        return None, None

    # histogram, so the sweep is O(bins) rather than O(n^2)
    counts = [0] * bins
    width = (hi - lo) / bins
    for v in vals:
        k = int((v - lo) / width)
        counts[min(k, bins - 1)] += 1
    centres = [lo + (k + 0.5) * width for k in range(bins)]

    best = (None, -1.0)
    w0 = 0
    s0 = 0.0
    total_sum = sum(c * m for c, m in zip(counts, centres))
    for k in range(bins - 1):
        w0 += counts[k]
        s0 += counts[k] * centres[k]
        w1 = n - w0
        if w0 == 0 or w1 == 0:
            continue
        m0 = s0 / w0
        m1 = (total_sum - s0) / w1
        between = (w0 / n) * (w1 / n) * (m0 - m1) ** 2
        if between > best[1]:
            best = (lo + (k + 1) * width, between)

    threshold, between = best
    if threshold is None:
        return None, None
    return threshold, between / total_var


def busy_threshold(series, column="sm_pct", explicit=None):
    """Decide what counts as 'this card is working' for THIS file."""
    if explicit is not None:
        return Threshold(float(explicit), "explicit", None, 0, None, None)

    vals = series.column(column)
    if not vals:
        return Threshold(None, "unknown", None, 0, None, None)

    threshold, sep = otsu(vals)
    if threshold is None:
        return Threshold(None, "unknown", None, len(vals), None, None)

    low = [v for v in vals if v < threshold]
    high = sorted(v for v in vals if v >= threshold)
    median = None
    if high:
        mid = len(high) // 2
        median = high[mid] if len(high) % 2 else (high[mid - 1] + high[mid]) / 2.0
    return Threshold(threshold, "otsu", sep, len(vals),
                     sum(low) / len(low) if low else None,
                     sum(high) / len(high) if high else None,
                     median)


def _load_flags(series):
    """Per sample: does this look like a weight load rather than idle or work?

    A separate signal from the busy set, and it has to be, because a load stretch
    sits below the busy threshold -- so it has the same busy set as the idle gaps on
    either side of it. Grouping on the busy set alone therefore glues load and idle
    into one run, and the first version of this module did exactly that: 20 idle
    samples, 12 load samples and 4 idle samples came out as one 36-sample phase
    whose mean rx was 1.83 GB/s, which is neither the idle floor nor the load rate.

    Phase boundaries do not only occur where the set of busy cards changes.
    """
    n_gpus = max(1, len(series.gpus))
    flags = {}
    rx = dict(series.per_sample("pcie_rx_gbs"))
    sm = dict(series.per_sample("sm_pct"))
    for index in {r.index for r in series.readings}:
        if index not in rx:
            flags[index] = False
            continue
        rx_mean = sum(rx[index].values()) / n_gpus
        sm_mean = (sum(sm[index].values()) / n_gpus) if index in sm else 0.0
        flags[index] = rx_mean > LOAD_RX_FLOOR and sm_mean < LOAD_SM_CEILING
    return flags


def _busy_sets(series, column, threshold):
    """-> [(index, t, frozenset(busy gpus), load_flag)] in sample order."""
    times = {}
    for r in series.readings:
        if r.t is not None and r.index not in times:
            times[r.index] = r.t
    flags = _load_flags(series)
    out = []
    for index, per_gpu in series.per_sample(column):
        busy = frozenset(g for g, v in per_gpu.items() if v >= threshold)
        out.append((index, times.get(index), busy, flags.get(index, False)))
    return out


def _classify(series, seg_indices, busy, is_load=None):
    """Tell a weight-load stretch from real work, and from a genuine idle gap.

    Loading a model is a multi-gigabyte host-to-device transfer with the SMs mostly
    idle: rx high, sm low but not zero. That is the signature the field note
    describes -- "加载期间 SM 只有 0-14%" -- and mistaking it for training traffic is
    what produced a fabricated '8 cards, 47 GB/s'.

    The check deliberately does not depend on the busy threshold. A load stretch
    usually sits *below* it, so keying off the busy set would file the load away as
    idle -- which is exactly what the first version of this function did, and it
    quietly poisoned the measured noise floor with 3-8 GB/s of weight traffic.
    """
    if is_load is None:
        wanted = set(seg_indices)
        sm, rx = [], []
        for r in series.readings:
            if r.index not in wanted:
                continue
            if busy and r.gpu not in busy:
                continue
            if "sm_pct" in r.values:
                sm.append(r.values["sm_pct"])
            if "pcie_rx_gbs" in r.values:
                rx.append(r.values["pcie_rx_gbs"])
        sm_mean = sum(sm) / len(sm) if sm else None
        rx_mean = sum(rx) / len(rx) if rx else None
        is_load = (sm_mean is not None and rx_mean is not None
                   and sm_mean < LOAD_SM_CEILING and rx_mean > LOAD_RX_FLOOR)
    if is_load:
        return LOAD
    return IDLE if not busy else ACTIVE


def segments(series, column="sm_pct", explicit_threshold=None,
             min_samples=MIN_PHASE_SAMPLES):
    """-> (list of Segment, Threshold). Empty list when the split is not usable."""
    th = busy_threshold(series, column, explicit_threshold)
    if not th.usable or th.value is None:
        return [], th

    marks = _busy_sets(series, column, th.value)
    if not marks:
        return [], th

    # Group on (busy set, load flag) together: either changing is a phase boundary.
    def key(m):
        return (m[2], m[3])

    runs = []
    start = 0
    for i in range(1, len(marks) + 1):
        if i == len(marks) or key(marks[i]) != key(marks[start]):
            runs.append((start, i - 1, marks[start][2], marks[start][3]))
            start = i

    out = []
    for i0, i1, busy, is_load in runs:
        n = i1 - i0 + 1
        if n < min_samples:
            continue
        idx = [marks[k][0] for k in range(i0, i1 + 1)]
        times = [marks[k][1] for k in range(i0, i1 + 1) if marks[k][1] is not None]
        out.append(Segment(_classify(series, idx, busy, is_load),
                           marks[i0][0], marks[i1][0], busy,
                           times[0] if times else None,
                           times[-1] if times else None, n))
    return out, th


def configurations(segs):
    """Group ACTIVE segments by busy-set label -> {label: [Segment, ...]}.

    A matrix run visits the same configuration once, but a run that was interrupted
    and restarted visits it twice, and quietly concatenating those is how two
    different machine states end up inside one mean.
    """
    out = {}
    for s in segs:
        if s.kind == ACTIVE:
            out.setdefault(s.label, []).append(s)
    return out
