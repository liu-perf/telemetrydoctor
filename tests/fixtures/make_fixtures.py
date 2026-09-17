"""Generate the test fixtures. Deterministic; committed output lives beside it.

    python tests/fixtures/make_fixtures.py

Run it and `git diff` should be empty. The generator is committed rather than just
the CSVs because the shape of these files is an argument, and an argument you can
only read as 400 rows of numbers is not one anybody will check.

**Every fixture is synthetic and says so on its first line.** The *targets* they are
built to -- the per-configuration PCIe means, the busy plateau, the idle floor, the
1.376 s interval, the 545 W per busy card -- are figures from one real 8-card
capture, quoted with their provenance in `docs/rules.md`. No row here came off a
machine. The rule the whole series follows: generated data has to declare itself
in-band, in the data, not in a filename that gets lost the first time somebody
pastes the numbers into a slide.

Three properties make these fixtures worth having:

  * the PCIe columns are *bursty* by construction, reproducing the real ring
    all-reduce shape -- most samples at a floor, occasional samples near 3 GB/s.
    That is what makes `peak` meaningless, and a smooth fixture would have hidden it.
  * the busy plateau is a parameter, not 100. `EXAMPLE_detection_plateau48.csv`
    plateaus at 48.4% because one real detection job did, and a fixed `sm >= 90`
    threshold selects zero rows from it.
  * **the realised means do not match the targets exactly, and that is left alone.**
    Over 19-23 samples the number of bursts can only be an integer, so a single
    burst more or less moves a configuration's mean by several percent -- up to ~30%
    on the low-traffic 2-card case. That residual is not sloppiness to be tuned out:
    it is the same +-1-burst quantisation that makes a real capture's tx mean
    uncertain at these sample counts, which is exactly what TL004 is about. A
    fixture that matched to three decimals would be quietly asserting a precision
    that this kind of measurement does not have. `tests/test_fixtures.py` therefore
    asserts the targets with a tolerance derived from the burst height, and asserts
    that the tolerance is needed.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- deterministic rng
class Rng:
    """A 32-bit LCG. Not good randomness; perfectly reproducible randomness, which
    is the only property a fixture generator needs."""

    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFF

    def next(self):
        self.state = (1103515245 * self.state + 12345) & 0x7FFFFFFF
        return self.state

    def uniform(self, lo, hi):
        return lo + (hi - lo) * (self.next() / 0x7FFFFFFF)

    def chance(self, p):
        return (self.next() / 0x7FFFFFFF) < p


def stamp(t):
    """2026-07-31T12:36:14.123456 -- the shape the field sampler wrote."""
    import time
    whole = int(t)
    frac = t - whole
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(whole)) + \
        "{:.6f}".format(frac)[1:]


# ------------------------------------------------------------------ the 8-card matrix
# Per-configuration figures from one real capture. sum_* are across the busy cards.
N_GPUS = 8
INTERVAL = 1.376              # measured, against a nominal 1.0
BURST_P = 1.0 / 6.0           # a tx sample lands on a burst about one time in six
BURST_GBS = 3.0

CONFIGS = [
    # label,          busy set,          n,  sum_tx, sum_rx, busy_sm, busy_power
    ("1gpu",          (0,),              19, 0.008,  0.040,  99.0,     545.0),
    ("2gpu",          (0, 1),            23, 0.504,  0.374,  98.6,    1050.0),
    ("4gpu",          (0, 1, 2, 3),      18, 2.172,  1.733,  98.6,    2194.0),
    ("4gpu_split",    (0, 1, 5, 6),      21, 1.941,  1.632,  98.0,    2236.0),
    ("8gpu",          tuple(range(8)),   19, 4.118,  4.288,  98.3,    4386.0),
]

IDLE_TX_TOTAL = 0.00643       # GB/s across all 8 cards, measured on a real idle file
IDLE_RX_TOTAL = 0.00326
IDLE_POWER_EACH = 19.6
LOAD_SAMPLES = 12             # weight load: rx high, sm low -- the phase that must
LOAD_SM = (2.0, 12.0)         # not be averaged into the work
LOAD_RX_EACH = (3.0, 8.0)


def wide_header():
    cells = ["timestamp"]
    for g in range(N_GPUS):
        for m in ("tx_gbs", "rx_gbs", "sm_pct", "mem_pct", "power_w"):
            cells.append("gpu{}_{}".format(g, m))
    return ",".join(cells)


_burst_acc = {}


def bursty(rng, target_mean, key="default"):
    """A sample of a bursty rate whose long-run mean is exactly `target_mean`.

    Most samples sit at a floor; some fraction `p` land on a ~3 GB/s burst. Given the
    floor and the burst height, `p` is not free -- it is whatever makes the mean come
    out at the target:

        target = p * BURST + (1 - p) * floor    ->    p = (target - floor) / (BURST - floor)

    `p` is realised with a fractional accumulator per `key` rather than a coin flip or
    a fixed period, so the achieved burst fraction is exact to within one sample and
    the per-card streams cannot alias against each other.

    Every part of that is a correction. Three earlier versions of these six lines got
    the mean wrong three different ways: an asymmetric burst jitter (mean 0.95B, not
    B), one global counter that aliased against the sixteen calls per row until tx
    never burst at all, and a fixed p = 1/6 which for the single-card target of
    0.008 GB/s implied a mean of 0.503 -- sixty times too high, because bursting one
    sample in six at 3 GB/s cannot average to 0.008 whatever the floor is. A fixture
    whose stated mean is not its actual mean is worse than no fixture: every test
    written against it encodes the error instead of catching it.

    The single-card case is the one that matters most, and now falls out on its own:
    p comes to 0.0013, so across 19 samples there is no burst at all. That is the
    real finding about that configuration -- one card has no peer traffic, and its tx
    column sits on the noise floor.
    """
    floor = 0.004
    if target_mean <= floor:
        return max(0.0, rng.uniform(0.0, 2 * target_mean))
    p = min(BURST_P, (target_mean - floor) / (BURST_GBS - floor))
    acc = _burst_acc.get(key, 0.0) + p
    if acc >= 1.0:
        _burst_acc[key] = acc - 1.0
        return rng.uniform(BURST_GBS * 0.9, BURST_GBS * 1.1)
    _burst_acc[key] = acc
    # the floor carries whatever the bursts do not, so the mean lands on target
    rest = (target_mean - p * BURST_GBS) / (1.0 - p)
    rest = max(0.0, rest)
    return rng.uniform(rest * 0.6, rest * 1.4)


def row(t, per_gpu):
    cells = [stamp(t)]
    for g in range(N_GPUS):
        v = per_gpu[g]
        cells.extend("{:.5f}".format(v[k]) for k in ("tx", "rx"))
        cells.append("{:.1f}".format(v["sm"]))
        cells.append("{:.1f}".format(v["mem"]))
        cells.append("{:.1f}".format(v["power"]))
    return ",".join(cells)


def idle_gpu(rng):
    return {"tx": max(0.0, rng.uniform(0.0, 2 * IDLE_TX_TOTAL / N_GPUS)),
            "rx": max(0.0, rng.uniform(0.0, 2 * IDLE_RX_TOTAL / N_GPUS)),
            "sm": 0.0, "mem": 0.0,
            "power": rng.uniform(IDLE_POWER_EACH * 0.9, IDLE_POWER_EACH * 1.1)}


def build_matrix():
    rng = Rng(20260731)
    t = 1000000.0
    lines = [
        "# EXAMPLE -- synthetic telemetry, not a real machine. Generated by "
        "tests/fixtures/make_fixtures.py.",
        "# Shaped after one real 8-card capture: the per-configuration means, the "
        "98-99% busy plateau, the idle floor and the 1.376 s interval are the real "
        "figures. No row here came off a machine.",
        wide_header(),
    ]

    def emit(n, fn):
        nonlocal t
        for _ in range(n):
            lines.append(row(t, fn()))
            t += INTERVAL + rng.uniform(-0.02, 0.02)

    emit(20, lambda: {g: idle_gpu(rng) for g in range(N_GPUS)})

    # weight load: every card receiving, SMs nearly idle
    def load_row():
        out = {}
        for g in range(N_GPUS):
            out[g] = {"tx": rng.uniform(0.0, 0.02),
                      "rx": rng.uniform(*LOAD_RX_EACH),
                      "sm": rng.uniform(*LOAD_SM), "mem": rng.uniform(1.0, 9.0),
                      "power": rng.uniform(85.0, 110.0)}
        return out
    emit(LOAD_SAMPLES, load_row)
    emit(4, lambda: {g: idle_gpu(rng) for g in range(N_GPUS)})

    for _, busy, n, sum_tx, sum_rx, busy_sm, busy_power in CONFIGS:
        k = len(busy)
        tx_each, rx_each = sum_tx / k, sum_rx / k
        pw_each = busy_power / k

        def make(busy=busy, tx_each=tx_each, rx_each=rx_each,
                 busy_sm=busy_sm, pw_each=pw_each):
            out = {}
            for g in range(N_GPUS):
                if g in busy:
                    out[g] = {"tx": bursty(rng, tx_each, ("tx", g)),
                              "rx": bursty(rng, rx_each, ("rx", g)),
                              "sm": rng.uniform(busy_sm - 1.2, min(100.0, busy_sm + 1.0)),
                              "mem": rng.uniform(40.0, 62.0),
                              "power": rng.uniform(pw_each * 0.97, pw_each * 1.03)}
                else:
                    out[g] = idle_gpu(rng)
            return out
        emit(n, make)
        emit(4, lambda: {g: idle_gpu(rng) for g in range(N_GPUS)})

    return "\n".join(lines) + "\n"


# --------------------------------------------------- a plateau a fixed 90 would miss
def build_plateau(plateau, seed, note):
    rng = Rng(seed)
    t = 2000000.0
    lines = [
        "# EXAMPLE -- synthetic telemetry, not a real machine. "
        "Generated by tests/fixtures/make_fixtures.py.",
        "# " + note,
        wide_header(),
    ]

    def emit(n, fn):
        nonlocal t
        for _ in range(n):
            lines.append(row(t, fn()))
            t += INTERVAL + rng.uniform(-0.02, 0.02)

    emit(12, lambda: {g: idle_gpu(rng) for g in range(N_GPUS)})
    for busy in ((0,), (0, 1, 2, 3), tuple(range(8))):
        def make(busy=busy):
            out = {}
            for g in range(N_GPUS):
                if g in busy:
                    out[g] = {"tx": bursty(rng, 0.4, ("tx", g)),
                              "rx": bursty(rng, 0.35, ("rx", g)),
                              "sm": rng.uniform(plateau - 1.5, plateau + 1.5),
                              "mem": rng.uniform(30.0, 50.0),
                              "power": rng.uniform(255.0, 270.0)}
                else:
                    out[g] = idle_gpu(rng)
            return out
        emit(18, make)
        emit(4, lambda: {g: idle_gpu(rng) for g in range(N_GPUS)})
    return "\n".join(lines) + "\n"


# --------------------------------------------------------- nothing to segment
def build_unimodal():
    rng = Rng(4242)
    t = 3000000.0
    lines = [
        "# EXAMPLE -- synthetic telemetry, not a real machine. "
        "Generated by tests/fixtures/make_fixtures.py.",
        "# Deliberately unimodal: sm wanders between 30 and 70 with no idle/busy gap. "
        "There is no honest place to cut this file, and the tool must say so rather "
        "than take Otsu's answer on a single mode.",
        wide_header(),
    ]
    for _ in range(90):
        per = {}
        for g in range(N_GPUS):
            per[g] = {"tx": rng.uniform(0.2, 0.6), "rx": rng.uniform(0.2, 0.6),
                      "sm": rng.uniform(30.0, 70.0), "mem": rng.uniform(20.0, 40.0),
                      "power": rng.uniform(300.0, 360.0)}
        lines.append(row(t, per))
        t += INTERVAL + rng.uniform(-0.02, 0.02)
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------- other formats
def build_dmon():
    rng = Rng(77)
    lines = [
        "# EXAMPLE -- synthetic nvidia-smi dmon capture, not a real machine.",
        "# Recorded without -o DT, so there are no timestamps at all: the interval "
        "cannot be measured from this file, only assumed. That is the point of it.",
        "# gpu    pwr  gtemp  mtemp     sm    mem    enc    dec   mclk   pclk",
        "# Idx      W      C      C      %      %      %      %    MHz    MHz",
    ]
    for i in range(60):
        busy = 20 <= i < 50
        for g in range(2):
            sm = rng.uniform(96.0, 99.5) if busy else 0.0
            pwr = rng.uniform(520.0, 560.0) if busy else rng.uniform(18.0, 22.0)
            lines.append("{:7d}{:7.0f}{:7.0f}{:>7}{:7.0f}{:7.0f}{:7d}{:7d}{:7d}{:7d}"
                         .format(g, pwr, 45 if busy else 32, "-", sm,
                                 rng.uniform(30, 55) if busy else 0, 0, 0, 9501,
                                 2610 if busy else 180))
    return "\n".join(lines) + "\n"


def build_query_gpu():
    rng = Rng(99)
    t = 4000000.0
    lines = [
        "# EXAMPLE -- synthetic nvidia-smi --query-gpu capture, not a real machine.",
        "# Long form: one row per card per sample. --format=csv,noheader deletes the "
        "record of which column is which, so --columns has to be supplied.",
    ]
    import time
    for i in range(60):
        busy = 15 <= i < 45
        for g in range(2):
            sm = rng.uniform(97.0, 99.6) if busy else 0.0
            pwr = rng.uniform(520.0, 555.0) if busy else rng.uniform(18.0, 22.0)
            ts = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime(int(t))) + \
                "{:.3f}".format(t - int(t))[1:]
            lines.append("{}, {}, {:.0f}, {:.2f}".format(ts, g, sm, pwr))
        t += 0.2578
    return "\n".join(lines) + "\n"


def build_single_gpu():
    rng = Rng(1234)
    t = 5000000.0
    lines = [
        "# EXAMPLE -- synthetic single-card telemetry, not a real machine.",
        "# One card, so busy-card mean and all-card mean are the same number: the "
        "scope rule has an ok path and this fixture is it.",
        "timestamp,gpu0_tx_gbs,gpu0_rx_gbs,gpu0_sm_pct,gpu0_mem_pct,gpu0_power_w",
    ]
    for i in range(70):
        busy = 15 <= i < 55
        lines.append("{},{:.5f},{:.5f},{:.1f},{:.1f},{:.1f}".format(
            stamp(t),
            rng.uniform(0.0, 0.002), rng.uniform(0.0, 0.002),
            rng.uniform(97.5, 99.5) if busy else 0.0,
            rng.uniform(35, 55) if busy else 0.0,
            rng.uniform(120.0, 135.0) if busy else rng.uniform(2.2, 2.7)))
        t += INTERVAL + rng.uniform(-0.02, 0.02)
    return "\n".join(lines) + "\n"


FIXTURES = {
    "EXAMPLE_matrix_8gpu.csv": build_matrix,
    "EXAMPLE_detection_plateau48.csv": lambda: build_plateau(
        48.4, 909090,
        "Busy plateau 48.4%, the figure one real detection training job reported on "
        "8 cards. A fixed `sm >= 90` threshold selects zero rows from this file, "
        "which is why the threshold is computed per file."),
    "EXAMPLE_segmentation_plateau94.csv": lambda: build_plateau(
        94.1, 606060,
        "Busy plateau 94.1%, from a real segmentation training job on the same 8 "
        "cards as the 48.4% detection file. Same hardware, same day, twice the "
        "reading -- and neither number ranks the two jobs."),
    "EXAMPLE_unimodal.csv": build_unimodal,
    "EXAMPLE_dmon_no_timestamps.txt": build_dmon,
    "EXAMPLE_query_gpu.csv": build_query_gpu,
    "EXAMPLE_single_gpu.csv": build_single_gpu,
}


def main():
    for name, fn in sorted(FIXTURES.items()):
        path = os.path.join(HERE, name)
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(fn())
        print("wrote {} ({:,} bytes)".format(name, os.path.getsize(path)))


if __name__ == "__main__":
    main()
