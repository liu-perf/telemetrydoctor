"""What arithmetic each telemetry column supports.

This is the centre of the library. Every other module asks questions of it.

The other tools in this series each carry one honesty field. `regressiondoctor`
attaches `noise_basis` to a verdict, so a reader can see whether the threshold was
measured or assumed. `fitdoctor` attaches `basis` to a number, so a reader can see
whether the figure was measured, taken off a spec sheet, derived, or assumed. Both
answer "where did this number come from".

This one answers a different question: **what may I do with it.**

The reason a telemetry column needs that is that the columns in one CSV, produced
by one sampling loop, on one machine, in one second, do not support the same
operations as each other:

  * `power.draw` is an instantaneous reading of a continuous quantity. Averaging
    it is fine. Multiplying the average by elapsed time gives you joules, and
    that is a real number you can put on a power bill.
  * `pcie.tx_bytes` throughput from NVML is the average over "approximately the
    last 20 ms". Averaging the samples estimates the mean rate. Multiplying that
    by elapsed time does NOT give you total bytes -- it gives you total bytes
    scaled by the fraction of time the counter was actually looking, which at a
    1.376 s sampling interval is 1.45%.
  * `utilization.gpu` is the fraction of the driver's sample window in which at
    least one kernel was resident. Averaging within a phase is fine. Summing it
    across GPUs is meaningless -- percentages do not add. And reading it as
    "how hard the card is working" is wrong in a way that has a measured size:
    two workloads ~1000x apart in delivered FLOPS both read ~28% on one RTX
    5060 Ti (see `examples/verify_on_device.py`).

A contract does not stop you doing arithmetic. It makes the tool say so.
"""

# Verdict vocabulary for each operation. Deliberately four words, not a boolean:
# "forbidden" and "meaningless" fail for different reasons and want different
# sentences in the report, and "unknown" is not a quiet "ok".
OK = "ok"                       # the operation is sound
WITHIN_PHASE = "within_phase"   # sound, but only over samples from one phase
MEANINGLESS = "meaningless"     # the operation runs and returns a number that means nothing
FORBIDDEN = "forbidden"         # the operation produces a wrong answer of known size
UNKNOWN = "unknown"             # nobody has established this either way

OPERATIONS = ("mean", "peak", "integrate", "sum_across_gpus")

# NVML's documented utilization sample window is a range, not a value: "between 1
# second and 1/6 second". A range is what gets stored, because collapsing it to a
# midpoint would be the exact move this library exists to object to.
UTILIZATION_WINDOW_MS = (167, 1000)
# nvmlDeviceGetPcieThroughput: "the average over the last ~20 ms".
PCIE_WINDOW_MS = 20


class Contract:
    """What one telemetry column supports, and why.

    `window_ms` is the width of the hardware/driver window the reading summarises,
    or None when the reading is instantaneous, or a (lo, hi) tuple when the vendor
    documents a range instead of a number.
    """

    __slots__ = ("column", "unit", "kind", "window_ms", "ops", "semantics", "why")

    def __init__(self, column, unit, kind, window_ms, ops, semantics, why):
        self.column = column
        self.unit = unit
        self.kind = kind                # instantaneous | windowed_rate | time_fraction | counter
        self.window_ms = window_ms
        self.ops = ops
        self.semantics = semantics
        self.why = why

    def verdict(self, op):
        if op not in OPERATIONS:
            raise KeyError("unknown operation: {!r}".format(op))
        return self.ops.get(op, UNKNOWN)

    def duty_cycle(self, interval_s):
        """Fraction of elapsed time this column was actually observing.

        None when the column is instantaneous or its window is undocumented as a
        single value. This is the number that decides whether integrating is a
        small error or a 69x one.
        """
        if not isinstance(self.window_ms, int) or interval_s <= 0:
            return None
        return (self.window_ms / 1000.0) / interval_s

    def __repr__(self):
        return "Contract({!r})".format(self.column)


_C = Contract

CONTRACTS = {
    "power_w": _C(
        column="power_w", unit="W", kind="instantaneous", window_ms=None,
        ops={"mean": OK, "peak": OK, "integrate": OK, "sum_across_gpus": OK},
        semantics="board power draw at the moment of the read",
        why="A continuous quantity sampled instantaneously. Mean x elapsed = joules, "
            "and that is the one integration in a telemetry CSV that is actually "
            "sound. It is in this table mostly to prove the table is not just a list "
            "of prohibitions."),

    "sm_pct": _C(
        column="sm_pct", unit="%", kind="time_fraction", window_ms=UTILIZATION_WINDOW_MS,
        ops={"mean": WITHIN_PHASE, "peak": OK, "integrate": FORBIDDEN,
             "sum_across_gpus": FORBIDDEN},
        semantics="fraction of the driver's sample window in which at least one kernel "
                  "was resident -- NOT SM occupancy, NOT tensor-core activity",
        why="Percentages do not add, so summing across GPUs is forbidden: on an 8-card "
            "node one busy card gives an all-card mean of 12.5%, which has been read as "
            "'the card is barely running' when the busy card was at 99.0%. Averaging is "
            "sound only inside one phase, because a mean over load+idle describes "
            "neither. And it is a time measure: on one RTX 5060 Ti a 64x64 matmul "
            "submitted back-to-back and a 4096x4096 matmul at quarter duty both read "
            "~28% while delivering 0.046 and 45.4 TFLOPS -- 978x apart."),

    "mem_pct": _C(
        column="mem_pct", unit="%", kind="time_fraction", window_ms=UTILIZATION_WINDOW_MS,
        ops={"mean": WITHIN_PHASE, "peak": OK, "integrate": FORBIDDEN,
             "sum_across_gpus": FORBIDDEN},
        semantics="fraction of the sample window in which memory was being read or "
                  "written -- NOT the fraction of VRAM in use",
        why="Same shape as sm_pct, and the name misleads in a second way: people read "
            "'mem 7%' as 'using 7% of VRAM'. It is a duty cycle on the memory "
            "interface. For occupancy of the framebuffer, read the bytes column."),

    "pcie_tx_gbs": _C(
        column="pcie_tx_gbs", unit="GB/s", kind="windowed_rate", window_ms=PCIE_WINDOW_MS,
        ops={"mean": OK, "peak": MEANINGLESS, "integrate": FORBIDDEN,
             "sum_across_gpus": OK},
        semantics="average transmit rate over approximately the 20 ms before the read",
        why="The window is 20 ms and the sampling interval on a real 8-card loop was "
            "1.376 s, so the counter is looking 1.45% of the time. The mean still "
            "estimates the mean rate. The peak does not estimate the peak rate -- it "
            "reports whether a sample happened to land on a burst, and all-reduce "
            "traffic is bursty by construction. Integrating is wrong by 1/duty, about "
            "69x. This is the column that produced a fabricated '8 cards, 47 GB/s'."),

    "pcie_rx_gbs": _C(
        column="pcie_rx_gbs", unit="GB/s", kind="windowed_rate", window_ms=PCIE_WINDOW_MS,
        ops={"mean": OK, "peak": MEANINGLESS, "integrate": FORBIDDEN,
             "sum_across_gpus": OK},
        semantics="average receive rate over approximately the 20 ms before the read",
        why="See pcie_tx_gbs. One extra trap on rx specifically: loading model weights "
            "is itself a multi-GB host-to-device transfer, so an rx mean taken over the "
            "whole file is mostly the weight load, not the training traffic."),

    "fb_used_mib": _C(
        column="fb_used_mib", unit="MiB", kind="instantaneous", window_ms=None,
        ops={"mean": WITHIN_PHASE, "peak": OK, "integrate": FORBIDDEN,
             "sum_across_gpus": OK},
        semantics="framebuffer bytes in use at the moment of the read",
        why="A level, not a rate: integrating a level over time answers no question. "
            "The peak is the number that decides whether the job fits, so unlike the "
            "rate columns the peak here is the useful one -- but a caching allocator "
            "means it is the allocator's peak, not the workload's."),

    "sm_clock_mhz": _C(
        column="sm_clock_mhz", unit="MHz", kind="instantaneous", window_ms=None,
        ops={"mean": WITHIN_PHASE, "peak": OK, "integrate": FORBIDDEN,
             "sum_across_gpus": FORBIDDEN},
        semantics="SM clock at the moment of the read",
        why="Averaging across a phase boundary mixes the idle clock with the loaded "
            "one and lands somewhere neither occurs: on this machine idle sits at "
            "180 MHz. Summing clocks across cards is not an operation."),

    "temp_c": _C(
        column="temp_c", unit="degC", kind="instantaneous", window_ms=None,
        ops={"mean": WITHIN_PHASE, "peak": OK, "integrate": FORBIDDEN,
             "sum_across_gpus": FORBIDDEN},
        semantics="core temperature at the moment of the read",
        why="A level with a long time constant, so the mean of a short phase is "
            "really a statement about the phase before it."),
}

# Column-name spellings seen in the wild, mapped onto the canonical names above.
# `nvidia-smi dmon` uses `sm`/`mem`/`pwr`; `--query-gpu` uses `utilization.gpu`;
# the field's pynvml loop wrote `sm_pct` and `tx_gbs` per card.
#
# `gpu` is deliberately NOT in this table. In `dmon` output the leading column is
# headed `# gpu` and holds the card index; elsewhere `gpu` is shorthand for
# utilisation. An alias that resolves one spelling to two different meanings is
# worse than no alias, so the dmon parser handles its index column by position.
ALIASES = {
    "utilization.gpu": "sm_pct", "util.gpu": "sm_pct", "sm": "sm_pct",
    "utilization.memory": "mem_pct", "util.mem": "mem_pct", "mem": "mem_pct",
    "power.draw": "power_w", "pwr": "power_w", "power": "power_w",
    "tx_gbs": "pcie_tx_gbs", "tx": "pcie_tx_gbs", "pcie.tx": "pcie_tx_gbs",
    "rx_gbs": "pcie_rx_gbs", "rx": "pcie_rx_gbs", "pcie.rx": "pcie_rx_gbs",
    "memory.used": "fb_used_mib", "fb": "fb_used_mib", "fb_used": "fb_used_mib",
    "clocks.sm": "sm_clock_mhz", "pclk": "sm_clock_mhz", "sclk": "sm_clock_mhz",
    "temperature.gpu": "temp_c", "gtemp": "temp_c", "temp": "temp_c",
}


def canonical(name):
    """Map a column spelling onto a canonical name, or None if unrecognised."""
    key = name.strip().lower().lstrip("#").strip()
    if key in CONTRACTS:
        return key
    return ALIASES.get(key)


def contract_for(name):
    """-> Contract, or None when the column has no contract on file.

    Returning None matters: an unknown column must not silently inherit a
    permissive default, because the whole point is that permissiveness is what
    goes wrong. Callers report `unknown` and refuse to aggregate.
    """
    canon = canonical(name)
    return CONTRACTS.get(canon) if canon else None


def integration_error_factor(contract, interval_s):
    """How wrong `mean x elapsed` would be for this column at this interval.

    -> float multiplier, or None when the column is integrable or has no single
    documented window. A factor of 69 means the true total is ~69x the computed
    one, because the counter only observed 1/69 of the elapsed time.
    """
    if contract.verdict("integrate") != FORBIDDEN:
        return None
    duty = contract.duty_cycle(interval_s)
    if not duty:
        return None
    return 1.0 / duty
