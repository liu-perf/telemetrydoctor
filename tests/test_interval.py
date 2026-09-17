"""Measured interval versus nominal, and what a repeat count is allowed to mean."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from telemetrydoctor import Context, load  # noqa: E402
from telemetrydoctor import interval as I  # noqa: E402
from telemetrydoctor import parse as P  # noqa: E402
from telemetrydoctor import segment as S  # noqa: E402

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
MATRIX = os.path.join(FIXTURES, "EXAMPLE_matrix_8gpu.csv")
DMON = os.path.join(FIXTURES, "EXAMPLE_dmon_no_timestamps.txt")
QUERY = os.path.join(FIXTURES, "EXAMPLE_query_gpu.csv")


def test_the_measured_interval_is_1_376_not_the_nominal_1_0():
    """The figure this whole rule exists for."""
    s = load(MATRIX, nominal_interval_s=1.0)
    iv = I.measure(s)
    assert iv.basis == "measured"
    assert iv.measured_s == pytest.approx(1.376, abs=0.01)
    assert iv.overshoot_pct == pytest.approx(37.6, abs=1.5)


def test_the_effective_interval_is_the_measured_one_when_both_exist():
    s = load(MATRIX, nominal_interval_s=1.0)
    iv = I.measure(s)
    assert iv.effective_s == iv.measured_s
    assert iv.effective_s != iv.nominal_s


def test_a_file_with_no_timestamps_has_no_measured_interval():
    s = load(DMON)
    iv = I.measure(s)
    assert iv.measured_s is None
    assert iv.basis == "unknown"
    assert iv.effective_s is None, (
        "with nothing measured and nothing supplied, the answer is None -- not a "
        "plausible default")


def test_a_supplied_nominal_is_used_but_labelled_as_unconfirmed():
    s = load(DMON, nominal_interval_s=1.0)
    iv = I.measure(s)
    assert iv.basis == "nominal_only"
    assert iv.effective_s == 1.0
    assert iv.measured_s is None


def test_gaps_between_configurations_do_not_inflate_the_interval():
    """A 40 s pause between two runs is not a sampling interval."""
    rows = ["timestamp,gpu0_sm_pct"]
    t = 0
    for _ in range(10):
        rows.append("2026-07-30T10:00:{:02d},99.0".format(t))
        t += 1
    rows.append("2026-07-30T10:05:00,99.0")          # a 4-minute pause
    for j in range(10):
        rows.append("2026-07-30T10:05:{:02d},99.0".format(1 + j))
    s = P.parse_wide_csv("\n".join(rows) + "\n")
    iv = I.measure(s)
    assert iv.measured_s == pytest.approx(1.0, abs=0.05), (
        "the long pause must be excluded; including it would make the inflated mean "
        "look like a property of the sampler")


def test_median_and_range_are_reported_alongside_the_mean():
    s = load(MATRIX)
    iv = I.measure(s)
    assert iv.min_s < iv.median_s < iv.max_s
    assert iv.spread_s > 0


def test_the_query_fixture_interval_is_about_a_quarter_second():
    s = load(QUERY, fmt="query_csv",
             columns="timestamp,index,utilization.gpu,power.draw")
    iv = I.measure(s)
    assert iv.measured_s == pytest.approx(0.258, abs=0.01)


# ------------------------------------------------------------------- repeat counts
def test_repeats_over_the_whole_file_are_dominated_by_the_idle_stretches():
    """Idle cards report exactly 0.0, so an unscoped repeat count measures idling."""
    s = load(MATRIX)
    everything = I.repeats(s, "sm_pct", 1.376)
    ctx = Context(s)
    active = [x for x in ctx.segments if x.kind == S.ACTIVE]
    scoped = I.repeats(s, "sm_pct", 1.376, segments=active)
    assert everything.repeat_fraction > scoped.repeat_fraction + 0.2, (
        "the unscoped count should be far higher; that gap is the false positive the "
        "scoped version exists to avoid")


def test_the_scoped_repeat_count_is_low_on_a_file_sampled_slower_than_the_window():
    s = load(MATRIX)
    ctx = Context(s)
    active = [x for x in ctx.segments if x.kind == S.ACTIVE]
    scoped = I.repeats(s, "sm_pct", 1.376, segments=active)
    assert scoped.repeat_fraction < I.REPEAT_WARN_FRACTION, (
        "1.376 s is slower than every window NVML documents, so repeats here would be "
        "a genuinely steady signal, not oversampling")


def test_oversampled_compares_against_the_slow_end_of_the_documented_range():
    assert I.oversampled(0.25) is True
    assert I.oversampled(1.376) is False
    assert I.oversampled(None) is None


def test_independent_rows_is_none_without_an_interval():
    s = load(MATRIX)
    rep = I.repeats(s, "sm_pct", None)
    assert rep.independent_rows is None
