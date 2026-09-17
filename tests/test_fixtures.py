"""The fixtures themselves: that they declare what they are, and are what they claim.

Unusual for a test suite, and the same reason `fitdoctor` tests its shipped profiles:
these files are an argument. If a fixture drifts from the figure it was built to
reproduce, every test written against it keeps passing while quietly asserting
something else.
"""
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from telemetrydoctor import Context, load  # noqa: E402
from telemetrydoctor import aggregate as A  # noqa: E402
from telemetrydoctor import segment as S  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "fixtures")
MATRIX = os.path.join(FIXTURES, "EXAMPLE_matrix_8gpu.csv")
GENERATOR = os.path.join(FIXTURES, "make_fixtures.py")

ALL_FIXTURES = sorted(f for f in os.listdir(FIXTURES)
                      if f.endswith((".csv", ".txt")))

# From make_fixtures.CONFIGS: the real capture's per-configuration PCIe sums.
TARGETS = {
    "{0}": (0.008, 0.040, 545.0),
    "{0,1}": (0.504, 0.374, 1050.0),
    "{0,1,2,3}": (2.172, 1.733, 2194.0),
    "{0,1,5,6}": (1.941, 1.632, 2236.0),
    "{0,1,2,3,4,5,6,7}": (4.118, 4.288, 4386.0),
}
# One burst is 3 GB/s. Over a 19-23 sample phase, +-1 burst moves the summed mean
# by roughly BURST / n_samples * n_cards. This is the tolerance that buys.
BURST_GBS = 3.0


def test_there_are_fixtures_to_test():
    assert len(ALL_FIXTURES) >= 6


@pytest.mark.parametrize("name", ALL_FIXTURES)
def test_every_fixture_declares_itself_synthetic_in_band(name):
    """Not in the filename -- in the data, where it survives being pasted elsewhere."""
    with open(os.path.join(FIXTURES, name), encoding="utf-8") as fh:
        head = "".join(fh.readline() for _ in range(4))
    assert "EXAMPLE" in head.upper()
    assert "not a real machine" in head


@pytest.mark.parametrize("name", ALL_FIXTURES)
def test_every_fixture_parses(name):
    path = os.path.join(FIXTURES, name)
    if "query_gpu" in name:
        series = load(path, fmt="query_csv",
                      columns="timestamp,index,utilization.gpu,power.draw")
    else:
        series = load(path)
    assert series.n_samples > 10
    assert series.gpus


def test_the_generator_is_deterministic():
    """Re-running it must not change a byte, or the committed files are unverifiable."""
    before = {}
    for name in ALL_FIXTURES:
        with open(os.path.join(FIXTURES, name), "rb") as fh:
            before[name] = fh.read()
    subprocess.run([sys.executable, GENERATOR], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for name in ALL_FIXTURES:
        with open(os.path.join(FIXTURES, name), "rb") as fh:
            assert fh.read() == before[name], name


@pytest.mark.parametrize("label", sorted(TARGETS))
def test_each_configuration_reproduces_its_target_within_burst_quantisation(label):
    series = load(MATRIX)
    ctx = Context(series)
    segs = [s for s in ctx.segments if s.kind == S.ACTIVE and s.label == label]
    assert len(segs) == 1, label
    seg = segs[0]

    tx_target, rx_target, power_target = TARGETS[label]
    n_cards = len(seg.busy_gpus)
    # +-1 burst on any one card moves the summed mean by BURST / n_samples
    tolerance = BURST_GBS / seg.n_samples * max(1, n_cards) * 1.1

    tx = A.per_sample_sum(series, seg, "pcie_tx_gbs")
    rx = A.per_sample_sum(series, seg, "pcie_rx_gbs")
    power = A.per_sample_sum(series, seg, "power_w")

    assert abs(tx.value - tx_target) <= tolerance, (label, tx.value, tx_target)
    assert abs(rx.value - rx_target) <= tolerance, (label, rx.value, rx_target)
    # power is not bursty, so it should be tight
    assert power.value == pytest.approx(power_target, rel=0.01), label


def test_the_burst_tolerance_is_actually_needed():
    """If every configuration matched to three decimals, the fixture would be smooth.

    A smooth fixture would hide the thing TL004 is about, so the residual is load
    bearing: this asserts at least one configuration is off by more than a percent.
    """
    series = load(MATRIX)
    ctx = Context(series)
    worst = 0.0
    for seg in ctx.segments:
        if seg.kind != S.ACTIVE or seg.label not in TARGETS:
            continue
        tx_target = TARGETS[seg.label][0]
        if tx_target < 0.05:
            continue                     # the 1-card case sits on the noise floor
        got = A.per_sample_sum(series, seg, "pcie_tx_gbs").value
        worst = max(worst, abs(got - tx_target) / tx_target)
    assert worst > 0.01, (
        "no configuration deviates by even 1%; either the generator became smooth or "
        "the bursts stopped, and in both cases the peak rule has nothing to bite on")


def test_the_busy_plateau_matches_the_real_figure_it_was_built_from():
    assert S.busy_threshold(load(MATRIX)).high_mean == pytest.approx(98.5, abs=0.7)


def test_the_measured_interval_matches_the_real_capture():
    ctx = Context(load(MATRIX))
    assert ctx.interval.measured_s == pytest.approx(1.376, abs=0.005)


def test_the_two_plateau_fixtures_bracket_the_field_observation():
    """48.4 and 94.1 on the same cards: the pair that killed the fixed threshold."""
    lo = S.busy_threshold(load(os.path.join(
        FIXTURES, "EXAMPLE_detection_plateau48.csv"))).high_mean
    hi = S.busy_threshold(load(os.path.join(
        FIXTURES, "EXAMPLE_segmentation_plateau94.csv"))).high_mean
    assert lo == pytest.approx(48.4, abs=1.0)
    assert hi == pytest.approx(94.1, abs=1.0)
    assert hi - lo > 40.0
