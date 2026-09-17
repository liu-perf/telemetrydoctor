"""Assert that the first version is still wrong, and by how much.

`PASSED` here does not mean the result is healthy. Every test in this file asserts
that `aggregate_naive` -- read the whole file, average everything, take every
maximum, multiply the rates by elapsed time -- still produces the numbers that put
a fabricated figure into a document.

Same precedent as `regressiondoctor`'s `diff_naive` and `fitdoctor`'s
`budget_naive`: the broken version stays in the source, CI re-runs it on every push,
and deleting it would delete the evidence that the contract table was hit rather
than imagined.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from telemetrydoctor import Context, load  # noqa: E402
from telemetrydoctor import aggregate as agg  # noqa: E402
from telemetrydoctor.aggregate_naive import (  # noqa: E402
    naive_peak_pcie_sum,
    naive_summary,
)

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
MATRIX = os.path.join(FIXTURES, "EXAMPLE_matrix_8gpu.csv")


def _matrix():
    series = load(MATRIX)
    ctx = Context(series)
    eight = [s for s in ctx.segments if len(s.busy_gpus) == 8]
    one = [s for s in ctx.segments if len(s.busy_gpus) == 1]
    assert eight and one, "fixture must contain a 1-card and an 8-card configuration"
    return series, ctx, eight[0], one[0]


def test_naive_peak_sum_overstates_the_real_traffic_by_an_order_of_magnitude(capsys):
    """The '8 cards, 47 GB/s' shape: per-card maxima, added up."""
    series, _, eight, _ = _matrix()
    naive_total, _parts = naive_peak_pcie_sum(series)
    correct = agg.per_sample_sum(series, eight, "pcie_tx_gbs")
    assert not correct.refused

    ratio = naive_total / correct.value
    with capsys.disabled():
        print("\n  naive peak-sum across cards: {:6.2f} GB/s"
              "\n  steady-phase per-sample sum: {:6.2f} GB/s"
              "\n  overstatement:               {:6.1f}x".format(
                  naive_total, correct.value, ratio))

    assert naive_total > 50.0, "the naive figure should be implausibly large"
    assert ratio > 10.0, (
        "the naive peak-sum should overstate the real traffic by at least an order of "
        "magnitude; got {:.1f}x".format(ratio))


def test_naive_sums_a_percentage_across_cards_and_exceeds_100(capsys):
    """A percentage added across 8 cards. Runs fine; returns 239%."""
    series, _, _, _ = _matrix()
    naive = naive_summary(series, nominal_interval_s=1.0)
    summed = naive["sm_pct"]["sum_across_gpus"]
    with capsys.disabled():
        print("\n  naive sm_pct summed across 8 cards: {:.2f}%".format(summed))
    assert summed > 100.0, (
        "the point of this test is that the operation yields a percentage above 100; "
        "got {:.2f}".format(summed))

    # and the contract-checked path refuses it rather than returning that number
    refused = agg.per_sample_sum(series, None, "sm_pct")
    assert refused.refused
    assert refused.status == "forbidden"


def test_naive_whole_file_mean_hides_the_steady_state(capsys):
    """36% of this file is not steady work, and the whole-file mean shows it."""
    series, _, eight, _ = _matrix()
    naive = naive_summary(series)["sm_pct"]["mean"]
    steady = agg.mean(series, eight, "sm_pct", scope=agg.BUSY)
    with capsys.disabled():
        print("\n  whole-file sm mean:        {:5.1f}%"
              "\n  8-card steady busy mean:   {:5.1f}%".format(naive, steady.value))
    assert steady.value > 95.0
    assert naive < 45.0, (
        "the whole-file mean should land far below the steady state, which is how a "
        "reader concludes the cards were a third busy")


def test_naive_integrates_a_windowed_rate_and_the_contract_refuses_it():
    """Both compute something; only one of them says what it computed."""
    series, ctx, eight, _ = _matrix()
    naive_total = naive_summary(series, nominal_interval_s=1.0)["pcie_tx_gbs"]
    assert naive_total["total_over_run"] > 0, "the naive version happily returns a total"

    refused = agg.integrate(series, eight, "pcie_tx_gbs", ctx.interval.effective_s)
    assert refused.refused
    assert refused.status == "forbidden"
    assert "69x" in refused.note or "x" in refused.note, (
        "the refusal has to carry the size of the error, not just the prohibition")


def test_naive_uses_the_nominal_interval_and_that_alone_is_a_38_percent_error():
    """Same file, same arithmetic, two intervals: nominal 1.0 and measured 1.376."""
    series, ctx, _, _ = _matrix()
    measured = ctx.interval.measured_s
    assert measured is not None
    nominal = naive_summary(series, nominal_interval_s=1.0)["power_w"]["total_over_run"]
    real = naive_summary(series, nominal_interval_s=measured)["power_w"]["total_over_run"]
    ratio = real / nominal
    assert 1.30 < ratio < 1.45, (
        "the measured interval is ~37.6% longer than nominal, so any per-run total "
        "built on the nominal value is out by that much; got {:.3f}x".format(ratio))


def test_the_contract_checked_path_and_the_naive_path_agree_on_power():
    """The control: power is integrable, so here the two should NOT disagree.

    Without this, every test above is consistent with the contract table simply
    refusing everything.
    """
    series, ctx, eight, _ = _matrix()
    energy = agg.energy_j(series, eight, ctx.interval.effective_s)
    assert not energy.refused
    assert energy.status == "ok"

    per_sample = agg.per_sample_sum(series, eight, "power_w")
    expected = per_sample.value * eight.duration_s()
    assert abs(energy.value - expected) < 1.0, (
        "energy must be the summed board power times elapsed, not a per-card mean "
        "times elapsed -- the first version made exactly that mistake and reported "
        "one eighth of the real figure")
