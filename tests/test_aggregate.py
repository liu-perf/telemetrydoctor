"""Legal aggregates, illegal aggregates, and scope."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from telemetrydoctor import Context, load  # noqa: E402
from telemetrydoctor import aggregate as A  # noqa: E402
from telemetrydoctor import segment as S  # noqa: E402

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
MATRIX = os.path.join(FIXTURES, "EXAMPLE_matrix_8gpu.csv")


@pytest.fixture(scope="module")
def matrix():
    series = load(MATRIX)
    ctx = Context(series)
    by_size = {}
    for seg in ctx.segments:
        if seg.kind == S.ACTIVE:
            by_size.setdefault(len(seg.busy_gpus), []).append(seg)
    return series, ctx, by_size


def test_the_two_scopes_of_a_percentage_are_both_reported_and_differ(matrix):
    """99.0% and 12.4% on the same samples. Printing only the second misled a reader."""
    series, _, by_size = matrix
    one = by_size[1][0]
    busy = A.mean(series, one, "sm_pct", scope=A.BUSY)
    everything = A.mean(series, one, "sm_pct", scope=A.ALL_DEVICE)
    assert busy.value == pytest.approx(99.0, abs=1.0)
    assert everything.value == pytest.approx(12.4, abs=1.0)
    assert busy.scope != everything.scope
    assert everything.value == pytest.approx(busy.value / 8.0, rel=0.02)


def test_on_the_full_configuration_the_two_scopes_coincide(matrix):
    series, _, by_size = matrix
    eight = by_size[8][0]
    busy = A.mean(series, eight, "sm_pct", scope=A.BUSY)
    everything = A.mean(series, eight, "sm_pct", scope=A.ALL_DEVICE)
    assert busy.value == pytest.approx(everything.value, abs=0.01)


def test_summing_a_percentage_across_cards_is_refused(matrix):
    series, _, by_size = matrix
    v = A.per_sample_sum(series, by_size[8][0], "sm_pct")
    assert v.refused
    assert v.status == "forbidden"
    assert "do not add" in v.note


def test_summing_a_rate_across_cards_is_allowed_and_sums_per_sample(matrix):
    """Sum within each sample, then average -- not average then multiply by 8."""
    series, _, by_size = matrix
    eight = by_size[8][0]
    total = A.per_sample_sum(series, eight, "pcie_tx_gbs")
    per_card = A.mean(series, eight, "pcie_tx_gbs", scope=A.BUSY)
    assert not total.refused
    assert total.value == pytest.approx(per_card.value * 8, rel=0.02)


def test_the_peak_of_a_windowed_rate_is_refused_with_a_reason(matrix):
    series, _, by_size = matrix
    v = A.peak(series, by_size[8][0], "pcie_tx_gbs")
    assert v.refused
    assert v.status == "meaningless"
    assert "20 ms" in v.note


def test_the_peak_of_an_instantaneous_reading_is_allowed(matrix):
    series, _, by_size = matrix
    v = A.peak(series, by_size[8][0], "power_w")
    assert not v.refused
    assert v.status == "ok"
    assert v.value > 500.0


def test_integrating_a_windowed_rate_is_refused_with_the_size_of_the_error(matrix):
    series, ctx, by_size = matrix
    v = A.integrate(series, by_size[8][0], "pcie_tx_gbs", ctx.interval.effective_s)
    assert v.refused
    assert v.status == "forbidden"
    assert "69x" in v.note
    assert "1.45%" in v.note


def test_integrating_power_is_allowed_and_gives_joules(matrix):
    series, ctx, by_size = matrix
    eight = by_size[8][0]
    energy = A.energy_j(series, eight, ctx.interval.effective_s)
    assert not energy.refused
    assert energy.unit == "J"
    total_w = A.per_sample_sum(series, eight, "power_w").value
    assert energy.value == pytest.approx(total_w * eight.duration_s(), rel=0.001)


def test_energy_covers_every_busy_card_not_one_of_them(matrix):
    """The first version integrated the per-card mean and was out by the card count."""
    series, ctx, by_size = matrix
    one = A.energy_j(series, by_size[1][0], ctx.interval.effective_s)
    eight = A.energy_j(series, by_size[8][0], ctx.interval.effective_s)
    per_second_1 = one.value / by_size[1][0].duration_s()
    per_second_8 = eight.value / by_size[8][0].duration_s()
    assert per_second_8 / per_second_1 == pytest.approx(8.0, rel=0.05), (
        "eight busy cards draw about eight times the power of one; if this comes back "
        "near 1.0 the sum across cards has been lost")


def test_a_whole_file_mean_of_a_within_phase_column_is_refused(matrix):
    """`segment=None` means the whole file, which for sm_pct is not a defined mean."""
    series, _, _ = matrix
    v = A.mean(series, None, "sm_pct", scope=A.ALL_DEVICE)
    assert v.refused
    assert v.status == "forbidden"
    assert "one phase" in v.note


def test_an_unknown_column_is_refused_rather_than_defaulted(matrix):
    series, _, by_size = matrix
    v = A.mean(series, by_size[8][0], "vibes")
    assert v.refused
    assert v.status == "unknown"
    assert "no contract" in v.note


def test_summarise_puts_the_two_percentage_scopes_next_to_each_other(matrix):
    """So a reader cannot quote one without seeing the other."""
    series, ctx, by_size = matrix
    values = A.summarise(series, by_size[1][0], ctx.interval.effective_s)
    sm = [v for v in values if v.column == "sm_pct" and v.op == "mean"]
    assert len(sm) == 2
    assert values.index(sm[0]) + 1 == values.index(sm[1])
    assert {v.scope for v in sm} == {A.BUSY, A.ALL_DEVICE}


def test_summarise_reports_refusals_as_well_as_values(matrix):
    series, ctx, by_size = matrix
    values = A.summarise(series, by_size[8][0], ctx.interval.effective_s)
    assert any(v.refused for v in values), "the refusals are the product"
    assert any(not v.refused for v in values), "and it still computes what it can"
