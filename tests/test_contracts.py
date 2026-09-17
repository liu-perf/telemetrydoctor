"""The contract table: that it is complete, consistent, and says something."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from telemetrydoctor import contracts as C  # noqa: E402

ALL = sorted(C.CONTRACTS)


def test_there_are_contracts_to_test():
    assert len(ALL) >= 6


@pytest.mark.parametrize("name", ALL)
def test_every_contract_answers_every_operation(name):
    c = C.CONTRACTS[name]
    for op in C.OPERATIONS:
        assert c.verdict(op) in (C.OK, C.WITHIN_PHASE, C.MEANINGLESS,
                                C.FORBIDDEN, C.UNKNOWN), (name, op)


@pytest.mark.parametrize("name", ALL)
def test_every_contract_explains_itself(name):
    """A verdict with no reason is an appeal to authority."""
    c = C.CONTRACTS[name]
    assert len(c.why) > 80, "{}: the reason is too short to be a reason".format(name)
    assert c.semantics.strip()
    assert c.column == name
    assert c.kind in ("instantaneous", "windowed_rate", "time_fraction", "counter")


@pytest.mark.parametrize("name", ALL)
def test_an_unknown_operation_raises_rather_than_defaulting(name):
    with pytest.raises(KeyError):
        C.CONTRACTS[name].verdict("smooth")


def test_the_table_is_not_only_prohibitions():
    """If nothing were permitted the tool would be useless and unfalsifiable."""
    integrable = [n for n in ALL if C.CONTRACTS[n].verdict("integrate") == C.OK]
    assert integrable == ["power_w"], (
        "power is the one column here that may be integrated; if that list grows or "
        "empties, the claim in the docs is stale")


def test_percentages_may_not_be_summed_across_cards():
    for name in ("sm_pct", "mem_pct"):
        assert C.CONTRACTS[name].verdict("sum_across_gpus") == C.FORBIDDEN


def test_rates_may_be_summed_across_cards():
    for name in ("pcie_tx_gbs", "pcie_rx_gbs", "power_w"):
        assert C.CONTRACTS[name].verdict("sum_across_gpus") == C.OK


def test_windowed_rate_peaks_are_meaningless_and_instantaneous_peaks_are_not():
    assert C.CONTRACTS["pcie_tx_gbs"].verdict("peak") == C.MEANINGLESS
    assert C.CONTRACTS["fb_used_mib"].verdict("peak") == C.OK
    assert C.CONTRACTS["power_w"].verdict("peak") == C.OK


def test_duty_cycle_is_the_window_over_the_interval():
    c = C.CONTRACTS["pcie_tx_gbs"]
    duty = c.duty_cycle(1.376)
    assert duty == pytest.approx(0.020 / 1.376, rel=1e-9)
    assert duty * 100 == pytest.approx(1.45, abs=0.01), (
        "1.45% is the figure quoted throughout the docs")


def test_duty_cycle_is_none_when_the_vendor_documents_a_range():
    """NVML gives utilisation's window as 1 s to 1/6 s. A midpoint would be a guess."""
    assert isinstance(C.CONTRACTS["sm_pct"].window_ms, tuple)
    assert C.CONTRACTS["sm_pct"].duty_cycle(1.0) is None


def test_integration_error_factor_is_about_69x_at_the_field_interval():
    c = C.CONTRACTS["pcie_tx_gbs"]
    factor = C.integration_error_factor(c, 1.376)
    assert factor == pytest.approx(68.8, abs=0.5)


def test_integration_error_factor_is_none_for_an_integrable_column():
    assert C.integration_error_factor(C.CONTRACTS["power_w"], 1.376) is None


def test_aliases_resolve_the_spellings_that_turn_up():
    for spelling, canon in (("utilization.gpu", "sm_pct"), ("sm", "sm_pct"),
                            ("pwr", "power_w"), ("power.draw", "power_w"),
                            ("tx_gbs", "pcie_tx_gbs"), ("rx", "pcie_rx_gbs"),
                            ("memory.used", "fb_used_mib"), ("gtemp", "temp_c")):
        assert C.canonical(spelling) == canon, spelling


def test_gpu_is_deliberately_not_an_alias():
    """In dmon output `gpu` is the card index; elsewhere it means utilisation.

    An alias resolving one spelling to two meanings is worse than none, and this
    exact collision made `from . import audit` return a function once already.
    """
    assert C.canonical("gpu") is None


def test_an_unrecognised_column_gets_no_contract_rather_than_a_permissive_default():
    assert C.contract_for("vibes") is None
    assert C.canonical("vibes") is None
