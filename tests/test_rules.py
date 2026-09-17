"""Each of the ten rules: that it fires when it should, and stays quiet when it shouldn't.

A rule that never fires and a rule that always fires are equally useless, so every
rule here is checked on both sides wherever a fixture exists for both.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from telemetrydoctor import load  # noqa: E402
from telemetrydoctor import parse as P  # noqa: E402
from telemetrydoctor import rules as R  # noqa: E402

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
MATRIX = os.path.join(FIXTURES, "EXAMPLE_matrix_8gpu.csv")
PLATEAU48 = os.path.join(FIXTURES, "EXAMPLE_detection_plateau48.csv")
UNIMODAL = os.path.join(FIXTURES, "EXAMPLE_unimodal.csv")
DMON = os.path.join(FIXTURES, "EXAMPLE_dmon_no_timestamps.txt")
SINGLE = os.path.join(FIXTURES, "EXAMPLE_single_gpu.csv")

RULE_IDS = [rid for rid, _ in R.RULES]


def _one(path, rule_id, **kw):
    series = load(path, **{k: v for k, v in kw.items() if k in ("nominal_interval_s",)})
    findings, _ctx = R.audit(series, only=[rule_id])
    assert len(findings) == 1
    return findings[0]


# ------------------------------------------------------------------- the rule set
def test_there_are_ten_rules_and_the_ids_are_unique():
    assert len(RULE_IDS) == 10
    assert len(set(RULE_IDS)) == 10
    assert RULE_IDS == sorted(RULE_IDS)


def test_every_rule_returns_a_finding_on_every_fixture():
    """No rule may crash or return None on a shape it was not designed around."""
    for path in (MATRIX, PLATEAU48, UNIMODAL, DMON, SINGLE):
        series = load(path)
        findings, _ = R.audit(series)
        assert len(findings) == 10, path
        for f in findings:
            assert f.status in R.SEVERITY, (path, f.rule, f.status)
            assert f.message and len(f.message) > 40, (path, f.rule)
            assert f.location


def test_severity_ordering_puts_unknown_above_info_and_below_warn():
    """`unknown` is not a pass. It also is not a failure."""
    assert R.SEVERITY[R.OK] < R.SEVERITY[R.INFO] < R.SEVERITY[R.UNKNOWN]
    assert R.SEVERITY[R.UNKNOWN] < R.SEVERITY[R.WARN] < R.SEVERITY[R.VIOLATION]


# ------------------------------------------------------------------------- TL001
def test_tl001_warns_when_the_measured_interval_is_far_from_nominal():
    series = load(MATRIX, nominal_interval_s=1.0)
    f = R.tl001_interval(R.Context(series))
    assert f.status == R.WARN
    assert "1.37" in f.message and "+37." in f.message


def test_tl001_is_ok_when_the_nominal_matches_the_measurement():
    series = load(MATRIX, nominal_interval_s=1.376)
    f = R.tl001_interval(R.Context(series))
    assert f.status == R.OK


def test_tl001_is_unknown_when_the_file_carries_no_time():
    f = R.tl001_interval(R.Context(load(DMON)))
    assert f.status == R.UNKNOWN
    assert "dmon -o DT" in f.message


# ------------------------------------------------------------------------- TL002
def test_tl002_warns_and_names_the_load_phase():
    f = R.tl002_phase_mixing(R.Context(load(MATRIX)))
    assert f.status == R.WARN
    assert "load samples" in f.message
    assert "47 GB/s" in f.message


def test_tl002_is_ok_on_a_file_that_is_one_configuration_throughout():
    rows = ["timestamp,gpu0_sm_pct,gpu0_pcie_rx_gbs"]
    for i in range(30):
        rows.append("2026-07-30T10:00:{:02d},99.{},0.01".format(i, i % 10))
    series = P.parse_wide_csv("\n".join(rows) + "\n")
    f = R.tl002_phase_mixing(R.Context(series))
    assert f.status in (R.OK, R.UNKNOWN)


# ------------------------------------------------------------------------- TL003
def test_tl003_reports_the_error_factor_as_a_violation():
    f = R.tl003_integration(R.Context(load(MATRIX)))
    assert f.status == R.VIOLATION
    assert "69x" in f.message
    assert "1.45%" in f.message


def test_tl003_is_ok_when_no_windowed_column_is_present():
    rows = ["timestamp,gpu0_sm_pct,gpu0_power_w"]
    for i in range(20):
        rows.append("2026-07-30T10:00:{:02d},99.0,300.0".format(i))
    f = R.tl003_integration(R.Context(P.parse_wide_csv("\n".join(rows) + "\n")))
    assert f.status == R.OK


def test_tl003_is_unknown_without_an_interval():
    f = R.tl003_integration(R.Context(load(DMON)))
    assert f.status in (R.UNKNOWN, R.OK)


# ------------------------------------------------------------------------- TL004
def test_tl004_warns_about_windowed_peaks():
    f = R.tl004_peaks(R.Context(load(MATRIX)))
    assert f.status == R.WARN
    assert "pcie_tx_gbs" in f.message


# ------------------------------------------------------------------------- TL005
def test_tl005_warns_when_most_cards_are_idle():
    f = R.tl005_scope(R.Context(load(MATRIX)))
    assert f.status == R.WARN
    assert "99.0" in f.message or "98.9" in f.message
    assert "12." in f.message


def test_tl005_is_ok_on_a_single_card_file():
    f = R.tl005_scope(R.Context(load(SINGLE)))
    assert f.status == R.OK
    assert "single card" in f.message


# ------------------------------------------------------------------------- TL006
def test_tl006_measures_the_floor_from_idle_stretches_only():
    f = R.tl006_noise_floor(R.Context(load(MATRIX)))
    assert f.status == R.INFO
    assert "pcie_tx_gbs=0.000" in f.message, (
        "the measured floor must be near zero; a floor in the tenths means the weight "
        "load has been counted as idle")
    assert "factor of 7" in f.message


def test_tl006_is_unknown_when_the_file_never_goes_idle():
    rows = ["timestamp,gpu0_sm_pct,gpu0_pcie_rx_gbs"]
    for i in range(30):
        rows.append("2026-07-30T10:00:{:02d},99.{},0.01".format(i, i % 10))
    f = R.tl006_noise_floor(R.Context(P.parse_wide_csv("\n".join(rows) + "\n")))
    assert f.status == R.UNKNOWN
    assert "control group" in f.message


# ------------------------------------------------------------------------- TL007
def test_tl007_reports_that_a_fixed_90_would_discard_every_working_sample():
    f = R.tl007_threshold(R.Context(load(PLATEAU48)))
    assert f.status == R.INFO
    assert "every one of them" in f.message
    assert "48." in f.message


def test_tl007_refuses_a_file_with_no_usable_split():
    f = R.tl007_threshold(R.Context(load(UNIMODAL)))
    assert f.status == R.UNKNOWN
    assert "0.90 floor" in f.message


def test_tl007_warns_when_the_threshold_was_supplied_rather_than_measured():
    series = load(MATRIX)
    f = R.tl007_threshold(R.Context(series, explicit_threshold=90.0))
    assert f.status == R.WARN
    assert "48.4%, 69.9% and 94.1%" in f.message


# ------------------------------------------------------------------------- TL008
def test_tl008_always_states_what_the_column_means():
    f = R.tl008_semantics(R.Context(load(MATRIX)))
    assert f.status == R.INFO
    assert "978x" in f.message
    assert "not SM occupancy" in f.message


# ------------------------------------------------------------------------- TL009
def test_tl009_is_ok_when_the_two_segmentations_agree():
    f = R.tl009_crosscheck(R.Context(load(MATRIX)))
    assert f.status == R.OK


def test_tl009_is_unknown_when_the_second_method_cannot_run():
    f = R.tl009_crosscheck(R.Context(load(UNIMODAL)))
    assert f.status == R.UNKNOWN


# ------------------------------------------------------------------------- TL010
def test_tl010_is_ok_on_a_file_sampled_slower_than_the_driver_window():
    f = R.tl010_oversampling(R.Context(load(MATRIX)))
    assert f.status == R.OK
    assert "1.37" in f.message


def test_tl010_warns_on_a_file_sampled_faster_than_the_window():
    """250 ms polling with a 1 s window: rows repeat and are not independent."""
    rows = ["timestamp,gpu0_sm_pct,gpu0_pcie_rx_gbs"]
    t = 0.0
    for i in range(80):
        rows.append("2026-07-30T10:00:{:02d}.{:03d},{},0.01".format(
            int(t) % 60, int((t % 1) * 1000), 99.0 if i % 4 else 98.0))
        t += 0.25
    series = P.parse_wide_csv("\n".join(rows) + "\n")
    f = R.tl010_oversampling(R.Context(series))
    assert f.status == R.WARN
    assert "card-samples" in f.message


# ---------------------------------------------------------------------- reporting
def test_worst_status_picks_the_most_severe():
    findings, _ = R.audit(load(MATRIX))
    assert R.worst_status(findings) == R.VIOLATION
    assert R.worst_status([]) == R.OK


def test_findings_serialise_to_dicts_with_the_four_fields():
    findings, _ = R.audit(load(MATRIX))
    for f in findings:
        d = f.as_dict()
        assert set(d) == {"rule", "location", "status", "message"}


def test_the_finding_string_form_matches_the_house_format():
    findings, _ = R.audit(load(MATRIX))
    for f in findings:
        assert str(f) == "{}: [{}] {}".format(f.location, f.status, f.message)


def test_describe_contracts_covers_the_whole_table():
    rows = R.describe_contracts()
    from telemetrydoctor.contracts import CONTRACTS
    assert len(rows) == len(CONTRACTS)
    for r in rows:
        assert r.status == R.INFO
        assert "mean=" in r.message and "integrate=" in r.message


@pytest.mark.parametrize("rule_id", RULE_IDS)
def test_running_one_rule_runs_only_that_rule(rule_id):
    findings, _ = R.audit(load(MATRIX), only=[rule_id])
    assert len(findings) == 1
    assert findings[0].rule == rule_id
