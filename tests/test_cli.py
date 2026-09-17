"""The command line: line format, JSON, exit codes, and the CI gate."""
import json
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from telemetrydoctor.cli import main  # noqa: E402

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
MATRIX = os.path.join(FIXTURES, "EXAMPLE_matrix_8gpu.csv")
UNIMODAL = os.path.join(FIXTURES, "EXAMPLE_unimodal.csv")
QUERY = os.path.join(FIXTURES, "EXAMPLE_query_gpu.csv")
SINGLE = os.path.join(FIXTURES, "EXAMPLE_single_gpu.csv")

# The one output shape shared by all six tools in the series.
LINE = re.compile(r"^[^\s:]+: \[(ok|info|warn|violation|unknown|error)\] \S.*$")


def run(capsys, *argv):
    code = main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


@pytest.mark.parametrize("cmd", ["audit", "phases", "aggregate"])
def test_every_line_matches_the_house_format(capsys, cmd):
    code, out, _ = run(capsys, cmd, MATRIX)
    assert code == 0
    lines = [ln for ln in out.splitlines() if ln.strip()]
    assert lines
    for ln in lines:
        assert LINE.match(ln), ln


def test_contracts_needs_no_file(capsys):
    code, out, _ = run(capsys, "contracts")
    assert code == 0
    assert "power_w" in out and "sm_pct" in out
    for ln in out.splitlines():
        assert LINE.match(ln), ln


def test_audit_emits_one_line_per_rule_plus_a_source_line(capsys):
    _code, out, _ = run(capsys, "audit", MATRIX)
    lines = [ln for ln in out.splitlines() if ln.strip()]
    assert len(lines) == 11


def test_json_carries_the_same_content(capsys):
    _code, out, _ = run(capsys, "audit", MATRIX, "--json")
    data = json.loads(out)
    assert len(data) == 11
    for row in data:
        assert set(row) == {"rule", "location", "status", "message"}


def test_a_single_rule_can_be_selected(capsys):
    _code, out, _ = run(capsys, "audit", MATRIX, "--rule", "TL003")
    lines = [ln for ln in out.splitlines() if ln.strip()]
    assert len(lines) == 2
    assert lines[1].startswith("integrate: [violation]")


# ------------------------------------------------------------------- exit codes
def test_no_gate_means_exit_zero_even_with_a_violation(capsys):
    code, out, _ = run(capsys, "audit", MATRIX)
    assert "[violation]" in out
    assert code == 0, "reporting a problem and failing a build are separate decisions"


def test_fail_on_violation_trips(capsys):
    code, _out, _ = run(capsys, "audit", MATRIX, "--fail-on", "violation")
    assert code == 1


def test_fail_on_is_a_threshold_not_an_equality(capsys):
    """`--fail-on warn` must also trip on the worse status."""
    code, _out, _ = run(capsys, "audit", MATRIX, "--fail-on", "warn")
    assert code == 1


def test_fail_on_unknown_catches_a_file_that_cannot_be_segmented(capsys):
    code, out, _ = run(capsys, "audit", UNIMODAL, "--fail-on", "unknown")
    assert "[unknown]" in out
    assert code == 1, (
        "a file that cannot be phase-segmented has to be able to fail a build; "
        "'we could not tell' is not 'it is fine'")


def test_the_gate_can_pass(capsys, tmp_path):
    """The gate must be capable of passing, or it proves nothing when it fails.

    Note what a file has to look like to get through `--fail-on warn`: no windowed
    rate columns at all. `EXAMPLE_single_gpu.csv` does not qualify, because it
    carries PCIe columns and TL003 is a violation on those at any card count -- the
    integration error does not care how many GPUs there are.
    """
    rows = ["timestamp,gpu0_sm_pct,gpu0_power_w"]
    for i in range(40):
        rows.append("2026-07-30T10:{:02d}:{:02d},9{}.0,300.{}".format(
            i // 60, i % 60, 7 + i % 3, i % 10))
    p = tmp_path / "clean.csv"
    p.write_text("\n".join(rows) + "\n", encoding="utf-8")

    code, out, _ = run(capsys, "audit", str(p), "--fail-on", "warn")
    assert "[violation]" not in out
    assert "[warn]" not in out
    assert code == 0, out


def test_the_single_card_fixture_still_trips_on_its_pcie_columns(capsys):
    """The other half of the pair above: one card is not the same as no rate columns."""
    code, out, _ = run(capsys, "audit", SINGLE, "--fail-on", "warn")
    assert "integrate: [violation]" in out
    assert code == 1


def test_a_missing_file_exits_two_not_one(capsys):
    code, _out, err = run(capsys, "audit", os.path.join(FIXTURES, "nope.csv"))
    assert code == 2
    assert "[error]" in err


def test_query_format_without_columns_exits_two(capsys):
    code, _out, err = run(capsys, "audit", QUERY, "--format", "query_csv")
    assert code == 2
    assert "columns" in err.lower()


def test_no_subcommand_prints_help_and_exits_two(capsys):
    code, _out, _err = run(capsys)
    assert code == 2


# ---------------------------------------------------------------------- content
def test_aggregate_refuses_a_file_it_cannot_segment(capsys):
    code, out, _ = run(capsys, "aggregate", UNIMODAL)
    assert "refusing to aggregate" in out
    assert code == 0


def test_aggregate_prints_both_percentage_scopes_adjacently(capsys):
    _code, out, _ = run(capsys, "aggregate", MATRIX)
    lines = out.splitlines()
    busy = [i for i, ln in enumerate(lines) if ln.startswith("{0}.sm_pct.mean.busy_mean")]
    assert busy
    assert lines[busy[0] + 1].startswith("{0}.sm_pct.mean.all_device_mean")


def test_phases_lists_load_idle_and_active(capsys):
    _code, out, _ = run(capsys, "phases", MATRIX)
    assert "load[" in out
    assert "idle[" in out
    assert "active[" in out


def test_an_explicit_threshold_is_reported_as_a_warning(capsys):
    _code, out, _ = run(capsys, "phases", MATRIX, "--threshold", "90")
    assert "threshold: [warn]" in out


def test_the_source_line_flags_a_positionally_guessed_layout(capsys, tmp_path):
    rows = ["2026-07-30T10:00:{:02d},{}".format(
        i, ",".join("{:.3f}".format(i + j / 10.0) for j in range(5)))
        for i in range(8)]
    p = tmp_path / "headerless.csv"
    p.write_text("\n".join(rows) + "\n", encoding="utf-8")
    _code, out, _ = run(capsys, "audit", str(p))
    assert "by POSITION" in out
