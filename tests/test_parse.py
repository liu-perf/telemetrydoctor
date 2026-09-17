"""Reading the three formats, and refusing to guess quietly."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from telemetrydoctor import load  # noqa: E402
from telemetrydoctor import parse as P  # noqa: E402

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")

WIDE = os.path.join(FIXTURES, "EXAMPLE_matrix_8gpu.csv")
DMON = os.path.join(FIXTURES, "EXAMPLE_dmon_no_timestamps.txt")
QUERY = os.path.join(FIXTURES, "EXAMPLE_query_gpu.csv")
SINGLE = os.path.join(FIXTURES, "EXAMPLE_single_gpu.csv")
QUERY_COLUMNS = "timestamp,index,utilization.gpu,power.draw"


def test_wide_csv_reads_eight_cards_and_five_columns_each():
    s = load(WIDE)
    assert s.source_format == "wide_csv"
    assert s.gpus == list(range(8))
    assert s.layout_basis == "header"
    for col in ("pcie_tx_gbs", "pcie_rx_gbs", "sm_pct", "mem_pct", "power_w"):
        assert col in s.columns
    assert s.n_samples == 156
    assert len(s.readings) == 156 * 8


def test_dmon_reads_and_has_no_timestamps():
    s = load(DMON)
    assert s.source_format == "dmon"
    assert s.gpus == [0, 1]
    assert not s.has_timestamps, (
        "dmon without -o DT carries no time at all; inventing one would be the exact "
        "mistake TL001 is about")
    assert s.span_s() is None
    assert "sm_pct" in s.columns and "power_w" in s.columns


def test_dmon_leading_index_column_is_not_read_as_a_metric():
    """`# gpu` heads the card index. Reading it as utilisation would put 0/1 in sm."""
    s = load(DMON)
    for r in s.readings:
        if "sm_pct" in r.values:
            assert r.values["sm_pct"] in (0.0,) or r.values["sm_pct"] > 50.0, (
                "sm should be either the idle 0 or the busy plateau, never a card index")


def test_query_csv_needs_a_column_spec():
    with pytest.raises(ValueError) as exc:
        load(QUERY, fmt="query_csv")
    assert "columns" in str(exc.value).lower()


def test_query_csv_refuses_a_spec_of_the_wrong_width():
    """A short spec would silently relabel power as utilisation."""
    with pytest.raises(ValueError) as exc:
        load(QUERY, fmt="query_csv", columns="timestamp,index")
    assert "mismatched" in str(exc.value) or "fields per row" in str(exc.value)


def test_query_csv_groups_rows_into_samples_by_card_index():
    s = load(QUERY, fmt="query_csv", columns=QUERY_COLUMNS)
    assert s.gpus == [0, 1]
    assert s.n_samples == 60
    assert len(s.readings) == 120


def test_positional_fallback_is_used_and_is_labelled_as_a_guess():
    """No usable header: fall back to the documented layout, and say so."""
    rows = ["2026-07-30T10:00:0{},{}".format(
        i, ",".join("{:.3f}".format(i + j / 10.0) for j in range(10)))
        for i in range(5)]
    s = P.parse_wide_csv("\n".join(rows) + "\n")
    assert s.layout_basis == "assumed_positional"
    assert s.gpus == [0, 1]
    assert set(s.columns) == set(P.FIELD_POSITIONAL)


def test_positional_fallback_refuses_a_width_it_cannot_explain():
    rows = ["2026-07-30T10:00:0{},1.0,2.0,3.0".format(i) for i in range(4)]
    with pytest.raises(ValueError) as exc:
        P.parse_wide_csv("\n".join(rows) + "\n")
    assert "positional fallback" in str(exc.value)


def test_comment_lines_are_kept_so_a_fixture_can_declare_itself():
    s = load(WIDE)
    assert s.comments, "the '# EXAMPLE' lines have to survive parsing"
    assert s.declares_synthetic


def test_a_real_capture_would_not_declare_itself_synthetic():
    rows = ["timestamp,gpu0_sm_pct", "2026-07-30T10:00:00,99.0",
            "2026-07-30T10:00:01,99.1", "2026-07-30T10:00:02,99.2"]
    s = P.parse_wide_csv("\n".join(rows) + "\n")
    assert not s.declares_synthetic


def test_sniff_does_not_mistake_a_declared_csv_for_dmon():
    """An earlier version keyed on the presence of a '#' line and got this wrong."""
    text = ("# EXAMPLE -- synthetic\ntimestamp,gpu0_sm_pct\n"
            "2026-07-30T10:00:00,99.0\n2026-07-30T10:00:01,99.0\n")
    assert P.sniff(text) == "wide_csv"


def test_sniff_recognises_dmon_by_the_absence_of_commas():
    assert P.sniff(open(DMON, encoding="utf-8").read()) == "dmon"


def test_sniff_refuses_a_file_that_is_only_comments():
    with pytest.raises(ValueError):
        P.sniff("# nothing but a note\n# and another\n")


@pytest.mark.parametrize("text,expected_frac", [
    ("2026-07-30T10:54:13.877", 0.877),
    ("2026/08/13 15:03:56.707", 0.707),
    ("2026-07-31T12:40:41.150276", 0.150276),
    ("2026-07-31T12:40:41+00:00", 0.0),
])
def test_timestamp_formats_that_turn_up_all_parse(text, expected_frac):
    t = P.parse_timestamp(text)
    assert t is not None, text
    assert abs((t - int(t)) - expected_frac) < 1e-4 or expected_frac == 0.0


def test_an_unparseable_timestamp_is_none_rather_than_zero():
    assert P.parse_timestamp("last tuesday") is None


def test_missing_and_not_supported_values_are_skipped_not_zeroed():
    rows = ["timestamp,gpu0_sm_pct,gpu0_power_w",
            "2026-07-30T10:00:00,99.0,[N/A]",
            "2026-07-30T10:00:01,99.0,-",
            "2026-07-30T10:00:02,99.0,300.0"]
    s = P.parse_wide_csv("\n".join(rows) + "\n")
    power = s.column("power_w")
    assert power == [300.0], (
        "an unsupported reading is absent, not zero; averaging zeros in would drag "
        "the mean down by however many samples the driver declined to answer")


def test_single_card_file_reads_as_one_card():
    s = load(SINGLE)
    assert s.gpus == [0]
    assert s.n_samples == 70
