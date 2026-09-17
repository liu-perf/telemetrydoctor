"""Two segmentations, and the refusal when they disagree."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from telemetrydoctor import crosscheck as CC  # noqa: E402
from telemetrydoctor import load  # noqa: E402
from telemetrydoctor import parse as P  # noqa: E402
from telemetrydoctor import segment as S  # noqa: E402

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
MATRIX = os.path.join(FIXTURES, "EXAMPLE_matrix_8gpu.csv")
UNIMODAL = os.path.join(FIXTURES, "EXAMPLE_unimodal.csv")

N = 8


def _wide(rows_sm):
    """Build an 8-card wide CSV from a list of per-sample sm lists."""
    header = ["timestamp"]
    for g in range(N):
        header.append("gpu{}_sm_pct".format(g))
    lines = [",".join(header)]
    for i, sms in enumerate(rows_sm):
        cells = ["2026-07-30T10:{:02d}:{:02d}".format(i // 60, i % 60)]
        cells.extend("{:.1f}".format(v) for v in sms)
        lines.append(",".join(cells))
    return P.parse_wide_csv("\n".join(lines) + "\n")


def _four_full_one_half():
    """Four cards at the plateau, one at half of it, three idle.

    Four-and-a-half rather than two-and-a-half so that the distribution is
    unambiguously two-class: with only two full cards the 0/50/99 histogram scores
    0.906 separability against a 0.90 floor, and a test balanced on the third
    decimal of a threshold is testing the threshold, not the thing it guards.
    """
    rows = [[0.0] * N for _ in range(12)]
    for _ in range(24):
        sms = [0.0] * N
        for g in range(4):
            sms[g] = 99.0
        sms[4] = 50.0
        rows.append(sms)
    rows.extend([0.0] * N for _ in range(12))
    return rows


def test_the_two_methods_agree_on_the_real_shaped_fixture():
    s = load(MATRIX)
    segs, th = S.segments(s)
    cc = CC.compare(s, segs, th)
    assert cc.basis == "compared"
    assert cc.agree is True
    assert cc.plateau > 95.0
    for label, (ok, total) in cc.by_label.items():
        assert ok / total >= CC.MIN_AGREEING_FRACTION, label


def test_a_half_busy_card_makes_the_two_methods_disagree():
    """Method A counts a card as busy or not; method B only sees the pooled level.

    Park one card at half the plateau. The set says five cards are working; the
    pooled level says four and a half. Neither is wrong -- what broke is the shared
    assumption that a card is either working or idle, and that is worth a refusal
    rather than whichever answer got computed first.
    """
    s = _wide(_four_full_one_half())
    segs, th = S.segments(s)
    active = [x for x in segs if x.kind == S.ACTIVE]
    assert active, "the file must still segment; the disagreement is the subject here"
    assert len(active[0].busy_gpus) == 5

    cc = CC.compare(s, segs, th)
    assert cc.basis == "compared"
    assert cc.agree is False
    label, frac = cc.worst
    assert frac < CC.MIN_AGREEING_FRACTION


def test_using_the_busy_mean_would_make_the_check_an_identity():
    """Why the plateau is a median. Without this the whole cross-check is vacuous.

    With plateau = mean(busy) = sum/k and pooled = sum/N, the estimate
    pooled/plateau*N collapses to k for every input -- the second method returns the
    first method's answer by arithmetic, not by agreement. This test recomputes the
    mean-based estimate on the very file that the median-based check rejects, and
    shows it landing exactly on the busy-set size.
    """
    s = _wide(_four_full_one_half())
    segs, th = S.segments(s)
    active = [x for x in segs if x.kind == S.ACTIVE][0]

    by_mean = dict(CC.pooled_counts(s, "sm_pct", th.high_mean, N))
    by_median = dict(CC.pooled_counts(s, "sm_pct", th.high_median, N))
    inside = [i for i in range(active.i0, active.i1 + 1) if i in by_mean]
    assert inside

    for i in inside:
        assert by_mean[i] == pytest.approx(5.0, abs=1e-6), (
            "the mean-based estimate reproduces the busy-set size exactly -- that is "
            "the identity, and it is why it can never disagree")
        assert by_median[i] == pytest.approx(4.5, abs=0.05), (
            "the median-based estimate sees four full cards and a half one")


def test_crosscheck_is_unknown_when_there_is_no_usable_threshold():
    """No confident split means nothing to cross-check -- not a quiet pass."""
    s = load(UNIMODAL)
    segs, th = S.segments(s)
    cc = CC.compare(s, segs, th)
    assert cc.agree is None
    assert cc.basis == "unknown"


def test_pooled_counts_never_look_at_an_individual_card():
    """If method B read per-card values it would not be independent of method A."""
    rows = [[0.0] * N for _ in range(6)]
    for _ in range(6):
        sms = [0.0] * N
        sms[0] = sms[1] = 100.0
        rows.append(sms)
    s = _wide(rows)
    counts = dict(CC.pooled_counts(s, "sm_pct", 100.0, N))
    busy_indices = [i for i, v in counts.items() if v and v > 1.5]
    assert busy_indices, "the pooled estimate should register two busy cards"
    for i in busy_indices:
        assert counts[i] == pytest.approx(2.0, abs=0.05)


def test_the_count_tolerance_is_generous_enough_for_clock_wobble():
    """One card out of eight is 12.5 points of pooled mean; the tolerance is 0.35 cards."""
    assert CC.COUNT_TOLERANCE < 0.5, (
        "above half a card the tolerance would let a genuine off-by-one through")
    assert CC.COUNT_TOLERANCE > 0.1, (
        "below a tenth of a card the check would fire on sampling jitter alone")
