"""Phase cutting: the adaptive threshold, and the refusal when there isn't one."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from telemetrydoctor import load  # noqa: E402
from telemetrydoctor import segment as S  # noqa: E402

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
MATRIX = os.path.join(FIXTURES, "EXAMPLE_matrix_8gpu.csv")
PLATEAU48 = os.path.join(FIXTURES, "EXAMPLE_detection_plateau48.csv")
PLATEAU94 = os.path.join(FIXTURES, "EXAMPLE_segmentation_plateau94.csv")
UNIMODAL = os.path.join(FIXTURES, "EXAMPLE_unimodal.csv")


# --------------------------------------------------------------------------- otsu
def test_otsu_splits_a_clean_two_class_distribution():
    values = [0.0] * 50 + [99.0] * 50
    threshold, sep = S.otsu(values)
    assert 0.0 < threshold < 99.0
    assert sep > 0.99


def test_otsu_reports_low_separability_on_one_mode():
    """A continuous band with no gap. Otsu still returns a cut; it must not be used."""
    values = [30.0 + 40.0 * (i / 199.0) for i in range(200)]
    _threshold, sep = S.otsu(values)
    assert sep == pytest.approx(0.75, abs=0.03), (
        "a uniform band scores about 0.75; this number is what sets MIN_SEPARABILITY, "
        "so pin it -- an earlier floor of 0.70 sat below it and let this through")
    assert sep < S.MIN_SEPARABILITY


def test_a_genuinely_bimodal_distribution_scores_far_above_the_floor():
    """The other side of the floor: the gap between 0.75 and 0.99 is what it exploits."""
    values = [0.0 + (i % 3) for i in range(60)] + [98.0 + (i % 3) for i in range(60)]
    _threshold, sep = S.otsu(values)
    assert sep > 0.99
    assert sep > S.MIN_SEPARABILITY


def test_otsu_declines_a_constant_column():
    assert S.otsu([42.0] * 30) == (None, None)


def test_otsu_declines_too_few_values():
    assert S.otsu([0.0, 99.0]) == (None, None)


# ---------------------------------------------------------------------- threshold
def test_threshold_is_computed_from_the_file_not_assumed():
    s = load(MATRIX)
    th = S.busy_threshold(s)
    assert th.basis == "otsu"
    assert th.usable
    assert th.high_mean > 95.0
    assert th.low_mean == pytest.approx(0.0, abs=2.0)


def test_a_forced_threshold_is_marked_explicit():
    s = load(MATRIX)
    th = S.busy_threshold(s, explicit=90.0)
    assert th.basis == "explicit"
    assert th.value == 90.0
    assert th.usable, "an explicit threshold is usable; it is just not measured"


def test_the_plateau_is_measured_and_differs_between_workloads():
    """48.4% and 94.1% are the two real plateaux this whole design exists for."""
    lo = S.busy_threshold(load(PLATEAU48))
    hi = S.busy_threshold(load(PLATEAU94))
    assert lo.high_mean == pytest.approx(48.4, abs=1.0)
    assert hi.high_mean == pytest.approx(94.1, abs=1.0)


def test_a_fixed_90_threshold_selects_nothing_on_the_48_percent_workload():
    """The counter-example that killed the constant."""
    s = load(PLATEAU48)
    segs_fixed, _ = S.segments(s, explicit_threshold=90.0)
    assert not [x for x in segs_fixed if x.kind == S.ACTIVE], (
        "a fixed 90 must find no work at all in a file whose plateau is 48.4%")

    segs_auto, th = S.segments(s)
    active = [x for x in segs_auto if x.kind == S.ACTIVE]
    assert len(active) == 3, "the same file has three configurations under an adaptive cut"
    assert th.basis == "otsu"


# ------------------------------------------------------------------- the refusal
def test_a_unimodal_file_is_refused_rather_than_cut_down_the_middle():
    s = load(UNIMODAL)
    segs, th = S.segments(s)
    assert segs == []
    assert not th.usable
    assert th.separability is not None and th.separability < S.MIN_SEPARABILITY


# -------------------------------------------------------------- load / idle split
def test_the_weight_load_is_its_own_phase_not_part_of_the_idle_around_it():
    """A load stretch sits below the busy threshold, so the busy set cannot see it."""
    s = load(MATRIX)
    segs, _ = S.segments(s)
    kinds = [x.kind for x in segs]
    assert S.LOAD in kinds
    loads = [x for x in segs if x.kind == S.LOAD]
    assert len(loads) == 1
    assert loads[0].n_samples == 12, (
        "the load is 12 samples; if this comes back as 36 the idle stretches on "
        "either side have been glued to it, which is what the first version did")
    assert loads[0].busy_gpus == ()


def test_the_idle_stretches_are_not_contaminated_by_the_load():
    s = load(MATRIX)
    segs, _ = S.segments(s)
    idle = [x for x in segs if x.kind == S.IDLE]
    assert idle
    rx = []
    for seg in idle:
        for r in s.readings:
            if seg.i0 <= r.index <= seg.i1 and "pcie_rx_gbs" in r.values:
                rx.append(r.values["pcie_rx_gbs"])
    assert max(rx) < 0.1, (
        "an idle stretch must not contain multi-GB/s receive traffic; if it does the "
        "measured noise floor is really a measurement of the weight load")


def test_configurations_are_found_and_labelled_by_busy_set():
    s = load(MATRIX)
    segs, _ = S.segments(s)
    labels = sorted(S.configurations(segs))
    assert labels == ["{0}", "{0,1}", "{0,1,2,3}", "{0,1,5,6}", "{0,1,2,3,4,5,6,7}"] or \
        set(labels) == {"{0}", "{0,1}", "{0,1,2,3}", "{0,1,5,6}", "{0,1,2,3,4,5,6,7}"}


def test_the_two_four_card_configurations_stay_distinct():
    """{0,1,2,3} and {0,1,5,6} are both four cards and must not merge."""
    s = load(MATRIX)
    segs, _ = S.segments(s)
    four = [x for x in segs if x.kind == S.ACTIVE and len(x.busy_gpus) == 4]
    assert len(four) == 2
    assert {x.label for x in four} == {"{0,1,2,3}", "{0,1,5,6}"}


def test_a_short_transition_is_not_promoted_to_a_phase():
    s = load(MATRIX)
    segs, _ = S.segments(s, min_samples=10)
    assert all(x.n_samples >= 10 for x in segs)


def test_segment_durations_come_from_the_timestamps():
    s = load(MATRIX)
    segs, _ = S.segments(s)
    for seg in segs:
        d = seg.duration_s()
        assert d is not None and d > 0
        # ~1.376 s per sample, so a 19-sample phase spans about 25 s
        assert d == pytest.approx((seg.n_samples - 1) * 1.376, rel=0.05)
