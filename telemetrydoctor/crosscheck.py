"""Cut the file twice, two different ways, and refuse to report if they disagree.

The field procedure ends with a step called "换方法独立复核（交报告前必做）" -- recut
the data by a second method and only believe the numbers if both agree. That step is
here, automated, because it is the one step in a manual analysis that always gets
skipped when the first answer looks reasonable.

The two methods have to be independent in what they read, not just different in how
they are written:

  **A. per-card identity.** For each sample, which cards are above the file's own
  busy threshold. The label is the *set*: `{0,1,5,6}`. This is `segment.py`.

  **B. pooled level only.** Take the mean across all cards, one number per sample,
  and divide by the busy plateau to estimate *how many* cards were working. This
  never looks at an individual card. It is what a human does reading the compressed
  five-column view -- and it is why the field report's "average SM = 12.5%" was
  misread: 12.5% on 8 cards is one card at 100%, and the pooled curve cannot tell
  you which one.

A reports a set, B reports a count. They agree when `len(set) == count`. When they
do not, something is wrong with the threshold, the plateau, or the assumption that
cards are either working or idle -- and the correct output is a refusal rather than
whichever answer came first.
"""
from . import segment as _seg

# How far the estimated card count may sit from an integer before the estimate is
# not a card count. 0.35 is generous: at 8 cards one card is 12.5 points of pooled
# mean, so 0.35 cards is ~4.4 points of slack for clock and sampling wobble.
COUNT_TOLERANCE = 0.35
# Fraction of a configuration's samples that must agree for the configuration to pass.
MIN_AGREEING_FRACTION = 0.90


class Row:
    __slots__ = ("index", "label", "n_set", "n_est", "agrees")

    def __init__(self, index, label, n_set, n_est, agrees):
        self.index = index
        self.label = label
        self.n_set = n_set
        self.n_est = n_est
        self.agrees = agrees


class CrossCheck:
    __slots__ = ("agree", "rows", "plateau", "n_gpus", "by_label", "basis")

    def __init__(self, agree, rows, plateau, n_gpus, by_label, basis):
        self.agree = agree
        self.rows = rows
        self.plateau = plateau
        self.n_gpus = n_gpus
        self.by_label = by_label        # {label: (n_agree, n_total)}
        self.basis = basis              # 'compared' | 'unknown'

    @property
    def worst(self):
        """(label, agreeing_fraction) for the configuration that agreed least."""
        worst = None
        for label, (ok, total) in self.by_label.items():
            frac = ok / total if total else 0.0
            if worst is None or frac < worst[1]:
                worst = (label, frac)
        return worst


def pooled_counts(series, column, plateau, n_gpus):
    """Method B: estimate how many cards were busy, from the pooled mean alone."""
    out = []
    for index, per_gpu in series.per_sample(column):
        if not per_gpu:
            continue
        pooled = sum(per_gpu.values()) / n_gpus
        out.append((index, pooled / plateau * n_gpus if plateau else None))
    return out


def compare(series, segs, threshold, column="sm_pct"):
    """Run both methods and report whether they label the same samples the same way."""
    n_gpus = len(series.gpus)
    if not segs or threshold.value is None or n_gpus == 0:
        return CrossCheck(None, [], None, n_gpus, {}, "unknown")

    # The busy plateau is measured, not assumed to be 100. That is the whole reason a
    # fixed threshold breaks across workloads: the plateau was 48.4% on one real
    # detection job and 94.1% on a segmentation job on the same cards.
    #
    # It is the MEDIAN of the busy readings, and that detail is what makes this a
    # check at all. With the mean, the two methods are algebraically the same thing:
    # pooled = sum/N and plateau = sum/k, so pooled/plateau*N returns k identically,
    # for any file, however ragged the cards. The first version used the mean, agreed
    # with itself on every input, and would have reported "two independent
    # segmentations agree" for the rest of time. The median does not absorb a card
    # sitting at part load, so the disagreement survives to be reported.
    plateau = threshold.high_median
    if not plateau:
        return CrossCheck(None, [], None, n_gpus, {}, "unknown")

    est = dict(pooled_counts(series, column, plateau, n_gpus))
    rows, by_label = [], {}
    for s in segs:
        if s.kind != _seg.ACTIVE:
            continue
        n_set = len(s.busy_gpus)
        ok = total = 0
        for index in range(s.i0, s.i1 + 1):
            if index not in est or est[index] is None:
                continue
            n_est = est[index]
            agrees = abs(n_est - n_set) <= COUNT_TOLERANCE
            rows.append(Row(index, s.label, n_set, n_est, agrees))
            total += 1
            ok += 1 if agrees else 0
        if total:
            prev = by_label.get(s.label, (0, 0))
            by_label[s.label] = (prev[0] + ok, prev[1] + total)

    if not by_label:
        return CrossCheck(None, rows, plateau, n_gpus, {}, "unknown")
    agree = all(ok / total >= MIN_AGREEING_FRACTION
                for ok, total in by_label.values() if total)
    return CrossCheck(agree, rows, plateau, n_gpus, by_label, "compared")
