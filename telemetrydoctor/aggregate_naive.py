"""The first version. Kept, and kept failing, on purpose.

This is what the analysis looked like before any of the contracts existed: read the
whole file, average every column, take every maximum, multiply the rates by elapsed
time to get totals. It is four lines of arithmetic and every line of it is defensible
in isolation.

It produced a number that went into a document: **8 cards, 47 GB/s of PCIe traffic.**
That figure was wrong three times over.

  1. It averaged across phases. Most of that file was the model weight load -- a
     multi-gigabyte host-to-device transfer with the SMs idle -- and the load
     dominates any whole-file mean.
  2. It reported the peak of a windowed counter. The PCIe reading covers about
     20 ms out of each sampling interval, and all-reduce traffic is bursty, so the
     maximum says whether a sample happened to land on a burst.
  3. It integrated a windowed rate into a total. At 1.376 s per row the counter is
     observing 1.45% of the elapsed time.

`telemetrydoctor` exists because none of those three raise, warn, or look wrong.
The number was plausible: 47 GB/s across 8 cards on PCIe 5.0 is a perfectly
believable figure, which is why it survived review.

This module and `tests/test_aggregate_naive.py` stay in the repository and CI re-runs
them on every push, following the same precedent as `regressiondoctor`'s `diff_naive`
and `fitdoctor`'s `budget_naive`. Deleting them would delete the evidence that the
contract table was hit rather than imagined.
"""


def naive_summary(series, nominal_interval_s=1.0):
    """Average everything over the whole file, peak everything, integrate everything.

    No phases, no scopes, no contracts, and the nominal interval used as if it were
    the measured one.
    """
    out = {}
    for column in series.columns:
        vals = [r.values[column] for r in series.readings if column in r.values]
        if not vals:
            continue
        n_samples = series.n_samples
        out[column] = {
            "mean": sum(vals) / len(vals),
            "peak": max(vals),
            # summing a per-card mean by multiplying by the card count -- valid only if
            # every card is symmetric at every instant, which ring all-reduce is not
            "sum_across_gpus": (sum(vals) / len(vals)) * len(series.gpus),
            "total_over_run": (sum(vals) / len(vals)) * n_samples * nominal_interval_s,
            "n": len(vals),
        }
    return out


def naive_peak_pcie_sum(series):
    """"What was the highest total PCIe traffic we saw?" -- the 47 GB/s question.

    Takes the per-card maximum, then adds those maxima across cards. Two errors
    compounded: each maximum is a windowed counter's luckiest sample, and the
    maxima did not occur at the same instant, so their sum describes a moment that
    never happened.
    """
    total = 0.0
    parts = {}
    for gpu in series.gpus:
        for column in ("pcie_tx_gbs", "pcie_rx_gbs"):
            vals = series.column(column, gpu=gpu)
            if vals:
                m = max(vals)
                parts[(gpu, column)] = m
                total += m
    return total, parts
