# Cutting a capture into phases

A long telemetry file is not one measurement. It is a model load, some idle, a
configuration, a gap, another configuration. Every number worth quoting is scoped to
one of those, and the whole-file mean is scoped to none of them.

This is how `telemetrydoctor` finds the boundaries, and — more to the point — when it
declines to.

## The threshold comes from the file

The procedure this automates binned by `sm >= 90`. Three real workloads on 8-card
nodes plateau at **48.4 %**, **69.9 %** and **94.1 %**, so that constant selects
nothing on two of them. See [rules.md](rules.md#tl007--a-fixed-busy-threshold).

`segment.otsu()` sweeps every candidate cut and keeps the one maximising the variance
*between* the two groups. It is the standard one-dimensional two-class split, it needs
no libraries, and it has no parameters to tune — which matters, because a parameter is
just a constant with better manners.

```console
$ telemetrydoctor audit tests/fixtures/EXAMPLE_detection_plateau48.csv --rule TL007
threshold: [info] busy threshold 0.2 computed from this file (Otsu, separability 0.99); busy plateau 48.4, idle level 0.0. A fixed 90 threshold would discard 234 of the 234 card-samples this file says were working -- every one of them.
```

## The refusal

Otsu returns a cut for **any** input, including one with nothing to cut. So a cut is
only used when the two classes are actually separated:

| distribution | separability |
|---|---|
| uniform band, no gap | **0.77** |
| two plateaux with idle between them | **0.99 – 1.00** |

`MIN_SEPARABILITY = 0.90` sits in that gap. Both figures are pinned by tests, so the
floor cannot quietly stop working if either moves.

```console
$ telemetrydoctor audit tests/fixtures/EXAMPLE_unimodal.csv --rule TL007
threshold: [unknown] no usable busy/idle split: separability 0.77 is under the 0.90 floor, so the distribution of sm_pct is effectively one class. This file cannot be phase-segmented, and every phase-scoped number below is withheld rather than computed against a threshold cut down the middle of one mode.

$ telemetrydoctor aggregate tests/fixtures/EXAMPLE_unimodal.csv
source: [info] 90 rows x 8 card(s), wide_csv format; span 122 s; 1.375 s/row measured
all: [unknown] refusing to aggregate: no usable phase split, so any mean would span the load, the idle gaps and the work at once
```

An earlier floor of 0.70 sat *below* the uniform case, and confidently cut that file
at 49.17 — a threshold with two decimal places, derived from data that contained no
boundary. Precision is not accuracy, and Otsu will supply as much of the first as you
ask for.

## Boundaries the busy set cannot see

Grouping samples by *which cards are busy* finds every boundary where the
configuration changes. It finds none of the boundaries where it does not.

The model-weight load is the case that matters: several GB of host-to-device transfer
with the SMs nearly idle. It sits **below** the busy threshold, so it has the same
busy set as the idle stretches on either side of it — and gets glued to them.

That is not hypothetical; it is what the first version of `segment.py` did:

```text
load[0-35]: [info] 0 card(s) busy {}, 36 samples, 48.2 s
```

Twenty idle samples, twelve load samples and four more idle, as one 36-sample phase
whose mean rx was 1.83 GB/s — neither the idle floor (0.0004) nor the load rate
(3–8). And because the whole run was then classified `load`, the measured noise floor
came out at `pcie_rx_gbs = 0.296 GB/s`: **the "noise floor" was 700× too high because
it was really a measurement of the weight load.**

The fix is a second signal. Samples are grouped by `(busy set, load flag)` together,
where the load flag is high receive rate with low SM — the signature the analysis note
describes as 「加载期间 SM 只有 0-14%」. Either changing is a boundary:

```console
$ telemetrydoctor phases tests/fixtures/EXAMPLE_matrix_8gpu.csv
idle[0-19]: [info] 0 card(s) busy {}, 20 samples, 26.2 s
load[20-31]: [info] 0 card(s) busy {}, 12 samples, 15.1 s
idle[32-35]: [info] 0 card(s) busy {}, 4 samples, 4.2 s
active[36-54]: [info] 1 card(s) busy {0}, 19 samples, 24.7 s
idle[55-58]: [info] 0 card(s) busy {}, 4 samples, 4.1 s
active[59-81]: [info] 2 card(s) busy {0,1}, 23 samples, 30.3 s
...
active[108-128]: [info] 4 card(s) busy {0,1,5,6}, 21 samples, 27.5 s
...
active[133-151]: [info] 8 card(s) busy {0,1,2,3,4,5,6,7}, 19 samples, 24.8 s
```

Note `{0,1,2,3}` and `{0,1,5,6}` staying distinct. Both are four cards; one spans two
NUMA domains and the other does not. A segmentation that binned by *count* would merge
them and average two different topologies into one figure.

## The cross-check, and why it nearly wasn't one

The analysis note's last step is 「换方法独立复核（交报告前必做）」: recut by a second
method and believe the numbers only if both agree.

- **A** reads per-card values → a *set*.
- **B** reads only the pooled mean and divides by the busy plateau → a *count*.

They agree when `len(set) == count`.

The first implementation divided by the **mean** of the busy readings. Given
`plateau = sum/k` and `pooled = sum/N`, the estimate `pooled/plateau × N` reduces to
`k` — identically, for every possible input. Method B was recomputing method A's
answer with extra steps, and would have reported agreement forever.

A test written to force a disagreement is what surfaced it. The plateau is now the
**median** of the busy readings, which does not get dragged down by a card at part
load:

```python
# four cards at 99%, one at 50%, three idle
by_mean[i]   == 5.0     # equals the busy-set size, always
by_median[i] == 4.5     # sees four full cards and a half one -> disagreement
```

```console
crosscheck: [violation] the two segmentations disagree: in {0,1,2,3,4} only 0% of samples agree on how many cards were busy (floor 90%). Either the threshold is wrong, the plateau is not flat, or cards are neither working nor idle. No per-configuration figure from this file should be reported until that is resolved.
```

`aggregate` refuses to print per-configuration numbers while that stands. `--force`
overrides it, and the output says you did.

## What this cannot do

- **No `sm_pct`, no phases.** A workload that never moves the SM counter does not
  segment. `--column` can point the split at another column, but there is no default
  that works for a file with nothing bimodal in it.
- **A steady state that looks like a load is labelled `load`.** High receive, low SM
  is the load signature; a workload whose real steady state has that shape will be
  misfiled. Nothing in a telemetry file distinguishes them, and inventing a signal
  that did would be precisely the failure this repository is about.
- **Two visits to the same configuration stay separate.** A run that was interrupted
  and restarted produces two segments with the same label, and they are not
  concatenated — two machine states silently sharing one mean is the thing being
  avoided.
