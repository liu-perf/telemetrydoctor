# telemetrydoctor

**What arithmetic does this telemetry column support?**

Point it at an `nvidia-smi` or NVML capture. It cuts the file into phases, computes
the aggregates that are sound, and refuses the ones that are not — with the size of
the error attached, because "not permitted" persuades nobody and "wrong by 69×"
persuades everybody.

Run against a shipped fixture, so this is reproducible after `pip install -e .`:

```console
$ telemetrydoctor audit tests/fixtures/EXAMPLE_matrix_8gpu.csv --interval 1.0
source: [info] 156 rows x 8 card(s), wide_csv format; span 213 s; 1.375 s/row measured
interval: [warn] measured 1.375 s/row over 155 gaps (median 1.375, min 1.356, max 1.396), against a nominal 1.000 s -- +37.5%. Any figure computed as value/nominal is off by that much, in the direction that flatters the machine.
phases: [warn] 56 of 156 samples (36%) are not steady work -- 100 active samples, 44 idle samples, 12 load samples. Configurations found: {0,1,2,3,4,5,6,7} {0,1,2,3} {0,1,5,6} {0,1} {0}. A mean over the whole file blends them, and the load stretch is the dangerous one: it is a multi-gigabyte host-to-device transfer with the SMs idle, so averaging it into training traffic is what produced a fabricated '8 cards, 47 GB/s'.
integrate: [violation] pcie_rx_gbs, pcie_tx_gbs observe 20 ms of each 1.375 s interval -- 1.45% of elapsed time. Multiplying a mean by elapsed time to get a total understates it by about 69x. The mean itself is fine as a mean rate; it is the product that is not a total.
peak: [warn] pcie_rx_gbs, pcie_tx_gbs carry a windowed reading, so their maxima are not peak rates. At 1.45% duty the maximum is drawn from 1.45% of the traffic, so it is a statement about which samples got lucky. Report the mean, or record with a tool that integrates in hardware.
scope: [warn] in {0} the busy-card mean is 98.9% and the all-card mean is 12.4% -- 86.5 points apart, because 7 of 8 cards were idle. Both are correct answers to different questions and a report that prints only the second one invites the reader to conclude the cards were not working. That has happened.
threshold: [info] busy threshold 12.1 computed from this file (Otsu, separability 0.99); busy plateau 98.3, idle level 0.8.
crosscheck: [ok] two independent segmentations agree on every configuration: per-card identity (which cards are above 12.1) and pooled level alone (all-card mean / 98.3 plateau x 8 cards). The second never looks at an individual card, so agreement is not the first method checking itself.
```

Zero dependencies, pure standard library, `--json` and `--fail-on` for CI.

---

## Why this exists

A monitoring CSV looks like a spreadsheet, so people do spreadsheet things to it:
average a column, take a maximum, multiply a rate by a duration. Every one of those
operations runs. None of them raises. And on a GPU telemetry file, three of them are
wrong in ways that produce believable numbers.

This repository is the automation of a hand-written analysis discipline — a page of
rules and a stack of `awk` one-liners that existed because each rule had already been
broken once. The headline casualty was a figure that reached a document: **"8 cards,
47 GB/s of PCIe traffic."** Plausible for PCIe 5.0. Wrong three times over: it
averaged the model-weight load into the training traffic, it reported the maximum of
a counter that only observes 20 ms out of every 1.376 s, and it multiplied a rate by
elapsed time as though the counter had been watching the whole time.

Run the naive version on the shipped fixture and the same shape comes straight back
out:

```console
$ pytest tests/test_aggregate_naive.py -q -s
  naive peak-sum across cards:  84.64 GB/s
  steady-phase per-sample sum:   3.86 GB/s
  overstatement:                 21.9x

  naive sm_pct summed across 8 cards: 239.47%

  whole-file sm mean:         29.9%
  8-card steady busy mean:    98.2%
6 passed in 0.05s
```

`PASSED` there does not mean healthy. Those tests assert that the first version is
**still** wrong, and CI re-runs them on every push — the same precedent as
[`regressiondoctor`](https://github.com/liu-perf/regressiondoctor)'s `diff_naive` and
[`fitdoctor`](https://github.com/liu-perf/fitdoctor)'s `budget_naive`.

---

## The three findings

### 1. `utilization.gpu` is a time measure, not an intensity

This is the one that changes how you read every dashboard you own.

`examples/verify_on_device.py` runs two workloads on one RTX 5060 Ti and asks
`nvidia-smi` what it sees. Reproduced across three consecutive runs:

| workload | delivered | `utilization.gpu` reports |
|---|---|---|
| 64×64 matmul, submitted back-to-back with no gap | **0.046 TFLOPS** | **29.6 %** |
| 4096×4096 matmul at quarter duty | **45.4 TFLOPS** | **27.9 %** |

**978× apart in real throughput. 1.7 points apart on the dashboard.** A panel reading
"29 % GPU utilisation" is showing the same number for both.

The column is not lying. It reports the fraction of the sample window in which at
least one kernel was resident, and it is blind to what that kernel did with the
hardware. It is **not** SM occupancy and **not** tensor-core activity; for those you
need DCGM prof counters or Nsight.

The consequence in the field: one 8-card node reported `sm=48.4%` on a detection
training job and `sm=94.1%` on a segmentation job — same cards, same day. Those two
numbers do not rank those two jobs.

A second finding falls out of the same experiment. The 64×64 arm was submitting with
**no host-side gap at all** — 100 % of wall clock inside the submit loop — and the
device still reported 29.6 %. The missing 70.4 points are launch overhead: real
device idle that a host-side timer cannot see. *"My training loop never sleeps"* is
not evidence that the GPU is busy.

### 2. The sampling interval is never the nominal one

A loop ending in `sleep(1.0)` does not produce one row per second. On a real 8-card
node reading four NVML counters per card, the body cost 0.37 s and the file came out
at **1.376 s/row — 37.6 % over nominal**. Every per-second figure computed with the
nominal value is 37.6 % wrong, in the direction that flatters the machine.

On one card through `nvidia-smi -lms 250` the same measurement gives **+3.0 %**.
Small, still not zero, and the gap between the two is the point: the overshoot scales
with how much work the sampler does, so it cannot be known from reading the loop —
only from the timestamps in the file it produced.

A file with no timestamps therefore has no measured interval, and `telemetrydoctor`
reports `unknown` rather than filling one in.

### 3. A fixed busy threshold does not survive a change of workload

The discipline this tool automates binned samples by which cards had `sm >= 90`. That
worked, and then it did not:

| workload, all on 8-card nodes | plateau |
|---|---|
| detection training (YOLOX) | **48.4 %** |
| inference decode (Qwen3-VL) | **69.9 %** |
| segmentation training (KNet-Swin-L) | **94.1 %** |

`sm >= 90` selects **zero** rows on two of the three. The inference file had to be
re-run with `sm >= 20` — a second constant, chosen the same way, for the same reason.

So the threshold is computed per file (Otsu's method, no libraries, no parameters),
and — more importantly — **refused** when the data does not support one. A file whose
`sm` distribution is a single mode cannot be phase-segmented, and the correct output
is to say so:

```console
threshold: [unknown] no usable busy/idle split: separability 0.77 is under the 0.90 floor, so the distribution of sm_pct is effectively one class. This file cannot be phase-segmented, and every phase-scoped number below is withheld rather than computed against a threshold cut down the middle of one mode.
```

The 0.90 floor is measured, not chosen: a uniform band with nothing to cut scores
0.77, and a real two-plateau file scores 0.99–1.00.

---

## The contract table

Every column carries a record of what may be done to it. This is the third generation
of the same honesty device: `regressiondoctor` attaches `noise_basis` to a verdict and
`fitdoctor` attaches `basis` to a number — both answer *where did this come from*.
This one answers *what may I do with it*.

```console
$ telemetrydoctor contracts
fb_used_mib: [info] MiB [instantaneous], window instant; mean=within_phase peak=ok integrate=forbidden sum_across_gpus=ok
mem_pct: [info] % [time_fraction], window 167-1000 ms; mean=within_phase peak=ok integrate=forbidden sum_across_gpus=forbidden
pcie_rx_gbs: [info] GB/s [windowed_rate], window 20 ms; mean=ok peak=meaningless integrate=forbidden sum_across_gpus=ok
pcie_tx_gbs: [info] GB/s [windowed_rate], window 20 ms; mean=ok peak=meaningless integrate=forbidden sum_across_gpus=ok
power_w: [info] W [instantaneous], window instant; mean=ok peak=ok integrate=ok sum_across_gpus=ok
sm_clock_mhz: [info] MHz [instantaneous], window instant; mean=within_phase peak=ok integrate=forbidden sum_across_gpus=forbidden
sm_pct: [info] % [time_fraction], window 167-1000 ms; mean=within_phase peak=ok integrate=forbidden sum_across_gpus=forbidden
temp_c: [info] degC [instantaneous], window instant; mean=within_phase peak=ok integrate=forbidden sum_across_gpus=forbidden
```

Four verdicts rather than a boolean, because the failures are not the same failure:

| verdict | meaning |
|---|---|
| `ok` | sound |
| `within_phase` | sound over one phase; a mean across load + idle + work describes none of them |
| `meaningless` | it returns a number that answers no question |
| `forbidden` | it returns a wrong answer whose size is known |

**`power_w` is the control.** It is instantaneous, so `mean × elapsed` really is
joules — the one sound integration in a telemetry file. Without it the table would be
a list of prohibitions and could not be falsified.

An unrecognised column gets **no** contract rather than a permissive default, and
aggregation is refused. Permissiveness is what goes wrong.

---

## The ten rules

| | fires on |
|---|---|
| TL001 | nominal interval used where a measured one was available, or none available at all |
| TL002 | a whole-file mean spanning load, idle and work |
| TL003 | a windowed rate integrated into a total (with the error factor) |
| TL004 | the maximum of a windowed rate reported as a peak rate |
| TL005 | busy-card mean and all-card mean far apart, only one of them quoted |
| TL006 | the noise floor assumed rather than measured from this file's own idle |
| TL007 | a busy threshold that was supplied, or one the data cannot support |
| TL008 | `utilization.gpu` read as an intensity |
| TL009 | two independent segmentations disagreeing |
| TL010 | polling faster than the driver's window and counting the repeats as samples |

Provenance for each — which analysis note, which terminal session, which experiment —
is in [`docs/rules.md`](docs/rules.md). None were invented by reading the NVML
documentation and imagining what could go wrong.

---

## Install and use

```bash
git clone https://github.com/liu-perf/telemetrydoctor
cd telemetrydoctor
pip install -e .
```

Three input formats, because all three turn up:

```bash
# the wide pynvml layout: one row per sample, five columns per card
telemetrydoctor audit pcie_train.csv

# nvidia-smi --query-gpu, long form. --columns is required: `--format=csv,noheader`
# deletes the only record of which column is which
telemetrydoctor audit run.csv --format query_csv \
  --columns "timestamp,index,utilization.gpu,power.draw"

# nvidia-smi dmon. Without -o DT it has no timestamps at all, and TL001 says so
telemetrydoctor audit dmon.txt
```

Then:

```bash
telemetrydoctor phases    run.csv    # where the load, the idle and each config are
telemetrydoctor aggregate run.csv    # every legal aggregate, and every refusal
telemetrydoctor contracts            # the table above
```

CI gate, threshold semantics (`--fail-on warn` also trips on `violation`):

```bash
telemetrydoctor audit run.csv --fail-on violation ; echo $?
```

Reporting a problem and failing a build are separate decisions: without `--fail-on`,
a `[violation]` still exits 0.

---

## What ships

Seven fixtures, all synthetic, each declaring itself on its first line — in the data,
not in the filename, so the declaration survives being pasted into a slide. They are
built to targets from one real 8-card capture (the per-configuration PCIe sums, the
98–99 % plateau, the idle floor, the 1.376 s interval, 545 W per busy card), and
`tests/fixtures/make_fixtures.py` regenerates them byte-identically.

The realised means do **not** match the targets exactly, and that is left alone: over
19–23 samples the number of bursts can only be an integer, so ±1 burst moves a
configuration's mean by several percent. That residual is the same quantisation that
makes a real capture's mean uncertain at these sample counts — which is what TL004 is
about. A fixture matching to three decimals would be asserting a precision this kind
of measurement does not have.

---

## Verifying on your own card

```bash
python examples/verify_on_device.py
```

Needs torch and a CUDA card; nothing else in the repo does. It measures the sampling
interval, the idle noise floor, the repeat rate at 250 ms polling, and runs the
duty-versus-intensity experiment above. If your machine disagrees with the table in
this README, **keep your output and open an issue** — that disagreement is
information, and re-running until the terminal matches the documentation is how a
measurement tool turns into an advertisement.

---

## Limits

- Thresholds and phases are computed from `sm_pct` by default. A workload that never
  moves the SM counter (pure host-side stalls, or a card doing only copies) will not
  segment, and the tool reports `unknown` rather than guessing.
- The load/work distinction uses the receive rate and the SM level together. A
  workload whose steady state genuinely looks like a weight load — high rx, low SM —
  will be labelled `load`. There is no signal in a telemetry file that separates
  those two, and inventing one would be the mistake this repo is about.
- `sm_pct` is what NVML gives. For SM occupancy or tensor-core activity you need DCGM
  prof counters or Nsight, and this tool does not read either.
- One capture, one machine. Comparing two runs is
  [`regressiondoctor`](https://github.com/liu-perf/regressiondoctor)'s job.

---

## The other five

A series about not fooling yourself with a measurement:

1. [nodebench](https://github.com/liu-perf/nodebench) — measure a multi-GPU node
2. [benchdoctor](https://github.com/liu-perf/benchdoctor) — read the measuring script for traps
3. [tracedoctor](https://github.com/liu-perf/tracedoctor) — read a kernel trace for what static analysis cannot see
4. [regressiondoctor](https://github.com/liu-perf/regressiondoctor) — is this run worse than the last one, or is that noise
5. [fitdoctor](https://github.com/liu-perf/fitdoctor) — what should this number have been
6. **telemetrydoctor** — what may I compute from this column

Key figures from all six are on one page:
[**Dashboard**](https://liu-perf.github.io/nodebench/dashboard.html).

MIT licensed.
