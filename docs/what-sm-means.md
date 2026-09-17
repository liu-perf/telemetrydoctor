# What `utilization.gpu` actually measures

The finding this repository is built around, and the experiment that settled it.

## The claim

`nvidia-smi`'s `utilization.gpu` — `sm` in `dmon`, `nvmlDeviceGetUtilizationRates`
in NVML — is **the fraction of the driver's sample window in which at least one
kernel was resident on the device.**

It is a *time* measure. It says nothing about how much of the hardware that kernel
used. It is not SM occupancy, not tensor-core activity, not achieved FLOPS, and not
"how hard the card is working". For any of those you need DCGM prof counters or
Nsight, which this tool does not read.

That is documented behaviour, not a discovery. What was worth measuring is **how
large the resulting gap is in practice**, because a well-known caveat that everyone
nods at and then ignores needs a number attached before it changes anything.

## The experiment

`examples/verify_on_device.py`. One RTX 5060 Ti, torch 2.11.0+cu128, CUDA 12.8,
Windows 11. A background `nvidia-smi --query-gpu=timestamp,utilization.gpu,power.draw,clocks.sm
--format=csv,noheader,nounits -lms 250` samples throughout.

Two arms, run through **the same** duty-cycle loop so that only the kernel size
differs:

- `light_64` — a 64×64 bf16 matmul, submitted in a tight loop. Tiny work per launch,
  so launch overhead dominates.
- `heavy_4096` — a 4096×4096 bf16 matmul. About 137 GFLOP per call, ~3 ms each.

Each arm runs at a host-side duty of 25 %, 50 % and 100 %, eight seconds per point.

## The output

```text
device: [info] NVIDIA GeForce RTX 5060 Ti, 8,150 MiB total, torch 2.11.0+cu128, CUDA 12.8
interval: [info] nominal 250 ms, measured 257.5 ms over 251 gaps (+3.0%); the field's 8-card pynvml loop asked for 1.0 s and got 1.376 s (+37.6%) -- overhead grows with the number of cards, so the nominal value is never the one to divide by
noise_floor: [info] 20 idle rows: utilization.gpu 0.0%, power 2.51 W, sm clock 180 MHz -- idle is a measured quantity, not zero, and it is not the same number on another file from the same machine
oversampling: [info] 252 rows, 64 of them differ from the row before (75% are repeats). NVML documents the utilization window as 1 s to 1/6 s, so polling at 250 ms returns the same reading several times -- more rows, not more information

  workload           n  duty_set  host_busy  util.gpu_rpt    TFLOPS
  light_64          64       25%      25.6%          7.0%     0.044
  light_64          64       50%      46.9%         13.5%     0.045
  light_64          64      100%     100.0%         29.6%     0.047
  heavy_4096      4096       25%      28.7%         27.9%    45.579
  heavy_4096      4096       50%      47.3%         46.1%    45.993
  heavy_4096      4096      100%     100.0%         99.1%    46.588

same_reading: [ok] utilization.gpu reads 29.6% on a 64x64 workload delivering 0.047 TFLOPS and 27.9% on a 4096x4096 workload delivering 45.579 TFLOPS -- 1.7 points apart, 973x apart in delivered throughput. A dashboard showing '29% GPU utilisation' is showing the same number for both.
launch_bound: [info] the 64x64 arm was submitting work back-to-back with no host-side gap (100.0% of wall clock inside the burst loop) and the device still reported 29.6%. The missing 70.4 points are launch overhead: real device idle that a host-side timer cannot see. 'My training loop never sleeps' is not evidence that the GPU is busy.
verdict: [ok] utilization.gpu is a *time* measure -- the fraction of the sample window in which at least one kernel was resident -- and it is blind to what that kernel did with the hardware. It is not SM occupancy and not tensor-core activity; for those you need DCGM prof counters or Nsight. Which is why one real 8-card node showed sm=48.4% on detection training and sm=94.1% on segmentation training: those two numbers do not rank the two jobs.
```

Three consecutive runs, unedited:

| run | light @100 % | heavy @25 % | apart | ratio |
|---|---|---|---|---|
| 1 | 29.6 % / 0.047 TFLOPS | 27.9 % / 45.579 | 1.7 pts | 973× |
| 2 | 29.6 % / 0.046 | 27.7 % / 45.382 | 1.9 pts | 980× |
| 3 | 29.6 % / 0.046 | 27.9 % / 45.349 | 1.8 pts | 978× |

**Two workloads three orders of magnitude apart in delivered arithmetic, reported by
the same column as the same number.**

## The prediction that was wrong, and why it is in this document

The first version of this experiment predicted something stronger and different: that
`utilization.gpu` would faithfully report the *duty cycle* and be blind only to
intensity. So it paired the arms by duty setting — light@25 % against heavy@25 %, and
so on — and expected each pair to match.

They did not. At the 100 % setting, light read 29.6 % and heavy read 99.1 %: 70 points
apart, and the script correctly reported `[differs]` and refused to claim the result.

The prediction was wrong because the *host-side* busy fraction is not the *device-side*
one. The light arm sat inside its submit loop 100 % of wall clock, but between those
tiny kernels the GPU genuinely had nothing resident. `utilization.gpu` was reporting
the device accurately; the host-side timer in my own harness was the thing that could
not see the gaps.

The fix was to pair by *reported utilisation* instead of by duty setting, at which
point the decisive comparison — 29.6 % versus 27.9 % — was already in the data. The
corrected claim is narrower than the original guess and better supported.

Two things worth keeping from that:

1. **The launch-overhead result is the more actionable half.** A training loop that
   never sleeps can still leave the device idle 70 % of the time, and no host-side
   instrumentation will show it. That is the gap `tracedoctor` was built to look
   into.
2. **The script reported `[differs]` rather than a result.** Had it been written to
   print a conclusion regardless, the wrong claim would have gone into this document
   with a table of real numbers underneath it, and it would have looked exactly as
   convincing as the correct one does now.

## Why it matters on a real machine

From a real 8-card node, three workloads, same cards:

| workload | plateau `sm` |
|---|---|
| detection training (YOLOX) | 48.4 % |
| inference decode (Qwen3-VL) | 69.9 % |
| segmentation training (KNet-Swin-L) | 94.1 % |

The obvious reading — "segmentation uses the hardware nearly twice as well as
detection" — is not supported by these numbers. Detection at 48.4 % was scaling
linearly to 8 cards at 82.6 % efficiency; its SM figure is low because its kernels
are short relative to the gaps between them, which is a statement about kernel
shape, not about waste.

The practical consequences:

- **Do not rank two jobs by this column.** Rank by throughput — images/s, tokens/s,
  steps/s — which is what you actually care about and which does not need a caveat.
- **Do not set an alert on it.** "GPU utilisation below 50 %" fires on a healthy
  detection job and stays quiet on a job doing 0.05 TFLOPS in a tight launch loop.
- **Do use it for what it is good at**: telling which cards were participating.
  `telemetrydoctor` uses exactly that and nothing more — the busy *set*, never the
  busy *amount*. See [phases-and-thresholds.md](phases-and-thresholds.md).
- **If you need occupancy or tensor-core activity, use the tool that measures them.**
  DCGM `DCGM_FI_PROF_SM_OCCUPANCY` / `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE`, or Nsight
  Compute. They cost more to collect. That is why the cheap number is the one on
  every dashboard, and why it is worth knowing what it is.
