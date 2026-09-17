# The ten rules, and where each one was hit

Every rule below has a provenance line. None of them were produced by reading the
NVML documentation and imagining what could go wrong — the failures that happen in
practice are a small and unobvious subset of the failures that are possible, and a
rule set derived from the manual would be mostly the second kind.

Sources, referred to by short name:

- **the analysis note** — a hand-written page of rules for reading one sampler's CSV,
  written after each rule had been broken once.
- **the session** — a real terminal session running that analysis over two captures
  from one 8-card node.
- **the device run** — [`examples/verify_on_device.py`](../examples/verify_on_device.py)
  on one RTX 5060 Ti; output in [what-sm-means.md](what-sm-means.md).
- **this repo** — a mistake made while building `telemetrydoctor` itself.

---

## TL001 — the nominal sampling interval

**Fires when** the file's measured interval is more than 5 % from the nominal one, or
when the file carries no timestamps at all.

**Provenance: the session.** Two captures, one sampler, `sleep(1.0)` at the bottom of
the loop:

```text
=== 0. sampling interval (nominal 1.0s, must be measured) ===
pcie_train.csv   rows=  211 span=  289s interval=1.376s/row
pcie_infer.csv   rows= 5364 span= 7198s interval=1.342s/row
```

Eight cards × four NVML calls each, plus a print, cost 0.37 s per row. Every
per-second figure computed against the nominal 1.0 s is 37.6 % wrong, in the direction
that flatters the machine.

The device run gives the other end of the range: one card, `nvidia-smi -lms 250`,
measured 257.5 ms — **+3.0 %**. The overshoot scales with the sampler's own work, so
it is not a constant anyone can look up. It has to be measured from the file.

**Status is `unknown`, not `ok`, for a file with no timestamps.** `dmon` without
`-o DT` produces one, and there is nothing in it to measure against.

---

## TL002 — a mean that spans phases

**Fires when** the file contains load, idle and work and a whole-file mean is
therefore not a statement about any of them.

**Provenance: the analysis note.**

> 必须先用 `sm_pct` 区分「模型加载」和「训练稳态」。加载大模型权重（如 8B = 16 GB
> 主机内存→显存）本身就是一次几十 GB/s 的 H2D 传输，若不排除，会把加载流量误当训练
> 通信（**曾因此得出"8 卡 47 GB/s"的假数据**）。

That parenthesis is the reason this repository exists. 47 GB/s across 8 cards on
PCIe 5.0 is a believable figure, which is why it survived review and reached a
document.

---

## TL003 — integrating a windowed rate

**Fires when** a column that summarises a fixed hardware window is multiplied by
elapsed time to produce a total. Reports the error factor.

**Provenance: the analysis note.**

> NVML 的 `nvmlDeviceGetPcieThroughput` 返回"过去约 20 ms 的平均值"，而采样间隔
> 1.37 s——每 1.37 s 只有 20 ms 被真正测到。…均值可作为平均速率的估计，但**不可乘以
> 时间当作总传输量**。

20 ms out of 1.376 s is a **1.45 % duty cycle**, so `mean × elapsed` understates the
true total by about **69×**. The mean is still a fine estimate of the mean rate; it is
the product that is not a total.

---

## TL004 — the maximum of a windowed rate

**Fires when** a windowed rate column is present, because its maximum reports sampling
luck rather than peak traffic.

**Provenance: the session.** Per-card tx during an 8-card all-reduce, one row per
sample, GB/s:

```text
12:40:13   3.06  0.01  0.01  0.01  0.01  2.99  1.77  0.00
12:40:15   0.01  3.00  1.73  0.00  0.01  0.01  0.01  0.01
12:40:18   0.00  0.01  0.01  0.01  0.01  0.01  0.00  0.49
12:40:19   3.06  0.01  0.01  0.01  0.01  2.98  1.77  0.00
```

Ring all-reduce moves data hop by hop, so at any instant only some links are active —
and the row at `12:40:18` caught none of them. With the counter observing 1.45 % of
elapsed time and the traffic arriving in bursts, the maximum answers "did a sample
land on a burst", which is a question about the sampler.

---

## TL005 — which cards the percentage was averaged over

**Fires when** the busy-card mean and the all-card mean are more than 5 points apart.

**Provenance: the analysis note**, which records the consequence rather than just the
rule:

> 「全卡平均 SM」和「参与卡 SM」是两个东西，报表里必须分开写。…若在报表里只写一列
> 「平均 SM = 12.5%」，看的人会判断成"卡根本没跑满、数据不合理"（**实际发生过**）。
> 真实利用率要只对参与该配置的卡求平均——实测五档均为 99.7%~99.8%。

On an 8-card node running a 1-card configuration, the two answers are **99.0 %** and
**12.5 %**. Both are correct. Printing only the second one caused a reader to conclude
the hardware was idle.

`telemetrydoctor` computes both, names each in its own label
(`sm_pct.mean.busy_mean` and `sm_pct.mean.all_device_mean`), and prints them on
adjacent lines so that quoting one without the other takes deliberate effort.

---

## TL006 — a noise floor that was assumed

**Fires** always, as `info`, reporting the floor measured from this file's own idle
stretches — or `unknown` when the file never goes idle.

**Provenance: the session.** Two captures, same machine, roughly an hour apart:

```text
pcie_train.csv   idle_rows=   72 sum_tx=0.04538 GB/s (=46.47 MB/s) idle_power=283W
pcie_infer.csv   idle_rows= 5295 sum_tx=0.00643 GB/s (= 6.59 MB/s) idle_power=157W
```

**A factor of 7 on tx, and 126 W on power, between two idle measurements of the same
hardware.** A floor carried over from another run is an assumption wearing a
measurement's clothes.

The control-group use matters more than the number: a single-card configuration has
no peer traffic, so its tx column *must* sit on the floor. If it does not, the column
is not measuring what you think and the analysis is void. On the session's data the
1-card configuration reported `per_gpu_tx = 0.0079 GB/s` against a floor of
0.0064 GB/s — the control passed, and only then was the rest worth reading.

---

## TL007 — a fixed busy threshold

**Fires as `warn`** when a threshold is supplied on the command line, and as
**`unknown`** when the file's own distribution will not support one.

**Provenance: the session and the field reports.** The analysis note bins by
`sm >= 90`. Then:

| workload, 8-card nodes | plateau `sm` |
|---|---|
| detection training (YOLOX) | 48.4 % |
| inference decode (Qwen3-VL) | 69.9 % |
| segmentation training (KNet-Swin-L) | 94.1 % |

`sm >= 90` selects zero rows on two of the three. The session's own script carries
the scar:

```text
=== 3. INFERENCE: auto-binned by SM>=20 (decode is memory-bound, SM never reaches 90) ===
```

A second constant, chosen the same way, for the same reason.

So the threshold comes from the file, by Otsu's method — sweep every candidate cut,
keep the one that maximises between-class variance. No parameters and no libraries.

**The refusal is the more important half.** Otsu returns a cut for any input,
including one with nothing to cut. So the split is accepted only when *separability*
— the between-class share of total variance — clears a floor:

| distribution | separability |
|---|---|
| uniform band, nothing to cut | **0.77** |
| two plateaux with idle between | **0.99 – 1.00** |

The floor is **0.90**, and it is measured rather than picked: it sits in the gap
between those two figures, both of which are pinned by tests. An earlier value of
0.70 sat *below* the uniform case and confidently cut an unsegmentable file at 49.17.

---

## TL008 — reading `utilization.gpu` as an intensity

**Fires** always, as `info`, whenever the column is present.

**Provenance: the device run.** 29.6 % on a workload delivering 0.047 TFLOPS, 27.9 %
on one delivering 45.579 — 978× apart. Full write-up in
[what-sm-means.md](what-sm-means.md), including the prediction this experiment
falsified.

---

## TL009 — one segmentation is not a check

**Fires as `violation`** when two independent segmentations label the same samples
differently.

**Provenance: the analysis note**, whose last step is titled
「换方法独立复核（**交报告前必做**）」 — recut by a second method, and believe the
numbers only if both agree. It is the step that always gets skipped when the first
answer looks reasonable, which is exactly why it is automated here.

- **Method A** reads per-card values: which cards are above the file's threshold. The
  label is the *set*, `{0,1,5,6}`.
- **Method B** reads only the pooled mean, one number per sample, and divides by the
  busy plateau to estimate *how many* cards were working. It never looks at an
  individual card.

**Provenance for the current implementation: this repo.** The first version divided
by the *mean* of the busy readings, and a test written to force a disagreement could
not. With `plateau = mean(busy) = sum/k` and `pooled = sum/N`, the estimate
`pooled/plateau × N` collapses to `k` **identically, for every input** — method B was
returning method A's answer by arithmetic. It would have reported "two independent
segmentations agree" forever.

The plateau is now the **median** of the busy readings, which does not absorb a card
sitting at part load. `tests/test_crosscheck.py` keeps both computations side by side
and asserts the mean-based one still lands exactly on the busy-set size, so the
identity cannot come back unnoticed.

---

## TL010 — polling faster than the driver's window

**Fires as `warn`** when more than 25 % of the active-phase card-samples repeat the
previous one.

**Provenance: the device run.** NVML documents the utilisation sample window as
"between 1 second and 1/6 second". Polling at 250 ms:

```text
oversampling: [info] 252 rows, 64 of them differ from the row before (75% are repeats).
```

Those repeats are not independent observations. Counting them as such shrinks every
error bar by a factor that has nothing to do with the measurement.

**Provenance for the scoping: this repo.** The first version counted repeats across
the whole file and reported oversampling on a file sampled at 1.376 s — four times
*slower* than the fastest documented window. An idle card reports exactly 0.0 every
time, so a file that is a third idle scores a third "repeats" no matter how it was
sampled. The count now covers active phases only: the same scope error the rest of the
library exists to catch, made inside the library.
