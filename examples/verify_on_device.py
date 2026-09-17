"""Check telemetrydoctor's claims against a real GPU and a real nvidia-smi.

    python examples/verify_on_device.py

Needs a CUDA card, torch, and nvidia-smi on PATH. Nothing else in the repo does --
the library itself is pure standard library and parses files. This script exists
because the library's central claim is a claim about what a *number means*, and
that cannot be settled by reading a CSV somebody else produced.

Four experiments, in order of how much they change what you do on Monday:

  1. utilization.gpu is a time measure, not an intensity. Run a 64x64 matmul
     back to back and a 4096x4096 matmul at a quarter duty; both land on ~28%,
     and they are ~1000x apart in delivered FLOPS. So "sm=48%" and "sm=94%" on
     two real training jobs do not rank those jobs -- which is exactly the field
     observation this repo was built around.

     Note what this experiment does NOT show, because the first version of it
     claimed the wrong thing: it is not the case that intensity is invisible
     while *duty* is faithfully reported. The light arm was submitting with no
     host-side gap at all -- nominal duty 100% -- and still read 29.6%. The
     column is honest about the device; it was my host-side timer that was
     wrong. See finding 1b (`launch_bound`).
  2. Polling faster than the driver's internal sample window buys rows, not
     information. NVML documents the window as "between 1 second and 1/6
     second"; at 250 ms we should see runs of identical values.
  3. The real sampling interval is not the nominal one. Here it is one card and
     nvidia-smi doing its own timing, so the overshoot is small -- the point is
     that it is nonzero and has to be measured, because on 8 cards through
     pynvml the same nominal 1.0 s came out at 1.376 s.
  4. Idle is not zero. The noise floor has to be measured per file, not assumed.

Output follows the same one-line-per-finding format as the other six tools:
`{location}: [{status}] {message}`.
"""
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

POLL_MS = 250
FIELD_NOMINAL_S = 1.0        # what the field script asked for
FIELD_MEASURED_S = 1.376     # what it actually got, on 8 cards, via pynvml
QUERY = "timestamp,utilization.gpu,power.draw,clocks.sm"


def say(location, status, message):
    print("{}: [{}] {}".format(location, status, message))


def sampler_start(path):
    """Start nvidia-smi polling into `path`. Returns the Popen handle."""
    fh = open(path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=" + QUERY,
         "--format=csv,noheader,nounits", "-lms", str(POLL_MS)],
        stdout=fh, stderr=subprocess.DEVNULL)
    proc._fh = fh
    return proc


def sampler_stop(proc):
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    proc._fh.close()


def read_samples(path):
    """-> [(epoch_seconds, util_pct, power_w, sm_clock_mhz)]"""
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            try:
                # nvidia-smi stamps local wall clock: 2026/08/13 15:03:56.707
                whole, _, frac = parts[0].partition(".")
                t = time.mktime(time.strptime(whole, "%Y/%m/%d %H:%M:%S"))
                t += float("0." + frac) if frac else 0.0
                out.append((t, float(parts[1]), float(parts[2]), float(parts[3])))
            except (ValueError, OverflowError):
                continue
    return out


def window(samples, t0, t1):
    return [s for s in samples if t0 <= s[0] <= t1]


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


# --------------------------------------------------------------------- workload
def run_duty(torch, n, target_duty, seconds):
    """Alternate GPU bursts with host-side idle gaps at `target_duty`.

    Returns (measured_busy_fraction, achieved_tflops, gpu_seconds).
    Both variants use exactly this function; only `n` changes. That matters --
    if the two arms differed in scheduling as well as in size, the comparison
    would not isolate intensity.
    """
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)

    # calibrate: how long does one matmul actually take, launch included
    for _ in range(3):
        a @ b
    torch.cuda.synchronize()
    t = time.perf_counter()
    reps = 50
    for _ in range(reps):
        a @ b
    torch.cuda.synchronize()
    per_iter = (time.perf_counter() - t) / reps

    period = 0.020                            # 20 ms cycle
    iters = max(1, int(round(period * target_duty / per_iter)))
    idle_gap = period * (1.0 - target_duty)

    flop_per_iter = 2.0 * n * n * n
    gpu_seconds = 0.0
    total_flop = 0.0
    t_start = time.perf_counter()
    t_end = t_start + seconds
    while time.perf_counter() < t_end:
        t_burst = time.perf_counter()
        for _ in range(iters):
            a @ b
        torch.cuda.synchronize()
        busy = time.perf_counter() - t_burst
        gpu_seconds += busy
        total_flop += iters * flop_per_iter
        # spin, not sleep: Windows sleep granularity is coarser than a 20 ms period
        gap_until = time.perf_counter() + idle_gap
        while time.perf_counter() < gap_until:
            pass
    elapsed = time.perf_counter() - t_start

    del a, b
    torch.cuda.empty_cache()
    return gpu_seconds / elapsed, total_flop / gpu_seconds / 1e12, gpu_seconds


def main():
    try:
        import torch
    except ImportError:
        say("device", "skipped", "torch is not installed; this script needs a real card")
        return 0
    if not torch.cuda.is_available():
        say("device", "skipped", "no CUDA device visible")
        return 0

    name = torch.cuda.get_device_name(0)
    total_mib = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
    say("device", "info", "{}, {:,} MiB total, torch {}, CUDA {}".format(
        name, total_mib, torch.__version__, torch.version.cuda))

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_samples.csv")
    proc = sampler_start(out)
    try:
        time.sleep(1.0)

        # ---------------------------------------------------- 4. idle noise floor
        idle_t0 = time.time()
        time.sleep(6.0)
        idle_t1 = time.time()

        # ------------------------------------- 1. duty cycle vs intensity
        marks = []
        for label, n in (("light_64", 64), ("heavy_4096", 4096)):
            for duty in (0.25, 0.50, 1.00):
                time.sleep(1.5)                        # let the reading fall back
                t0 = time.time()
                frac, tflops, gpu_s = run_duty(torch, n, duty, 8.0)
                t1 = time.time()
                marks.append((label, n, duty, frac, tflops, gpu_s, t0, t1))
    finally:
        sampler_stop(proc)

    samples = read_samples(out)
    if len(samples) < 20:
        say("sampler", "error", "only {} rows came back from nvidia-smi".format(len(samples)))
        return 1

    # ------------------------------------------------ 3. nominal vs real interval
    gaps = [b[0] - a[0] for a, b in zip(samples, samples[1:]) if 0 < b[0] - a[0] < 5]
    real_ms = mean(gaps) * 1000.0
    over = (real_ms / POLL_MS - 1.0) * 100.0
    say("interval", "info",
        "nominal {} ms, measured {:.1f} ms over {} gaps (+{:.1f}%); the field's 8-card "
        "pynvml loop asked for {:.1f} s and got {:.3f} s (+{:.1f}%) -- overhead grows "
        "with the number of cards, so the nominal value is never the one to divide by"
        .format(POLL_MS, real_ms, len(gaps), over, FIELD_NOMINAL_S, FIELD_MEASURED_S,
                (FIELD_MEASURED_S / FIELD_NOMINAL_S - 1.0) * 100.0))

    # ----------------------------------------------------- 4. idle noise floor
    idle = window(samples, idle_t0 + 1.0, idle_t1)
    say("noise_floor", "info",
        "{} idle rows: utilization.gpu {:.1f}%, power {:.2f} W, sm clock {:.0f} MHz -- "
        "idle is a measured quantity, not zero, and it is not the same number on "
        "another file from the same machine"
        .format(len(idle), mean([s[1] for s in idle]), mean([s[2] for s in idle]),
                mean([s[3] for s in idle])))

    # ------------------------------------------- 2. oversampling buys no information
    vals = [s[1] for s in samples]
    changes = sum(1 for a, b in zip(vals, vals[1:]) if a != b)
    say("oversampling", "info",
        "{} rows, {} of them differ from the row before ({:.0f}% are repeats). NVML "
        "documents the utilization window as 1 s to 1/6 s, so polling at {} ms "
        "returns the same reading several times -- more rows, not more information"
        .format(len(vals), changes, 100.0 * (1 - changes / max(1, len(vals) - 1)), POLL_MS))

    # ------------------------------------- 1. the central claim
    print()
    # `host_busy` is deliberately named for where it was measured. It is the fraction
    # of wall clock the host spent inside the submit-and-synchronize loop, which is NOT
    # the fraction of time the device had work resident -- see `launch_bound` below.
    print("  {:<12} {:>7} {:>9} {:>10} {:>13} {:>9}".format(
        "workload", "n", "duty_set", "host_busy", "util.gpu_rpt", "TFLOPS"))
    rows = []
    for label, n, duty, frac, tflops, _gpu_s, t0, t1 in marks:
        w = window(samples, t0 + 1.0, t1)
        rpt = mean([s[1] for s in w])
        rows.append((label, n, duty, frac, rpt, tflops, len(w)))
        print("  {:<12} {:>7} {:>8.0f}% {:>9.1f}% {:>12.1f}% {:>9.3f}".format(
            label, n, duty * 100, frac * 100, rpt, tflops))
    print()

    # The comparison that settles it is NOT "same duty setting" -- the host-side busy
    # fraction is not the device-side one, which is the second finding below. It is
    # "same *reported* utilization": find the closest-reading pair across the two arms.
    light_rows = [r for r in rows if r[1] == 64]
    heavy_rows = [r for r in rows if r[1] == 4096]
    best = None
    for lo in light_rows:
        for hi in heavy_rows:
            gap = abs(lo[4] - hi[4])
            if best is None or gap < best[0]:
                best = (gap, lo, hi)
    gap, lo, hi = best
    ratio = hi[5] / lo[5] if lo[5] else float("inf")
    matched = gap <= 5.0 and ratio >= 100.0
    say("same_reading", "ok" if matched else "differs",
        "utilization.gpu reads {:.1f}% on a {}x{} workload delivering {:.3f} TFLOPS and "
        "{:.1f}% on a {}x{} workload delivering {:.3f} TFLOPS -- {:.1f} points apart, "
        "{:.0f}x apart in delivered throughput. A dashboard showing '{:.0f}% GPU "
        "utilisation' is showing the same number for both."
        .format(lo[4], lo[1], lo[1], lo[5], hi[4], hi[1], hi[1], hi[5], gap, ratio,
                (lo[4] + hi[4]) / 2))

    # Second finding, and the reason the naive pairing above fails: for a launch-bound
    # workload the host thinks it is saturated while the device is mostly idle.
    saturated = [r for r in light_rows if r[2] >= 0.999]
    if saturated:
        s = saturated[0]
        say("launch_bound", "info",
            "the {}x{} arm was submitting work back-to-back with no host-side gap "
            "({:.1f}% of wall clock inside the burst loop) and the device still reported "
            "{:.1f}%. The missing {:.1f} points are launch overhead: real device idle "
            "that a host-side timer cannot see. 'My training loop never sleeps' is not "
            "evidence that the GPU is busy."
            .format(s[1], s[1], s[3] * 100, s[4], s[3] * 100 - s[4]))

    say("verdict", "ok" if matched else "differs",
        "utilization.gpu is a *time* measure -- the fraction of the sample window in "
        "which at least one kernel was resident -- and it is blind to what that kernel "
        "did with the hardware. It is not SM occupancy and not tensor-core activity; for "
        "those you need DCGM prof counters or Nsight. Which is why one real 8-card node "
        "showed sm=48.4% on detection training and sm=94.1% on segmentation training: "
        "those two numbers do not rank the two jobs." if matched else
        "no pair landed within 5 points at a 100x throughput gap; on this machine the "
        "claim is not reproduced -- keep the output and look at why rather than "
        "re-running until it agrees")

    try:
        os.remove(out)
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
