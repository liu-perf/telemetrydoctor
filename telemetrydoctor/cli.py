"""Command line entry point.

    telemetrydoctor audit      run.csv
    telemetrydoctor phases     run.csv
    telemetrydoctor aggregate  run.csv
    telemetrydoctor contracts

Output is one finding per line in the same shape the other six tools use:

    {location}: [{status}] {message}

`--json` emits the same content as a list of objects. `--fail-on` turns a status
into a non-zero exit for CI; like the other tools it is a threshold and not an
equality, so `--fail-on warn` also trips on `violation`.

Exit codes: 0 clean or gate not reached, 1 gate tripped, 2 could not read the input.
"""
import argparse
import json
import sys

from . import aggregate as _agg
from . import parse as _parse
from . import rules as _rules
from . import segment as _seg

FAIL_ORDER = ("ok", "info", "unknown", "warn", "violation")


def _fail_reached(statuses, gate):
    if not gate:
        return False
    if gate == "any":
        gate = "info"
    floor = _rules.SEVERITY.get(gate)
    if floor is None:
        return False
    return any(_rules.SEVERITY.get(s, 0) >= floor for s in statuses)


def _emit(findings, as_json):
    if as_json:
        print(json.dumps([f.as_dict() for f in findings], indent=2, ensure_ascii=False))
    else:
        for f in findings:
            print(f)


def _load(args):
    return _parse.load(args.file, fmt=None if args.format == "auto" else args.format,
                       columns=args.columns, nominal_interval_s=args.interval)


def _series_header(series, interval):
    src = "{} rows x {} card(s), {} format".format(
        series.n_samples, len(series.gpus), series.source_format)
    if series.layout_basis == "assumed_positional":
        src += ("; columns named by POSITION against the documented 5-per-card layout "
                "because the header could not name them -- if a column was ever added "
                "to this sampler, every label below is shifted")
    if series.unnamed_columns:
        src += "; unrecognised columns ignored: " + ", ".join(series.unnamed_columns)
    span = series.span_s()
    if span:
        src += "; span {:.0f} s".format(span)
    if interval.basis == "measured":
        src += "; {:.3f} s/row measured".format(interval.measured_s)
    return _rules.Finding("input", "source", _rules.INFO, src)


# ------------------------------------------------------------------------ audit
def cmd_audit(args):
    series = _load(args)
    findings, ctx = _rules.audit(series, args.column, args.threshold,
                                 only=args.rule or None)
    out = [_series_header(series, ctx.interval)] + findings
    _emit(out, args.json)
    return 1 if _fail_reached([f.status for f in findings], args.fail_on) else 0


# ----------------------------------------------------------------------- phases
def cmd_phases(args):
    series = _load(args)
    ctx = _rules.Context(series, args.column, args.threshold)
    out = [_series_header(series, ctx.interval), _rules.tl007_threshold(ctx)]
    if not ctx.segments:
        out.append(_rules.Finding(
            "TL007", "phases", _rules.UNKNOWN,
            "no phases emitted; with no usable busy/idle split there is no honest way "
            "to say where one configuration ends and the next begins"))
        _emit(out, args.json)
        return 1 if _fail_reached([f.status for f in out], args.fail_on) else 0

    for s in ctx.segments:
        dur = s.duration_s()
        out.append(_rules.Finding(
            "phase", "{}[{}-{}]".format(s.kind, s.i0, s.i1), _rules.INFO,
            "{} card(s) busy {}, {} samples{}".format(
                len(s.busy_gpus), s.label, s.n_samples,
                ", {:.1f} s".format(dur) if dur else "")))
    out.append(_rules.tl009_crosscheck(ctx))
    _emit(out, args.json)
    return 1 if _fail_reached([f.status for f in out], args.fail_on) else 0


# -------------------------------------------------------------------- aggregate
def cmd_aggregate(args):
    series = _load(args)
    ctx = _rules.Context(series, args.column, args.threshold)
    interval_s = ctx.interval.effective_s
    rows = []

    if not ctx.segments:
        rows.append(_rules.Finding(
            "aggregate", "all", _rules.UNKNOWN,
            "refusing to aggregate: no usable phase split, so any mean would span the "
            "load, the idle gaps and the work at once"))
        _emit([_series_header(series, ctx.interval)] + rows, args.json)
        return 1 if _fail_reached([r.status for r in rows], args.fail_on) else 0

    cross = _rules.tl009_crosscheck(ctx)
    header = [_series_header(series, ctx.interval), cross]
    if cross.status == _rules.VIOLATION and not args.force:
        header.append(_rules.Finding(
            "aggregate", "all", _rules.UNKNOWN,
            "refusing to aggregate while the two segmentations disagree; pass --force "
            "to print the numbers anyway, and say in the report that you did"))
        _emit(header, args.json)
        return 1 if _fail_reached([f.status for f in header], args.fail_on) else 0

    for label, segs in sorted(_seg.configurations(ctx.segments).items()):
        for s in segs:
            for v in _agg.summarise(series, s, interval_s):
                status = {"ok": _rules.OK, "forbidden": _rules.VIOLATION,
                          "meaningless": _rules.WARN,
                          "unknown": _rules.UNKNOWN}.get(v.status, _rules.INFO)
                shown = ("{:.5f} {}".format(v.value, v.unit) if v.value is not None
                         else "refused")
                msg = "{}{}".format(shown, "; " + v.note if v.note else "")
                rows.append(_rules.Finding(
                    "aggregate", "{}.{}.{}.{}".format(label, v.column, v.op, v.scope),
                    status, msg))
    _emit(header + rows, args.json)
    return 1 if _fail_reached([r.status for r in rows], args.fail_on) else 0


# -------------------------------------------------------------------- contracts
def cmd_contracts(args):
    _emit(_rules.describe_contracts(), args.json)
    return 0


def build_parser():
    p = argparse.ArgumentParser(
        prog="telemetrydoctor",
        description="Audit GPU telemetry for aggregation mistakes: phase mixing, "
                    "forbidden integration, percentage scope, and thresholds that do "
                    "not survive a change of workload.")
    sub = p.add_subparsers(dest="cmd")

    def common(sp, needs_file=True):
        if needs_file:
            sp.add_argument("file", help="telemetry CSV or dmon capture")
            sp.add_argument("--format", default="auto",
                            choices=("auto", "wide_csv", "query_csv", "dmon"))
            sp.add_argument("--columns", default=None,
                            help="query-gpu spec, e.g. "
                                 "'timestamp,index,utilization.gpu,power.draw'")
            sp.add_argument("--interval", type=float, default=None, metavar="SECONDS",
                            help="nominal sampling interval, for files with no "
                                 "timestamps; it is never used in place of a measured "
                                 "one when timestamps exist")
            sp.add_argument("--column", default="sm_pct",
                            help="column used to decide busy/idle (default sm_pct)")
            sp.add_argument("--threshold", type=float, default=None,
                            help="force the busy threshold instead of computing it "
                                 "from the file; reported as a warning because a "
                                 "constant does not survive a change of workload")
        sp.add_argument("--json", action="store_true")
        sp.add_argument("--fail-on", default=None,
                        choices=("info", "unknown", "warn", "violation", "any"),
                        help="exit 1 when any finding reaches this status or worse")

    sp = sub.add_parser("audit", help="run the ten rules against a telemetry file")
    common(sp)
    sp.add_argument("--rule", action="append", metavar="TLNNN",
                    help="run only this rule; repeatable")
    sp.set_defaults(func=cmd_audit)

    sp = sub.add_parser("phases", help="show the phases cut out of the file")
    common(sp)
    sp.set_defaults(func=cmd_phases)

    sp = sub.add_parser("aggregate", help="per-configuration aggregates, refusals included")
    common(sp)
    sp.add_argument("--force", action="store_true",
                    help="aggregate even when the two segmentations disagree")
    sp.set_defaults(func=cmd_aggregate)

    sp = sub.add_parser("contracts", help="what arithmetic each column supports")
    common(sp, needs_file=False)
    sp.set_defaults(func=cmd_contracts)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not getattr(args, "func", None):
        build_parser().print_help()
        return 2
    try:
        return args.func(args)
    except (OSError, ValueError) as exc:
        print("input: [error] {}".format(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
