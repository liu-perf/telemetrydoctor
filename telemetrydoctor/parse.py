"""Read the three telemetry formats that actually turn up, into one shape.

    pynvml wide     timestamp,gpu0_tx_gbs,gpu0_rx_gbs,gpu0_sm_pct,...  (41 cols on 8 cards)
    query-gpu       nvidia-smi --query-gpu=... --format=csv
    dmon            nvidia-smi dmon   (fixed width, two '#' header lines)

All three end up as a `Series` of `Reading(index, t, gpu, values)`.

Two decisions here are load-bearing.

**A missing header is not a reason to guess silently.** The field's wide CSV is
read by position in a one-line awk script -- `tx` at column `2+i*5`, `rx` at
`3+i*5` -- and that works right up until somebody adds a column. When this parser
cannot name the columns from a header, it falls back to that documented layout and
records `layout_basis = "assumed_positional"`, and every downstream report carries
the word. Guessing is allowed; guessing quietly is not.

**A file with no timestamps has no measured interval.** `dmon` without `-o DT`
gives you rows and nothing else. The parser does not fill in the nominal interval
on the user's behalf; `series.interval_s` is None and the tools that need it say
`unknown` instead of assuming. This is the whole subject of the library applied to
the library's own input handling.
"""
import re

from .contracts import canonical

# The field's own layout, from the export this reader was written against: timestamp,
# then five columns per card in this order. Used only when a header cannot name the columns.
FIELD_POSITIONAL = ("pcie_tx_gbs", "pcie_rx_gbs", "sm_pct", "mem_pct", "power_w")

# gpu0_tx_gbs / gpu0.tx / 0_tx / tx_gbs_0 / tx0 / gpu_0_sm
_GPU_PREFIX = re.compile(r"^(?:gpu[_.\-]?)?(\d+)[_.\-](.+)$")
_GPU_SUFFIX = re.compile(r"^(.+?)[_.\-](?:gpu[_.\-]?)?(\d+)$")

_TS_PATTERNS = (
    "%Y-%m-%dT%H:%M:%S",        # 2026-07-30T10:54:13.877
    "%Y/%m/%d %H:%M:%S",        # 2026/08/13 15:03:56.707  (nvidia-smi)
    "%Y-%m-%d %H:%M:%S",
)


class Reading:
    """One card, one sample."""

    __slots__ = ("index", "t", "gpu", "values")

    def __init__(self, index, t, gpu, values):
        self.index = index
        self.t = t
        self.gpu = gpu
        self.values = values

    def __repr__(self):
        return "Reading(index={}, gpu={}, {})".format(self.index, self.gpu, self.values)


class Series:
    """Everything read out of one file, plus how confident the reading was."""

    __slots__ = ("readings", "columns", "gpus", "source_format", "layout_basis",
                 "nominal_interval_s", "path", "unnamed_columns", "comments")

    def __init__(self, readings, columns, gpus, source_format, layout_basis,
                 nominal_interval_s=None, path=None, unnamed_columns=(), comments=()):
        self.readings = readings
        self.columns = columns
        self.gpus = gpus
        self.source_format = source_format
        self.layout_basis = layout_basis
        self.nominal_interval_s = nominal_interval_s
        self.path = path
        self.unnamed_columns = tuple(unnamed_columns)
        self.comments = tuple(comments)

    @property
    def declares_synthetic(self):
        """True when the file says in-band that it is not from a real machine.

        Same rule the other tools' fixtures follow: generated data has to say so
        inside the data, not in a filename that gets lost the first time somebody
        copies the numbers into a slide.
        """
        return any("EXAMPLE" in c.upper() or "SYNTHETIC" in c.upper()
                   for c in self.comments)

    # ---------------------------------------------------------------- accessors
    @property
    def n_samples(self):
        return len({r.index for r in self.readings})

    @property
    def has_timestamps(self):
        return any(r.t is not None for r in self.readings)

    def times(self):
        """Sorted list of one timestamp per sample index, or [] if untimed."""
        by_index = {}
        for r in self.readings:
            if r.t is not None and r.index not in by_index:
                by_index[r.index] = r.t
        return [by_index[k] for k in sorted(by_index)]

    def span_s(self):
        ts = self.times()
        return (ts[-1] - ts[0]) if len(ts) >= 2 else None

    def column(self, name, gpu=None):
        """Values of one canonical column, in sample order. `gpu=None` = every card."""
        out = []
        for r in sorted(self.readings, key=lambda r: (r.index, r.gpu)):
            if gpu is not None and r.gpu != gpu:
                continue
            v = r.values.get(name)
            if v is not None:
                out.append(v)
        return out

    def per_sample(self, name):
        """-> [(index, {gpu: value})] in sample order, for the one column."""
        grouped = {}
        for r in self.readings:
            v = r.values.get(name)
            if v is not None:
                grouped.setdefault(r.index, {})[r.gpu] = v
        return [(k, grouped[k]) for k in sorted(grouped)]


# ------------------------------------------------------------------- timestamps
def parse_timestamp(text):
    """-> epoch seconds, or None. Tolerant of the fractional part and of 'Z'/offsets."""
    import time as _time

    s = text.strip().rstrip("Z")
    # drop a trailing +00:00 / -07:00 offset; these files are single-machine logs and
    # mixing offsets in one file is itself something the audit reports elsewhere
    s = re.sub(r"[+-]\d{2}:?\d{2}$", "", s).strip()
    whole, _, frac = s.partition(".")
    for pat in _TS_PATTERNS:
        try:
            base = _time.mktime(_time.strptime(whole, pat))
        except ValueError:
            continue
        try:
            return base + (float("0." + frac) if frac else 0.0)
        except ValueError:
            return base
    return None


# ------------------------------------------------------------------ header logic
def _split_header_cell(cell):
    """-> (gpu_index or None, canonical column name or None).

    `nvidia-smi --format=csv` decorates names with units: `power.draw [W]`.
    """
    cell = cell.strip()
    cell = re.sub(r"\s*\[[^\]]*\]\s*$", "", cell).strip()
    if not cell:
        return None, None
    low = cell.lower()
    if low in ("timestamp", "time", "#timestamp", "date"):
        return None, "timestamp"

    direct = canonical(cell)
    if direct:
        return None, direct

    for rx, order in ((_GPU_PREFIX, "prefix"), (_GPU_SUFFIX, "suffix")):
        m = rx.match(low)
        if not m:
            continue
        idx, rest = (m.group(1), m.group(2)) if order == "prefix" else (m.group(2), m.group(1))
        canon = canonical(rest)
        if canon:
            return int(idx), canon
    return None, None


def _looks_like_header(cells):
    """A header is a row with no cell that parses as a number OR as a timestamp.

    The timestamp half is not decoration. `nvidia-smi --query-gpu` rows begin with
    `2026/08/13 15:03:56.707`, which is not a number, so a rule that only asked "is
    any cell non-numeric" classified the first *data* row of every query-format file
    as a header and dropped it. One row lost out of sixty is exactly the kind of
    quiet loss that never shows up in a mean.
    """
    for c in cells:
        c = c.strip()
        if not c:
            continue
        try:
            float(c)
            return False
        except ValueError:
            pass
        if parse_timestamp(c) is not None:
            return False
    return True


# ---------------------------------------------------------------------- parsers
def parse_wide_csv(text, path=None, nominal_interval_s=None):
    """The field's pynvml layout: one row per sample, five columns per card."""
    all_lines = [ln for ln in text.splitlines() if ln.strip()]
    comments = [ln.lstrip().lstrip("#").strip() for ln in all_lines
                if ln.lstrip().startswith("#")]
    lines = [ln for ln in all_lines if not ln.lstrip().startswith("#")]
    if not lines:
        raise ValueError("empty file")

    first = list(lines[0].split(","))
    have_header = _looks_like_header(first)
    body = lines[1:] if have_header else lines

    layout_basis = "header"
    unnamed = []
    mapping = []            # per data column: (gpu or None, canonical or None)
    if have_header:
        mapping = [_split_header_cell(c) for c in first]
        named = [m for m in mapping if m[1] and m[1] != "timestamp"]
        unnamed = [first[i].strip() for i, m in enumerate(mapping)
                   if m[1] is None and first[i].strip()]
        if not named:
            have_header, mapping = True, []          # header present but unusable

    if not mapping or not any(m[1] and m[1] != "timestamp" for m in mapping):
        # positional fallback against the documented field layout
        ncols = len(body[0].split(",")) if body else len(first)
        per_card = len(FIELD_POSITIONAL)
        if ncols < 1 + per_card or (ncols - 1) % per_card:
            raise ValueError(
                "cannot name the columns from a header and the width ({}) is not "
                "1 + {}*cards, so the positional fallback does not apply".format(
                    ncols, per_card))
        mapping = [(None, "timestamp")]
        for gpu in range((ncols - 1) // per_card):
            for metric in FIELD_POSITIONAL:
                mapping.append((gpu, metric))
        layout_basis = "assumed_positional"

    readings, gpus, columns = [], set(), set()
    for index, line in enumerate(body):
        cells = line.split(",")
        t = None
        per_gpu = {}
        for i, (gpu, metric) in enumerate(mapping):
            if i >= len(cells) or metric is None:
                continue
            raw = cells[i].strip()
            if metric == "timestamp":
                t = parse_timestamp(raw)
                continue
            if raw in ("", "-", "N/A", "[N/A]", "[Not Supported]"):
                continue
            try:
                val = float(raw)
            except ValueError:
                continue
            g = 0 if gpu is None else gpu
            per_gpu.setdefault(g, {})[metric] = val
            gpus.add(g)
            columns.add(metric)
        for g, values in per_gpu.items():
            readings.append(Reading(index, t, g, values))

    return Series(readings, sorted(columns), sorted(gpus), "wide_csv", layout_basis,
                  nominal_interval_s, path, unnamed, comments)


def parse_query_csv(text, columns, path=None, nominal_interval_s=None):
    """`nvidia-smi --query-gpu` output, long form: one row per card per sample.

    `columns` is the query spec, because `--format=csv,noheader` deletes the only
    record of what the columns were. Passing the wrong spec is a real risk, so the
    parser refuses a spec whose width does not match the file.
    """
    spec = [_split_header_cell(c)[1] for c in columns.split(",")]
    comments = [ln.lstrip().lstrip("#").strip() for ln in text.splitlines()
                if ln.lstrip().startswith("#")]
    lines = [ln for ln in text.splitlines()
             if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        raise ValueError("empty file")
    if _looks_like_header(lines[0].split(",")) and len(lines) > 1:
        lines = lines[1:]

    width = len(lines[0].split(","))
    if width != len(spec):
        raise ValueError(
            "--columns lists {} names but the file has {} fields per row; a mismatched "
            "spec would silently relabel every number".format(len(spec), width))

    if "index" in [c.strip().lower() for c in columns.split(",")]:
        gpu_at = [c.strip().lower() for c in columns.split(",")].index("index")
    else:
        gpu_at = None

    readings, gpus, present = [], set(), set()
    index, seen_gpus = -1, set()
    for line in lines:
        cells = [c.strip() for c in line.split(",")]
        gpu = int(cells[gpu_at]) if gpu_at is not None else 0
        if gpu in seen_gpus:
            seen_gpus = set()
        if not seen_gpus:
            index += 1
        seen_gpus.add(gpu)

        t, values = None, {}
        for name, raw in zip(spec, cells):
            if name == "timestamp":
                t = parse_timestamp(raw)
                continue
            if name is None or raw in ("", "-", "N/A", "[N/A]", "[Not Supported]"):
                continue
            try:
                values[name] = float(raw.split()[0])
            except (ValueError, IndexError):
                continue
        readings.append(Reading(index, t, gpu, values))
        gpus.add(gpu)
        present.update(values)

    return Series(readings, sorted(present), sorted(gpus), "query_csv", "header",
                  nominal_interval_s, path, (), comments)


def parse_dmon(text, path=None, nominal_interval_s=None):
    """`nvidia-smi dmon` fixed-width output.

    Two '#' lines: names then units. Without `-o DT` there are no timestamps at
    all, which is exactly why the field README calls this format easier to get the
    accounting wrong with -- there is nothing in the file to measure the interval
    against, so the nominal value is the only thing available, and it is the thing
    that is wrong.
    """
    names = None
    comments = []
    readings, gpus, present = [], set(), set()
    index, seen_gpus = -1, set()

    for line in text.splitlines():
        if not line.strip():
            continue
        if line.lstrip().startswith("#"):
            stripped = line.lstrip().lstrip("#").strip()
            cells = stripped.split()
            # Two '#' lines come past: names, then units. The names line is the one
            # with at least one recognisable metric in it. Anything else is prose --
            # a self-declaration, a capture note -- and is kept as a comment.
            if names is None and any(canonical(c) for c in cells):
                names = cells
            else:
                comments.append(stripped)
            continue
        if names is None:
            continue
        cells = line.split()
        if len(cells) < 2:
            continue

        # dmon's leading column is the card index, headed '# gpu' or 'Idx'. It is not
        # a metric, so it is consumed here and never offered to `canonical()`.
        offset = 0
        gpu = 0
        if names[0].strip().lower() in ("gpu", "idx"):
            offset = 1
            try:
                gpu = int(cells[0])
            except ValueError:
                gpu = 0

        t, values = None, {}
        for name, raw in zip(names[offset:], cells[offset:]):
            if raw in ("-", "", "N/A"):
                continue
            if name.strip().lower() in ("date", "time"):
                continue
            canon = canonical(name)
            if canon is None:
                continue
            try:
                values[canon] = float(raw)
            except ValueError:
                continue

        if gpu in seen_gpus:
            seen_gpus = set()
        if not seen_gpus:
            index += 1
        seen_gpus.add(gpu)

        readings.append(Reading(index, t, gpu, values))
        gpus.add(gpu)
        present.update(values)

    if not readings:
        raise ValueError("no dmon data rows found (is the '# gpu pwr ...' header there?)")
    return Series(readings, sorted(present), sorted(gpus), "dmon", "header",
                  nominal_interval_s, path, (), comments)


def sniff(text):
    """-> 'dmon' | 'wide_csv' | 'query_csv'.

    Decided on the first *data* row, not on the presence of a '#' line: a CSV whose
    first line is a `# EXAMPLE -- synthetic` self-declaration is still a CSV, and an
    earlier version of this function called it dmon and then failed to parse it.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    data = [ln for ln in lines if not ln.lstrip().startswith("#")]
    if not data:
        raise ValueError("file has no data rows, only comments")
    if "," not in data[0]:
        return "dmon"
    first = data[0].split(",")
    if _looks_like_header(first):
        # A header that indexes its columns by card is the wide layout, at any width:
        # `timestamp,gpu0_sm_pct` is a one-card wide CSV, and deciding by column count
        # alone called that one query-format and then failed on it.
        if any(g is not None for g, _ in (_split_header_cell(c) for c in first)):
            return "wide_csv"
        return "query_csv"
    # No header at all: the only signal left is the width, against the documented
    # 1 + 5-per-card layout.
    per_card = len(FIELD_POSITIONAL)
    return ("wide_csv" if len(first) >= 1 + per_card and (len(first) - 1) % per_card == 0
            else "query_csv")


def load(path, fmt=None, columns=None, nominal_interval_s=None):
    """Read a telemetry file. `fmt=None` sniffs."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    fmt = fmt or sniff(text)
    if fmt == "dmon":
        return parse_dmon(text, path, nominal_interval_s)
    if fmt == "query_csv":
        if not columns:
            raise ValueError(
                "query-gpu output needs --columns: '--format=csv,noheader' removes the "
                "only record of which column is which, and relabelling them by guess is "
                "how a power column becomes a utilisation column")
        return parse_query_csv(text, columns, path, nominal_interval_s)
    return parse_wide_csv(text, path, nominal_interval_s)
