"""telemetrydoctor -- what arithmetic a GPU telemetry column supports.

The sixth tool in a series about not fooling yourself with a measurement.
`nodebench` measures a node, `benchdoctor` reads the measuring script, `tracedoctor`
reads a kernel trace, `regressiondoctor` compares two runs, `fitdoctor` says what a
number should have been. This one reads a long telemetry capture and answers a
question none of the others ask: given this column, from this sampler, at this
interval -- what may I compute with it?

    from telemetrydoctor import audit, load

    series = load("pcie_train.csv")
    findings, ctx = audit(series)
    for f in findings:
        print(f)
"""
from .contracts import CONTRACTS, Contract, contract_for
from .interval import measure as measure_interval
from .parse import Reading, Series, load
from .rules import RULES, Context, Finding, audit, describe_contracts, worst_status
from .segment import busy_threshold, configurations, segments

__version__ = "0.1.0"

__all__ = [
    "CONTRACTS", "Context", "Contract", "Finding", "RULES", "Reading", "Series",
    "audit", "busy_threshold", "configurations", "contract_for", "describe_contracts",
    "load", "measure_interval", "segments", "worst_status",
]
