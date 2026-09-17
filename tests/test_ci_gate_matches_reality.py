"""Every `--fail-on` command in ci.yml, run for real, exit code compared.

`test_cli.py` already pins what `--fail-on` does. That was not enough, and the
way it was not enough is worth writing down.

The CI workflow used to end a `run:` block with

    telemetrydoctor audit tests/fixtures/EXAMPLE_unimodal.csv --fail-on violation

which is *supposed* to exit 1 -- TL003 is the 69x PCIe integration finding,
this project's headline result, and that fixture has PCIe columns. But GitHub
runs `run:` blocks under `bash -e`, and that was the block's last command, so
the step took its exit code and CI would have been red on every push. The
command was correct; the missing character was `!`.

No test could see it, because the bug was in YAML and every test lives in
Python. `pytest -q` was green, `ruff` was clean, the bundle's three axes all
passed -- and the badge on the README would have been red from the first push.

So this file crosses the boundary: it reads the workflow file as text, finds
every line that invokes the CLI with `--fail-on`, runs that exact invocation
in-process, and asserts the exit code agrees with how the line is written --
`!`-prefixed means "must exit non-zero", bare means "must exit zero".

It also fails when there is *no* expected-failure line at all, because a gate
that is only ever exercised in its passing direction is decoration. That is
the same rule this project applies to everyone else's telemetry, turned
around: a check that cannot fail is not a check.
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from telemetrydoctor.cli import main  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CI = os.path.join(REPO, ".github", "workflows", "ci.yml")

# `telemetrydoctor <args...>`, possibly `!`-prefixed, possibly continued with
# a trailing backslash. Comment lines are skipped by the leading-`#` guard.
INVOCATION = re.compile(r"^\s*(?P<bang>!\s*)?telemetrydoctor\s+(?P<args>.*)$")


def _commands():
    """(expect_failure, argv) for every --fail-on invocation in the workflow."""
    with open(CI, encoding="utf-8") as fh:
        raw = fh.read()
    # join backslash continuations first, so a wrapped command is one line
    joined = re.sub(r"\\\n\s*", " ", raw)
    out = []
    for line in joined.splitlines():
        if line.lstrip().startswith("#"):
            continue
        m = INVOCATION.match(line)
        if not m or "--fail-on" not in m.group("args"):
            continue
        argv = m.group("args").split()
        # paths in the workflow are relative to the repo root
        argv = [os.path.join(REPO, a) if a.startswith("tests/") else a
                for a in argv]
        out.append((bool(m.group("bang")), argv))
    return out


def test_the_workflow_still_has_gate_commands_to_check():
    cmds = _commands()
    assert cmds, f"no --fail-on invocation found in {CI} -- did the step move?"


def test_at_least_one_gate_command_is_expected_to_fail():
    """A gate only ever run in its passing direction proves nothing."""
    cmds = _commands()
    assert any(expect_fail for expect_fail, _ in cmds), (
        "every --fail-on line in ci.yml expects success. The gate is never "
        "shown to fire, which is the state this project complains about in "
        "other people's dashboards."
    )


def test_every_gate_command_exits_the_way_the_workflow_assumes(capsys):
    for expect_fail, argv in _commands():
        code = main(list(argv))
        capsys.readouterr()
        shown = ("! " if expect_fail else "") + "telemetrydoctor " + " ".join(argv)
        if expect_fail:
            assert code != 0, (
                f"{shown}\n  exits 0, but ci.yml writes it as an expected "
                f"failure (`!`), so `bash -e` will fail that step."
            )
        else:
            assert code == 0, (
                f"{shown}\n  exits {code}, and ci.yml writes it as an "
                f"expected success. Under `bash -e` this makes CI red on "
                f"every push. If the non-zero exit is intended, prefix the "
                f"line with `!` the way the sibling projects do."
            )
