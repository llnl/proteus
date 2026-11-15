#!/usr/bin/env python3
"""Ground-truth tracking for lambda kernel-arg specialization decisions.

The ProteusPass trace (PROTEUS_DEBUG_OUTPUT=1, .proteus-logs/proteus-pass.*.log)
records one line per analysis outcome:

  "Found termination case from function <kernel>"            -> SPECIALIZE
  "[WARNING]: Kernel arg analysis failed for functor ..."    -> DECLINE

These lines are identical whether the log came from a real ROCm build (e.g. a
ctest run with PROTEUS_DEBUG_OUTPUT=1) or from the pseudo-GPU device pipeline,
so one ground-truth file serves both.

Modes:
  capture  parse per-test log files into a ground-truth JSON
  check    parse per-test log files and diff against a ground-truth JSON
  table    render the JSON as a Markdown table

Log layout for capture/check: a directory containing <test>.log files, one per
test (concatenating that test's proteus-pass logs is fine; duplicate decisions
from repeated pass runs are deduplicated).
"""

import argparse
import datetime
import json
import re
import subprocess
import sys
from pathlib import Path

SPECIALIZE_RE = re.compile(r"Found termination case from function (\S+)")
DECLINE_RE = re.compile(
    r"\[WARNING\]: Kernel arg analysis failed for functor beginning at\s+"
    r"call \S+ @(\S+?)\(")

# Itanium length-prefixed identifiers, e.g. 26kernelPredecessorOverwrite.
LENGTH_PREFIXED_RE = re.compile(r"(\d+)([A-Za-z_])")
PLUMBING_NAMES = {"proteus", "detail", "LambdaFunctorWrapper"}


def extract_identifiers(mangled: str):
    """Yield the length-prefixed identifiers embedded in a mangled name."""
    pos = 0
    while True:
        match = LENGTH_PREFIXED_RE.search(mangled, pos)
        if not match:
            return
        length = int(match.group(1))
        start = match.end(1)
        ident = mangled[start:start + length]
        if len(ident) == length and re.fullmatch(r"[A-Za-z0-9_]+", ident):
            yield ident
            pos = start + length
        else:
            pos = match.end(1)


def friendly_name(mangled: str) -> str:
    """Stable human-readable key for a kernel or wrapper-call mangled name.

    The first non-plumbing length-prefixed identifier is the symbol's own
    name (the kernel for specializations; for wrapper-operator declines the
    plumbing filter skips to the registered-lambda factory). Later
    identifiers are template-argument noise and Itanium substitution refs.
    """
    for ident in extract_identifiers(mangled):
        if ident not in PLUMBING_NAMES and len(ident) >= 2:
            return ident
    return mangled


def parse_log(path: Path):
    text = path.read_text(errors="replace")
    specialize = sorted({friendly_name(M) for M in SPECIALIZE_RE.findall(text)})
    decline = sorted({friendly_name(M) for M in DECLINE_RE.findall(text)})
    return {"specialize": specialize, "decline": decline}


def collect(logs_dir: Path):
    tests = {}
    for log in sorted(logs_dir.glob("*.log")):
        tests[log.stem] = parse_log(log)
    if not tests:
        sys.exit(f"error: no .log files under {logs_dir}")
    return tests


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def cmd_capture(args):
    truth = {
        "meta": {
            "generated": datetime.date.today().isoformat(),
            "commit": git_commit(),
            "pipeline": args.pipeline,
            "note": args.note,
        },
        "tests": collect(Path(args.logs)),
    }
    Path(args.truth).write_text(json.dumps(truth, indent=2) + "\n")
    total = sum(len(T["specialize"]) + len(T["decline"])
                for T in truth["tests"].values())
    print(f"captured {len(truth['tests'])} tests, {total} decisions "
          f"-> {args.truth}")


def cmd_check(args):
    truth = json.loads(Path(args.truth).read_text())
    actual = collect(Path(args.logs))
    failures = []
    for test, expected in truth["tests"].items():
        if test not in actual:
            failures.append(f"{test}: MISSING (no log captured)")
            continue
        for kind in ("specialize", "decline"):
            want, got = set(expected[kind]), set(actual[test][kind])
            for name in sorted(want - got):
                failures.append(f"{test}: {name} expected {kind.upper()}, "
                                f"decision absent")
            for name in sorted(got - want):
                failures.append(f"{test}: {name} unexpectedly {kind.upper()}")
    for test in sorted(set(actual) - set(truth["tests"])):
        print(f"note: {test} not in ground truth (new test?); "
              f"re-capture to adopt it")
    if failures:
        print(f"LAMBDA-SPEC-CI: FAIL ({len(failures)} drift(s) vs "
              f"{args.truth} @ {truth['meta'].get('commit')})")
        for F in failures:
            print(f"  {F}")
        sys.exit(1)
    print(f"LAMBDA-SPEC-CI: OK ({len(truth['tests'])} tests match "
          f"{args.truth} @ {truth['meta'].get('commit')})")


def cmd_table(args):
    truth = json.loads(Path(args.truth).read_text())
    print("| Test | Specializes | Declines |")
    print("| --- | --- | --- |")
    for test, T in sorted(truth["tests"].items()):
        spec = ", ".join(T["specialize"]) or "—"
        decl = ", ".join(T["decline"]) or "—"
        print(f"| {test} | {spec} | {decl} |")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    cap = sub.add_parser("capture")
    cap.add_argument("--logs", required=True)
    cap.add_argument("--truth", required=True)
    cap.add_argument("--pipeline", default="unspecified",
                     help="e.g. rocm-6.4.3-llvm19 or pseudo-gpu-llvm19")
    cap.add_argument("--note", default="")
    cap.set_defaults(func=cmd_capture)

    chk = sub.add_parser("check")
    chk.add_argument("--logs", required=True)
    chk.add_argument("--truth", required=True)
    chk.set_defaults(func=cmd_check)

    tab = sub.add_parser("table")
    tab.add_argument("--truth", required=True)
    tab.set_defaults(func=cmd_table)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
