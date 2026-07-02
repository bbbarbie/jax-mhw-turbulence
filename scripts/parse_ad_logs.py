#!/usr/bin/env python3
"""Parse historical direct-AD trajectory sweep logs into a CSV table."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path


LOG_RE = re.compile(r"ad_a(?P<alpha_tag>\d+)_s(?P<steps>\d+)-(?P<job>\d+)\.out$")
BACKEND_RE = re.compile(r"Running with JAX backend on (?P<backend>\S+)")
RES_RE = re.compile(r"Resolution:\s+(?P<nx>\d+)\s+x\s+(?P<ny>\d+)")
DT_RE = re.compile(r"Time Step:\s+dt = (?P<dt>[0-9.eE+-]+)")
ALPHA_RE = re.compile(r"Alpha = (?P<alpha>[0-9.eE+-]+)")
KAPPA_RE = re.compile(r"Kappa = (?P<kappa>[0-9.eE+-]+)")
SOLVER_RE = re.compile(r"Solver:\s+(?P<solver>\w+)\s+\|")
OBJECTIVE_RE = re.compile(r"Objective Value:\s+(?P<objective>[0-9.eE+-]+)")
GRAD_RE = re.compile(
    r"Gradient w\.r\.t \[alpha, kappa\]: \[\s*(?P<ga>[0-9.eE+-]+)\s+(?P<gk>[0-9.eE+-]+)\s*\]"
)
TIME_RE = re.compile(r"Time:\s+(?P<runtime>[0-9.eE+-]+)s")


def parse_log(path: Path) -> dict[str, object]:
    text = path.read_text()
    name_match = LOG_RE.search(path.name)
    if not name_match:
        raise ValueError(f"Unexpected direct-AD log filename: {path}")

    def require(pattern: re.Pattern[str], label: str) -> re.Match[str]:
        match = pattern.search(text)
        if not match:
            raise ValueError(f"Missing {label} in {path}")
        return match

    steps = int(name_match.group("steps"))
    objective = float(require(OBJECTIVE_RE, "objective").group("objective"))
    grad_match = require(GRAD_RE, "gradient")
    grad_alpha = float(grad_match.group("ga"))
    grad_kappa = float(grad_match.group("gk"))

    row = {
        "alpha": float(require(ALPHA_RE, "alpha").group("alpha")),
        "ad_steps": steps,
        "objective_raw": objective,
        "grad_alpha_raw": grad_alpha,
        "grad_kappa_raw": grad_kappa,
        "objective_per_step": objective / steps,
        "grad_alpha_per_step": grad_alpha / steps,
        "grad_kappa_per_step": grad_kappa / steps,
        "runtime_seconds": float(require(TIME_RE, "runtime").group("runtime")),
        "backend": require(BACKEND_RE, "backend").group("backend"),
        "resolution": require(RES_RE, "resolution").group("nx"),
        "dt": float(require(DT_RE, "dt").group("dt")),
        "kappa": float(require(KAPPA_RE, "kappa").group("kappa")),
        "solver": require(SOLVER_RE, "solver").group("solver"),
        "status": "completed" if "Objective Value:" in text else "unknown",
        "source_log": path.as_posix(),
    }

    for key in ("objective_per_step", "grad_alpha_per_step", "grad_kappa_per_step"):
        if not math.isfinite(float(row[key])):
            raise ValueError(f"Non-finite {key} in {path}")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("results/raw_logs/direct_ad"),
        help="Directory containing ad_a*_s*.out logs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/summary/tables/direct_ad_window_sweep.csv"),
        help="CSV output path.",
    )
    args = parser.parse_args()

    if not args.log_dir.is_dir():
        raise SystemExit(f"Missing log directory: {args.log_dir}")

    rows = [parse_log(path) for path in sorted(args.log_dir.glob("ad_a*_s*.out"))]
    rows.sort(key=lambda row: (float(row["alpha"]), int(row["ad_steps"])))
    if not rows:
        raise SystemExit(f"No direct-AD logs found in {args.log_dir}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "alpha",
        "ad_steps",
        "objective_raw",
        "grad_alpha_raw",
        "grad_kappa_raw",
        "objective_per_step",
        "grad_alpha_per_step",
        "grad_kappa_per_step",
        "runtime_seconds",
        "backend",
        "resolution",
        "dt",
        "kappa",
        "solver",
        "status",
        "source_log",
    ]
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
