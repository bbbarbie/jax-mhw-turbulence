#!/usr/bin/env python3
"""Parse ensemble direct-AD logs into a CSV table."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path


LOG_RE = re.compile(r"ens_ad-54341982_(?P<index>\d+)\.out$")
ALPHA_RE = re.compile(r"Alpha = (?P<alpha>[0-9.eE+-]+)")
OBJECTIVE_RE = re.compile(r"Objective Value:\s+(?P<objective>[0-9.eE+-]+)")
GRAD_RE = re.compile(
    r"Gradient w\.r\.t \[alpha, kappa\]: \[\s*(?P<ga>[0-9.eE+-]+)\s+(?P<gk>[0-9.eE+-]+)\s*\]"
)
TIME_RE = re.compile(r"Time:\s+(?P<runtime>[0-9.eE+-]+)s")


def parse_log(path: Path, ad_steps: int) -> dict[str, object]:
    text = path.read_text()
    name_match = LOG_RE.search(path.name)
    if not name_match:
        raise ValueError(f"Unexpected ensemble log filename: {path}")

    def require(pattern: re.Pattern[str], label: str) -> re.Match[str]:
        match = pattern.search(text)
        if not match:
            raise ValueError(f"Missing {label} in {path}")
        return match

    objective = float(require(OBJECTIVE_RE, "objective").group("objective"))
    grad_match = require(GRAD_RE, "gradient")
    grad_alpha = float(grad_match.group("ga"))
    grad_kappa = float(grad_match.group("gk"))

    row = {
        "array_index": int(name_match.group("index")),
        "alpha": float(require(ALPHA_RE, "alpha").group("alpha")),
        "seed": "",
        "seed_mapping_status": "not recoverable from available artifacts",
        "ad_steps": ad_steps,
        "objective_raw": objective,
        "grad_alpha_raw": grad_alpha,
        "grad_kappa_raw": grad_kappa,
        "objective_per_step": objective / ad_steps,
        "grad_alpha_per_step": grad_alpha / ad_steps,
        "grad_kappa_per_step": grad_kappa / ad_steps,
        "runtime_seconds": float(require(TIME_RE, "runtime").group("runtime")),
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
        default=Path("results/raw_logs/ensemble_ad"),
        help="Directory containing ens_ad-54341982_*.out logs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/summary/tables/ensemble_ad_results.csv"),
        help="CSV output path.",
    )
    parser.add_argument(
        "--ad-steps",
        type=int,
        default=10000,
        help="AD step count supported by the existing Stage 2 ensemble summary.",
    )
    args = parser.parse_args()

    if not args.log_dir.is_dir():
        raise SystemExit(f"Missing log directory: {args.log_dir}")
    if args.ad_steps <= 0:
        raise SystemExit("--ad-steps must be positive")

    rows = [parse_log(path, args.ad_steps) for path in sorted(args.log_dir.glob("ens_ad-54341982_*.out"))]
    rows.sort(key=lambda row: int(row["array_index"]))
    if not rows:
        raise SystemExit(f"No ensemble logs found in {args.log_dir}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "array_index",
        "alpha",
        "seed",
        "seed_mapping_status",
        "ad_steps",
        "objective_raw",
        "grad_alpha_raw",
        "grad_kappa_raw",
        "objective_per_step",
        "grad_alpha_per_step",
        "grad_kappa_per_step",
        "runtime_seconds",
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
