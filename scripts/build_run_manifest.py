#!/usr/bin/env python3
"""Build summary CSV tables and a run manifest from archived evidence."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


FD_VALUES = [
    {
        "baseline_alpha": 0.2,
        "alpha_minus": 0.18,
        "gamma_minus": 0.02776784,
        "alpha_plus": 0.22,
        "gamma_plus": 0.03121801,
        "delta_alpha_total": 0.04,
        "finite_difference_sensitivity": 0.08625416,
    },
    {
        "baseline_alpha": 0.8,
        "alpha_minus": 0.78,
        "gamma_minus": 0.01744796,
        "alpha_plus": 0.82,
        "gamma_plus": 0.01702120,
        "delta_alpha_total": 0.04,
        "finite_difference_sensitivity": -0.01066905,
    },
]


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


def finite_difference_summary(output: Path) -> None:
    source = "results/stage1/stage1_fd_summary.txt"
    rows = []
    for row in FD_VALUES:
        rows.append(
            {
                **row,
                "source_summary": source,
                "raw_hdf5_available": "false",
                "verification_status": "supported by existing postprocessed summary; raw HDF5 unavailable during wrap-up",
            }
        )

    write_rows(
        output,
        [
            "baseline_alpha",
            "alpha_minus",
            "gamma_minus",
            "alpha_plus",
            "gamma_plus",
            "delta_alpha_total",
            "finite_difference_sensitivity",
            "source_summary",
            "raw_hdf5_available",
            "verification_status",
        ],
        rows,
    )


def manifest(output: Path, repo_root: Path) -> None:
    rows: list[dict[str, object]] = []

    fd_log_dir = repo_root / "results/raw_logs/finite_difference"
    fd_re = re.compile(r"mhw_a(?P<tag>\d+)-(?P<job>\d+)\.out")
    for path in sorted(fd_log_dir.glob("mhw_a*.out")):
        text = path.read_text()
        alpha_match = re.search(r"Alpha = (?P<alpha>[0-9.eE+-]+)", text)
        res_match = re.search(r"Resolution:\s+(?P<nx>\d+)\s+x\s+(?P<ny>\d+)", text)
        dt_match = re.search(r"Time Step:\s+dt = (?P<dt>[0-9.eE+-]+)", text)
        solver_match = re.search(r"Solver:\s+(?P<solver>\w+)", text)
        output_match = re.search(r"Finished\. Saved to (?P<output>\S+)", text)
        rows.append(
            {
                "category": "finite_difference_production",
                "alpha": alpha_match.group("alpha") if alpha_match else "",
                "resolution": res_match.group("nx") if res_match else "",
                "dt": dt_match.group("dt") if dt_match else "",
                "solver": solver_match.group("solver") if solver_match else "",
                "steps_or_final_time": "400000 steps; T=1000",
                "status": "completed" if output_match else "unknown",
                "output": output_match.group("output") if output_match else "",
                "source_log": path.relative_to(repo_root).as_posix(),
                "notes": "HDF5 output is not available in this local archive",
            }
        )

    direct_dir = repo_root / "results/raw_logs/direct_ad"
    for path in sorted(direct_dir.glob("ad_a*_s*.out")):
        text = path.read_text()
        alpha = re.search(r"Alpha = (?P<alpha>[0-9.eE+-]+)", text)
        steps = re.search(r"_s(?P<steps>\d+)-", path.name)
        rows.append(
            {
                "category": "historical_direct_ad_sweep",
                "alpha": alpha.group("alpha") if alpha else "",
                "resolution": "64",
                "dt": "0.0025",
                "solver": "rk4",
                "steps_or_final_time": f"{steps.group('steps')} AD steps" if steps else "",
                "status": "completed" if "Objective Value:" in text else "unknown",
                "output": "log only",
                "source_log": path.relative_to(repo_root).as_posix(),
                "notes": "historical accumulated objective; no detached saturated spin-up",
            }
        )

    ensemble_dir = repo_root / "results/raw_logs/ensemble_ad"
    for path in sorted(ensemble_dir.glob("ens_ad-54341982_*.out")):
        text = path.read_text()
        alpha = re.search(r"Alpha = (?P<alpha>[0-9.eE+-]+)", text)
        index = re.search(r"_(?P<index>\d+)\.out", path.name)
        rows.append(
            {
                "category": "ensemble_ad",
                "alpha": alpha.group("alpha") if alpha else "",
                "resolution": "64",
                "dt": "0.0025",
                "solver": "rk4",
                "steps_or_final_time": "10000 AD steps",
                "status": "completed" if "Objective Value:" in text else "unknown",
                "output": "log only",
                "source_log": path.relative_to(repo_root).as_posix(),
                "notes": f"array index {index.group('index')}; seed mapping not recoverable from available artifacts" if index else "seed mapping not recoverable",
            }
        )

    failure_dir = repo_root / "results/raw_logs/failures"
    failure_paths = sorted(failure_dir.rglob("*.log")) + sorted(failure_dir.rglob("*.out"))
    for path in failure_paths:
        text = path.read_text()
        rows.append(
            {
                "category": "failure_or_debug",
                "alpha": "",
                "resolution": "",
                "dt": "",
                "solver": "",
                "steps_or_final_time": "",
                "status": "failed" if "error:" in text or "can't open file" in text else "completed",
                "output": "log only",
                "source_log": path.relative_to(repo_root).as_posix(),
                "notes": "see failure log for exact message",
            }
        )

    write_rows(
        output,
        [
            "category",
            "alpha",
            "resolution",
            "dt",
            "solver",
            "steps_or_final_time",
            "status",
            "output",
            "source_log",
            "notes",
        ],
        rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."), help="Repository root.")
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("results/summary/tables/finite_difference_summary.csv"),
        help="Finite-difference summary CSV output path.",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path("results/summary/tables/run_manifest.csv"),
        help="Run manifest CSV output path.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    if not (repo_root / "results/stage1/stage1_fd_summary.txt").is_file():
        raise SystemExit("Missing results/stage1/stage1_fd_summary.txt")

    finite_difference_summary(args.summary_output)
    manifest(args.manifest_output, repo_root)


if __name__ == "__main__":
    main()
