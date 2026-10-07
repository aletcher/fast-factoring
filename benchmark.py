"""Reproducible CPU benchmarks; reference factors are used only for checking.

    python3 benchmark.py --timeout 2 --output benchmark_results.json
    python3 benchmark.py --digits 60 --limit 1 --timeout 180 --output completion.json

Each input runs in a fresh isolated interpreter without site-packages (-I -S).
The solver receives ONLY N and the time budget, never the reference factors.
The two-second target applies to factorization CPU time; imports are recorded
separately. Timeouts are explicit failures, not successful factorizations.
"""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from time import process_time


def worker(solver_path, n, timeout):
    import importlib.util

    started = process_time()
    spec = importlib.util.spec_from_file_location("solver", solver_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import_cpu = process_time() - started
    started = process_time()
    try:
        factors = module.factor_semiprime(n, timeout=timeout)
    except TimeoutError:
        result = {"status": "timeout"}
    else:
        result = {"status": "factored", "factors": list(map(str, factors))}
    result.update(cpu_seconds=process_time() - started, import_cpu_seconds=import_cpu)
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path,
                        default=Path(__file__).with_name("semiprimes.json"))
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--digits", type=int, help="select actual decimal length of N")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, default=Path("benchmark_results.json"))
    parser.add_argument("--worker", nargs=2, metavar=("SOLVER_PATH", "N"))
    args = parser.parse_args()
    if args.worker:
        worker(*args.worker, args.timeout)
        return
    solver_path = Path(__file__).with_name("solver.py").resolve()
    rows = json.loads(args.dataset.read_text())
    selected = [(i, row) for i, row in enumerate(rows)
                if args.digits is None or len(str(row["N"])) == args.digits]
    if args.limit is not None:
        selected = selected[:args.limit]
    report = {"python": sys.version, "platform": platform.platform(),
              "solver_sha256": hashlib.sha256(solver_path.read_bytes()).hexdigest(),
              "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
              "cpu_budget_seconds": args.timeout, "fresh_interpreter_per_case": True,
              "site_packages_disabled": True, "results": []}
    for i, row in selected:
        result = subprocess.run(
            [sys.executable, "-I", "-S", str(Path(__file__).resolve()),
             "--worker", str(solver_path), str(row["N"]), "--timeout", str(args.timeout)],
            check=True, capture_output=True, text=True)
        record = json.loads(result.stdout)
        record.update(index=i, digits=len(str(row["N"])))
        if record["status"] == "factored":
            p, q = map(int, record.pop("factors"))
            if not (1 < p <= q and p * q == int(row["N"]) and
                    (p, q) == tuple(sorted((int(row["p"]), int(row["q"]))))):
                raise AssertionError(f"Incorrect factors for case {i}")
            record["verified"] = True
        record["under_two_cpu_seconds"] = (record["status"] == "factored" and
                                           record["cpu_seconds"] < 2)
        report["results"].append(record)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"case {i:2d}, {record['digits']} digits: {record['status']}, "
              f"{record['cpu_seconds']:.6f} CPU s", flush=True)
    groups = defaultdict(list)
    for record in report["results"]:
        groups[record["digits"]].append(record)
    summary = {}
    for digits, records in sorted(groups.items()):
        completed = [r["cpu_seconds"] for r in records if r["status"] == "factored"]
        summary[digits] = {"cases": len(records), "factored": len(completed),
                           "under_two_cpu_seconds": sum(r["under_two_cpu_seconds"]
                                                        for r in records),
                           "timeouts": len(records) - len(completed),
                           "median_completed_cpu_seconds":
                               statistics.median(completed) if completed else None,
                           "max_completed_cpu_seconds": max(completed) if completed else None}
    report["summary"] = summary
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
