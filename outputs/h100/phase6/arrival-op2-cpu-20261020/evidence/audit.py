#!/usr/bin/env python3
"""Recompute frozen op1-v3 timing/mappings and inspect diagnostic CUDA traces.

Run from any directory: python3 /path/to/evidence/audit.py --out /tmp/op2-audit.json
Original experimental files are read-only. Profile durations are not unprofiled
performance estimates. An output file must not already exist.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[5])
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    root = args.root.resolve()
    batch = root / "logs/h100/phase6/ablation7-new-2x-50steps-20261020-110009-429282"
    hashes = {}

    def record(path):
        hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    def read(path):
        return json.loads(record(path).read_text())

    source = record(root / "tools/h100/phase6/arrival_profile_audit.py")
    spec = importlib.util.spec_from_file_location("profile_audit", source)
    profile = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(profile)
    policies = {
        "A": "interleaved_all_attn_mlp",
        "B": "interleaved_arrival_assignment_all_attn_mlp",
        "C": "interleaved_arrival_selective_b16_attn_mlp",
    }
    result = {
        "schema": "arrival-op2-existing-evidence-audit-v1",
        "new_gpu_execution": False,
        "profile_limitations": (
            "Only rank0/rank4 traces from the two stated compares are checked. "
            "Profiler perturbation prevents using durations as unprofiled step "
            "costs, attainable speedups, or an upper bound on performance."
        ),
        "comparisons": [],
    }
    all_a = []
    total_records = 0
    projection_calls = set()
    for name in ("compare1", "compare2"):
        compare = batch / name
        report = read(compare / "report.json")
        assert report["completed"] and report["correctness_passed"]
        by_policy = defaultdict(list)
        blocks = defaultdict(dict)
        count = 0
        for sample in report["all_samples"]:
            rows = [
                read(compare / "results" / sample["directory"] / f"rank{rank}.json")
                for rank in range(8)
            ]
            assert all(row["completed"] and row["passed"] for row in rows)
            assert all(row["timed_steps"] == 50 and len(row["losses"]) == 50 for row in rows)
            assert all(row["losses"] == rows[0]["losses"] for row in rows)
            ms = max(row["elapsed_seconds"] for row in rows) * 1000 / 50
            assert abs(ms - sample["ms_per_step"]) < 1e-10
            count += len(rows)
            by_policy[sample["policy"]].append(ms)
            blocks[(sample["round"], sample["block"])][sample["policy"]] = ms
            projection_calls.add(rows[0]["timed_target_calls_expected"] / 50)
        assert count == 288 and len(blocks) == 12
        total_records += count
        all_a.extend(by_policy[policies["A"]])
        log_ratios = [
            math.log(block[policies["B"]] / block[policies["A"]])
            for block in blocks.values()
        ]
        block_gains = [100 * (1 - math.exp(value)) for value in log_ratios]
        item = {
            "compare": name,
            "run_id": report["run_id"],
            "verified_rank_records": count,
            "verified_windows": len(report["all_samples"]),
            "mean_ms_per_step": {
                label: statistics.mean(by_policy[policy])
                for label, policy in policies.items()
            },
            "paired_B_over_A_time_reduction_percent": 100 * (1 - math.exp(statistics.mean(log_ratios))),
            "block_time_reduction_percent_range": [min(block_gains), max(block_gains)],
            "paired_log_ratio_standard_deviation_percent": 100 * statistics.stdev(log_ratios),
            "mappings": {},
            "profiles": {},
        }
        for role in ("attn", "mlp"):
            tables = {}
            for label in ("A", "B"):
                path = record(compare / "results" / f"mapping-{policies[label]}-{role}.csv")
                with path.open() as handle:
                    rows = [{key: int(value) for key, value in row.items()} for row in csv.DictReader(handle)]
                tables[label] = {(row["rank"], row["index"]): row for row in rows}
                assert len(tables[label]) == len(rows) == 4096
                for rank in range(8):
                    coordinates = sorted((row["m"], row["n"]) for row in rows if row["rank"] == rank)
                    assert coordinates == [(m, n) for m in range(16) for n in range(32)]
            a, b = tables["A"], tables["B"]
            assert a.keys() == b.keys()
            changed = [
                sum((a[key]["m"], a[key]["n"]) != (b[key]["m"], b[key]["n"])
                    for key in a if key[0] == rank)
                for rank in range(8)
            ]
            destinations = sum(a[key]["destination"] != b[key]["destination"] for key in a)
            assert destinations == 0
            item["mappings"][role] = {
                "logical_slots": len(a),
                "changed_slots": sum(changed),
                "changed_per_rank": changed,
                "destination_changes": destinations,
                "physical_bijection_verified": True,
            }
        for label in ("A", "B"):
            item["profiles"][label] = {}
            for rank in (0, 4):
                path = record(compare / "results" / f"profile-{policies[label]}" / f"rank{rank}-profile.json.gz")
                records = profile.trace_projection_records(path)
                item["profiles"][label][str(rank)] = {
                    role: profile.summarize(rows) for role, rows in records.items()
                }
        result["comparisons"].append(item)
    assert total_records == 576 and projection_calls == {256.0}
    baseline = statistics.mean(all_a)
    result["target_budget"] = {
        "definition": "1.5% reduction in complete optimizer-step time versus existing A",
        "baseline_A_mean_ms_per_step": baseline,
        "required_step_reduction_ms": baseline * 0.015,
        "required_candidate_ms_per_step_at_this_descriptive_baseline": baseline * 0.985,
        "targeted_projection_calls_per_step": 256,
        "required_average_us_per_call_if_256_calls_improve": baseline * 0.015 * 1000 / 256,
        "required_average_us_per_call_if_only_128_calls_improve": baseline * 0.015 * 1000 / 128,
        "limitation": "Arithmetic engineering budget, not an observed improvement or attainable bound.",
    }
    result["input_sha256"] = dict(sorted(hashes.items()))
    result["audit_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.out.open("x") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(json.dumps({
        "output": str(args.out),
        "verified_rank_records": total_records,
        "input_files_hashed": len(hashes),
        "profiled_projection_calls": sum(
            role["projection_calls"] for compare in result["comparisons"]
            for policy in compare["profiles"].values()
            for rank in policy.values() for role in rank.values()
        ),
        "target_budget": result["target_budget"],
    }, indent=2))


if __name__ == "__main__":
    main()
