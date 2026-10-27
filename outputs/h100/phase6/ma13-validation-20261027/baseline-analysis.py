#!/usr/bin/env python3
"""Read-only reconstruction of MA11 softmax work from all eight original traces.

Run from any directory: python3 baseline-analysis.py [--root REPOSITORY]
Only --output (default: the adjacent baseline-analysis.json) is written.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re
import statistics


BATCH = "ma11-v1-causal-softmax-1x-50steps-20261027-121342-060643"
STRATEGY = "F4-cache-zcount-csoftmax"
EXPECTED_CALLS = 32 * 4


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def describe_us(values):
    values = sorted(float(value) for value in values)
    require(bool(values), "empty duration set")
    position = (len(values) - 1) * 0.90
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return {
        "count": len(values),
        "sum_ms": sum(values) / 1000.0,
        "min_us": values[0],
        "median_us": statistics.median(values),
        "p90_us_linear": values[lower] + (values[upper] - values[lower]) * (position - lower),
        "max_us": values[-1],
    }


def contains(outer, inner):
    return (outer["pid"], outer["tid"]) == (inner["pid"], inner["tid"]) and (
        outer["ts"] <= inner["ts"]
        and inner["ts"] + inner["dur"] <= outer["ts"] + outer["dur"] + 0.01
    )


def analyze_rank(root, rank):
    path = root / "logs/h100/phase6" / BATCH / "fullstep/results" / (
        "profile-" + STRATEGY
    ) / f"rank{rank}-profile.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        events = json.load(handle)["traceEvents"]
    runtimes = {}
    for event in events:
        if event.get("cat") == "cuda_runtime":
            correlation = event.get("args", {}).get("correlation")
            if correlation is not None:
                runtimes.setdefault(correlation, []).append(event)
    result = {"rank": rank, "trace": str(path.relative_to(root)), "sha256": digest(path)}
    for direction, function, block, registers, shared in (
        ("forward", "MA11MaskedSoftmax", [1024, 1, 1], 30, 4224),
        ("backward", "MA11MaskedSoftmaxBackward", [128, 1, 1], 52, 512),
    ):
        kernels = [event for event in events if event.get("cat") == "kernel"
                   and f"ma11_softmax_{direction}_kernel(" in event.get("name", "")]
        cpu_functions = [event for event in events if event.get("cat") == "cpu_op"
                         and event.get("name") == function]
        markers = [event for event in events if event.get("cat") == "user_annotation"
                   and event.get("name") == f"ma11.attention.softmax.{direction}"]
        require(len(kernels) == len(cpu_functions) == len(markers) == EXPECTED_CALLS,
                f"rank{rank} {direction}: expected 128 kernels/functions/markers")
        functions_by_external_id = {event["args"]["External id"]: event for event in cpu_functions}
        require(len(functions_by_external_id) == EXPECTED_CALLS, "duplicate function external IDs")
        seen_functions, seen_markers = set(), set()
        geometry = Counter()
        streams = set()
        launch_durations = []
        for kernel in kernels:
            args = kernel["args"]
            require(args["grid"] == [8192, 1, 1] and args["block"] == block,
                    f"rank{rank} {direction}: unexpected geometry")
            require(args["registers per thread"] == registers and args["shared memory"] == shared,
                    f"rank{rank} {direction}: unexpected resource metadata")
            function_event = functions_by_external_id.get(args["External id"])
            require(function_event is not None, "kernel has no original autograd function")
            require(args["External id"] not in seen_functions, "function owns multiple kernels")
            seen_functions.add(args["External id"])
            launches = runtimes.get(args["correlation"], [])
            require(len(launches) == 1 and "LaunchKernel" in launches[0]["name"],
                    "kernel lacks a unique runtime launch correlation")
            launch = launches[0]
            require(contains(function_event, launch), "launch outside owning autograd function")
            matching = [event for event in markers if contains(event, launch)]
            require(len(matching) == 1, "launch lacks a unique same-thread softmax marker")
            marker_id = matching[0]["args"]["External id"]
            require(marker_id not in seen_markers, "marker owns multiple kernels")
            seen_markers.add(marker_id)
            launch_durations.append(launch["dur"])
            streams.add(args["stream"])
            key = json.dumps({"grid": args["grid"], "block": args["block"],
                              "registers_per_thread": registers, "shared_memory_bytes": shared},
                             sort_keys=True)
            geometry[key] += 1
        require(len(streams) == 1, "softmax unexpectedly spans multiple compute streams")
        result[direction] = {
            "gpu_work": describe_us([event["dur"] for event in kernels]),
            "cpu_autograd_function": describe_us([event["dur"] for event in cpu_functions]),
            "cpu_profile_marker": describe_us([event["dur"] for event in markers]),
            "cpu_runtime_launch": describe_us(launch_durations),
            "geometry": [dict(json.loads(key), count=count) for key, count in geometry.items()],
            "compute_streams": sorted(streams),
            "unique_function_launch_marker_associations": len(seen_functions),
        }
    result["combined_softmax_gpu_work_ms"] = sum(
        result[direction]["gpu_work"]["sum_ms"] for direction in ("forward", "backward")
    )
    return result


def history_review(root):
    directory = root / "docs/design/design-h100/New-Mechanism"
    patterns = {
        "exponent_residency": r"指数驻留|驻留.{0,20}指数|exp\w*.{0,20}寄存器|寄存器.{0,20}exp|resident.{0,20}(softmax|exponent)|softmax.{0,20}resident",
        "virtual_empty_warps": r"虚拟.{0,15}warp|virtual.{0,15}warp|空(?:闲)?.{0,8}warp|empty.{0,15}warp",
    }
    records = []
    for path in sorted(directory.glob("*.md")):
        if path.name == "ma-13.md":
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        hits = []
        for label, pattern in patterns.items():
            for number, line in enumerate(lines, 1):
                if re.search(pattern, line, flags=re.IGNORECASE):
                    hits.append({"pattern": label, "line": number, "text": line})
        records.append({"path": str(path.relative_to(root)), "sha256": digest(path), "hits": hits})
    return {
        "scope": "All prior Markdown files in New-Mechanism; ma-13.md excluded",
        "patterns": patterns,
        "documents": records,
        "total_matching_lines": sum(len(record["hits"]) for record in records),
        "interpretation": "Keyword search plus manual review found no prior proposal of exponent residency with virtual empty-warp reduction. This is a repository-history distinction, not a claim of research novelty. MA11 preserves the original forward reduction schedule; MA6 changed decoder tiling/layout, a different operation.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--output", type=Path, default=Path(__file__).with_suffix(".json"))
    args = parser.parse_args()
    root = args.root.resolve()
    ranks = [analyze_rank(root, rank) for rank in range(8)]
    summary = {}
    for direction in ("forward", "backward"):
        summary[direction] = {
            measure: {
                "median_across_rank_sums_ms": statistics.median(
                    rank[direction][measure]["sum_ms"] for rank in ranks
                ),
                "median_across_rank_per_call_medians_us": statistics.median(
                    rank[direction][measure]["median_us"] for rank in ranks
                ),
                "min_rank_sum_ms": min(rank[direction][measure]["sum_ms"] for rank in ranks),
                "max_rank_sum_ms": max(rank[direction][measure]["sum_ms"] for rank in ranks),
            }
            for measure in ("gpu_work", "cpu_autograd_function", "cpu_profile_marker", "cpu_runtime_launch")
        }
    summary["median_combined_gpu_work_ms"] = statistics.median(
        rank["combined_softmax_gpu_work_ms"] for rank in ranks
    )
    source = root / "tools/h100/phase6/ma11_softmax.cuh"
    output = {
        "protocol": "ma13-ma11-softmax-baseline-analysis-v1",
        "source_batch": BATCH,
        "source_strategy": STRATEGY,
        "script_sha256": digest(Path(__file__)),
        "method": "One profiled optimizer step per rank; 128 calls per direction. Correlate each kernel with one runtime launch, original autograd function, and same-thread profile marker. Sum GPU durations per rank, then take the eight-rank median; never sum across ranks.",
        "limits": [
            "These are profiler work durations, not exposed critical-path savings or unprofiled step time.",
            "CPU function/marker/runtime intervals are nested alternative scopes; do not add them together or add them to GPU work.",
            "No GPU was run by this script; all evidence is reconstructed from the historical MA11 candidate traces.",
            "Reported launch resource metadata is not measured achieved occupancy or proof of a performance bottleneck.",
        ],
        "baseline_source": {"path": str(source.relative_to(root)), "sha256": digest(source)},
        "ranks": ranks,
        "summary": summary,
        "history_review": history_review(root),
    }
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": summary,
                      "history_matching_lines": output["history_review"]["total_matching_lines"]},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
