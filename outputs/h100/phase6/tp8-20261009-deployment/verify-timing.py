"""Verify this frozen H100 campaign and export the source data for its record."""
import datetime
import json
from pathlib import Path
import subprocess
import sys

repo = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(repo / "tools/h100/phase6"))
from common import orders, read, sha, verify_files, write
from report import fingerprints, report, verify_inputs, window

root = repo / "logs/h100/phase6/tp8-20261009-deployment/compare"
info = verify_inputs(root)
verify_files("/", info["controllers"])
assert read(root / "timing-state.json")["status"] == "completed"
gate = read(root / "preflight.json")
saved = read(root / "report.json")
recomputed = report(root)
assert saved == recomputed, "Recomputed statistics differ"
rows = []
all_records = []
references = {}
repeatability = {p: dict(max_initial_loss_delta=0, max_timed_loss_delta=0,
    max_timed_grad_norm_delta=0) for p in info["policies"]}
for rep, blocks in enumerate(orders(info["config"]), 1):
    index = 0
    for block, policies in enumerate(blocks):
        for policy in policies:
            directory = root / "results" / f"round{rep}/window-{index:03d}-{policy}"
            records = window(root, directory, policy, "timing", block, index)
            assert fingerprints(records) == [tuple(x) for x in gate["fingerprints"]]
            assert all(not r["sampling"] and not r["profiling"] for r in records)
            for record in records:
                key = (policy, record["rank"])
                ref = references.setdefault(key, record)
                stats = repeatability[policy]
                stats["max_initial_loss_delta"] = max(stats["max_initial_loss_delta"],
                    abs(record["initial_loss"] - ref["initial_loss"]))
                for field, output in (("losses", "max_timed_loss_delta"),
                                      ("grad_norms", "max_timed_grad_norm_delta")):
                    stats[output] = max(stats[output],
                        max(abs(a-b) for a, b in zip(record[field], ref[field])))
            all_records.extend(records)
            rows.append(dict(round=rep, block=block, index=index, policy=policy,
                path=str(directory.relative_to(root)),
                ms_per_step=max(r["elapsed_seconds"] for r in records) * 1000 / info["config"]["timed_steps"],
                rank_elapsed_seconds=[r["elapsed_seconds"] for r in records],
                peak_allocated_gib=max(r["peak_allocated_gib"] for r in records),
                peak_reserved_gib=max(r["peak_reserved_gib"] for r in records)))
            index += 1
assert len(rows) == 72 and len(all_records) == 576
smoke = {}
for policy in info["policies"]:
    records = window(root, root / "results" / ("smoke-" + policy), policy, "smoke")
    checks = [check for r in records for check in r["forward_checks"].values()]
    quant = [check for check in checks if "budget_vs_bf16" in check]
    smoke[policy] = dict(rank_records=len(records), forward_checks=len(checks),
        quantized_forward_checks=len(quant),
        max_relative_l2=max((x.get("relative_l2", 0) for x in checks), default=0),
        max_protocol_relative_l2=max((x.get("protocol_relative_l2", 0) for x in quant), default=0))
zone = datetime.timezone(datetime.timedelta(hours=8))
stamp = lambda p: datetime.datetime.fromtimestamp(p.stat().st_mtime, zone).isoformat()
plan = read(Path(__file__).parent / "selection-plan.json")
calibration = {}
for name, entries in plan["policies"].items():
    e = entries[0]
    selected = sum(map(sum, e["mask"]))
    calibration[name] = dict(selected_per_source=e["selected_per_source"],
        selected_remote_tiles=selected, available_remote_tiles=1792,
        selected_fraction=selected / 1792,
        training_valid_tiles=e["training_valid_tiles"], total_tiles=e["tiles"],
        invalid_observations=len(e["invalid_observations"]),
        heldout_near_critical_fraction=e["heldout_near_critical_fraction"],
        sampled_median_overhead_percent=e["sampled_median_overhead_percent"])
result = dict(completed=True, report_recomputed_identically=True,
    protocol=info["protocol"], run_id=info["run_id"],
    submission_sha256=sha(root / "submission.json"),
    started_beijing=stamp(root / "results/round1/window-000-native/command.json"),
    finished_beijing=stamp(root / "timing-state.json"),
    timing_windows=len(rows), timing_rank_records=len(all_records),
    preflight_rank_records=96, rank_profile_traces=48,
    optimizer_steps_timed=72 * info["config"]["timed_steps"],
    optimizer_steps_warmup=72 * info["config"]["warmup_steps"],
    skips=sum(r["skips"] for r in all_records),
    fallback_calls=sum(r["fallback_calls"] for r in all_records),
    all_finite_trajectories=True, identical_initialization_inputs_hardware=True,
    all_timing_sampling_disabled=True, all_timing_profiling_disabled=True,
    peak_allocated_gib=max(r["peak_allocated_gib"] for r in all_records),
    peak_reserved_gib=max(r["peak_reserved_gib"] for r in all_records),
    smoke=smoke, calibration=calibration, repeatability=repeatability,
    accepted=recomputed["accepted"])
write(root / "window-statistics.json", rows)
write(root / "verification.json", result)
write(root / "timing-artifact-hashes.json", {
    str(p.relative_to(root)): sha(p)
    for p in (root / "results").glob("round*/window-*/*") if p.is_file()})
print(json.dumps(result, ensure_ascii=False, indent=2))
for r in recomputed["rounds"]:
    print("MEDIANS", r["round"], json.dumps(r["median_ms"]))
    for candidate, comparisons in r["comparisons"].items():
        parts = []
        for baseline in ("native_taco", "taco_fused"):
            s = comparisons[baseline]
            parts.append(f'{s["percent"]:.3f}% [{s["ci95"][0]:.3f}, {s["ci95"][1]:.3f}]')
        print("DOCROW", candidate, r["round"],
            f'{r["median_ms"][candidate]:.3f}', *parts,
            f'{-comparisons["native"]["percent"]:+.3f}',
            f'{-comparisons["original"]["percent"]:+.3f}')
print(subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.used,utilization.gpu", "--format=csv"], text=True))
