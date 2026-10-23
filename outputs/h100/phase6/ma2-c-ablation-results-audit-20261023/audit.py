"""Recompute returned MA2-R x C records and traces without running a GPU job."""
import hashlib
import json
from pathlib import Path
import statistics
import sys

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "tools/h100/phase6"))
import ma2_ablation_bench as bench
import ma2_ablation_fullstep as fullstep

BATCH = REPO / "logs/h100/phase6/ma2r-c-ablation-2x-50steps-20261023-104402-375258"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


files = [BATCH / "batch-state.json"] + [
    BATCH / part / name for part in ("operator", "fullstep")
    for name in ("plan.json", "preflight.json", "report.json", "profile-summary.json")
] + [BATCH / "fullstep/report.md"]
before = {str(p.relative_to(REPO)): sha(p) for p in files}
controllers = {}
for frozen, current in (
    ("operator/scripts/bench.py", "ma2_ablation_bench.py"),
    ("fullstep/scripts/operator-controller.py", "ma2_ablation_bench.py"),
    ("fullstep/scripts/fullstep-controller.py", "ma2_ablation_fullstep.py"),
):
    digest = sha(BATCH / frozen)
    assert digest == sha(REPO / "tools/h100/phase6" / current), frozen
    controllers[frozen] = digest

checks, recomputed = {}, {}
for name, callback in (
    ("operator/report.json", lambda: bench.report(BATCH / "operator", save=False, offline=True)),
    ("fullstep/report.json", lambda: fullstep.report(BATCH / "fullstep", save=False, offline=True)),
    ("operator/profile-summary.json", lambda: bench.profile_summary(BATCH / "operator", offline=True)),
    ("fullstep/profile-summary.json", lambda: fullstep.profile_summary(BATCH / "fullstep", offline=True)),
):
    result = callback()
    assert result == read(BATCH / name), name + ": recomputation differs"
    checks[name] = True
    recomputed[name] = result
    print("Recomputed identically:", name, flush=True)

state = read(BATCH / "batch-state.json")
train = recomputed["fullstep/report.json"]
op = recomputed["operator/report.json"]
assert state["status"] == "completed" and state["stage"] == "done"
assert state["formal_optimizer_steps"] == train["formal_optimizer_steps"] == 3200
assert state["timing_windows"] == len(train["all_samples"]) == 64
assert train["timing_rank_records"] == 512
assert len(op["all_samples"]) == 32
assert len(recomputed["operator/profile-summary.json"]["profiles"]) == 64
assert len(recomputed["fullstep/profile-summary.json"]["profiles"]) == 32
for field in ("target_validated", *fullstep.ACCEPTANCE_FIELDS):
    assert state[field] == train[field]
assert state["operator_bitwise_equal_within_pairs"] == op["bitwise_equal_within_pairs"]
assert state["operator_execution_path_validated"] == recomputed["operator/profile-summary.json"]["execution_path_validated"]
assert state["fullstep_execution_path_validated"] == recomputed["fullstep/profile-summary.json"]["execution_validated"]
assert state["operator_comparisons"] == {role: row["comparisons"] for role, row in op["roles"].items()}
assert train["identical_trajectories_within_pairs"] and train["correctness_passed"]
assert train["equivalence_pairs"] == op["equivalence_pairs"] == [["CR", "C"], ["R", "A"]]

medians = {}
for part in ("operator", "fullstep"):
    profiles = recomputed[part + "/profile-summary.json"]["profiles"]
    medians[part] = {}
    for arm in bench.ARMS:
        medians[part][arm] = {}
        for role in bench.ROLES:
            rows = ([v[role] for k, v in profiles.items() if k.startswith(arm + "/")]
                    if part == "fullstep" else
                    [v for k, v in profiles.items() if k.startswith(arm + "/") and k.endswith("/" + role)])
            assert len(rows) == 8
            medians[part][arm][role] = dict(rank_count=len(rows), **{
                key: statistics.median(v[key] for v in rows)
                for key in rows[0] if key.endswith("median_us")})

variability = {}
for trial in (1, 2):
    samples = [s for s in train["all_samples"] if s["trial"] == trial]
    variability[str(trial)] = dict(
        minimum_ms=min(s["ms_per_step"] for s in samples),
        maximum_ms=max(s["ms_per_step"] for s in samples),
        arm_max_over_median={a: max(s["ms_per_step"] for s in samples if s["arm"] == a) /
            statistics.median(s["ms_per_step"] for s in samples if s["arm"] == a) for a in bench.ARMS})

assert before == {str(p.relative_to(REPO)): sha(p) for p in files}
audit = dict(
    completed=True, gpu_used=False, date="2026-10-23", batch=str(BATCH.relative_to(REPO)),
    recomputed_identically=checks, source_controllers_match=controllers,
    original_reports_plans_gates_unchanged=True, evidence_sha256=before,
    batch_state=state, training_comparisons=train["comparisons"],
    numeric_summary=train["numeric_summary"], operator_comparisons=op["roles"],
    profile_rank_medians=medians, training_window_variability=variability,
    limitation="Offline records and trace verification only; server runtime libraries are not reloaded or rehashed.")
(Path(__file__).parent / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
print("Audit complete: original results unchanged; target_validated =", train["target_validated"], flush=True)
