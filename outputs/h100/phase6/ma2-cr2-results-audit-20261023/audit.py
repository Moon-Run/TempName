"""Recompute returned CR2 records, native cache evidence and traces without running a GPU job."""
import hashlib
import json
from pathlib import Path
import statistics
import sys

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "tools/h100/phase6"))
import ma2_cr2_bench as bench
import ma2_cr2_fullstep as fullstep

BATCH = REPO / "logs/h100/phase6/ma2-cr2-2x-50steps-20261023-153237-090898"


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
    ("operator/scripts/bench.py", "ma2_cr2_bench.py"),
    ("fullstep/scripts/operator-controller.py", "ma2_cr2_bench.py"),
    ("fullstep/scripts/fullstep-controller.py", "ma2_cr2_fullstep.py"),
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
for field in fullstep.aggregate_acceptance(train["comparisons"]):
    assert state[field] == train[field]
assert fullstep.aggregate_acceptance(train['comparisons']) == {
    key:train[key] for key in fullstep.aggregate_acceptance(train['comparisons'])}
assert state["operator_bitwise_equal_within_pairs"] == op["bitwise_equal_within_pairs"]
assert state["operator_execution_path_validated"] == recomputed["operator/profile-summary.json"]["execution_path_validated"]
assert state["fullstep_execution_path_validated"] == recomputed["fullstep/profile-summary.json"]["execution_validated"]
assert state["operator_comparisons"] == {role: row["comparisons"] for role, row in op["roles"].items()}
assert train["identical_trajectories_within_pairs"] and train["correctness_passed"]
assert train["equivalence_pairs"] == op["equivalence_pairs"] == [["CR2", "CR"], ["CR", "C"]]

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

native = dict(formal_rank_records=0, module_records=0, slot_records=0,
              initializations=set(), successful_calls=set(), bound_k=set(), failed=set())
for sample in train['all_samples']:
    if sample['arm'] != 'CR2':
        continue
    for rank in range(8):
        row = read(BATCH/'fullstep/results'/sample['directory']/f'rank{rank}.json')
        native['formal_rank_records'] += 1
        for slots in row['cr2_submission_state'].values():
            native['module_records'] += 1
            for slot in slots:
                native['slot_records'] += 1
                for field in ('initializations','successful_calls','bound_k','failed'):
                    native[field].add(slot[field])
for key,value in list(native.items()):
    if isinstance(value,set):
        native[key] = sorted(value)
assert native == dict(formal_rank_records=128, module_records=8192, slot_records=16384,
                     initializations=[1], successful_calls=[120], bound_k=[512,2048], failed=[False])

profile_counts = {}
for arm in bench.ARMS:
    rows = [v[role] for k,v in recomputed['fullstep/profile-summary.json']['profiles'].items()
            if k.startswith(arm+'/') for role in bench.ROLES]
    profile_counts[arm] = {field:sorted({r[field] for r in rows}) for field in
        ('projection_calls','barrier_calls','output_clones','output_slice_view_count','adapter_view_count')}
assert profile_counts['CR2'] == dict(projection_calls=[128],barrier_calls=[128],output_clones=[0],
                                   output_slice_view_count=[0],adapter_view_count=[256])

assert before == {str(p.relative_to(REPO)): sha(p) for p in files}
audit = dict(
    completed=True, gpu_used=False, date="2026-10-23", batch=str(BATCH.relative_to(REPO)),
    recomputed_identically=checks, source_controllers_match=controllers,
    original_reports_plans_gates_unchanged=True, evidence_sha256=before,
    batch_state=state, training_comparisons=train["comparisons"],
    numeric_summary=train["numeric_summary"], operator_comparisons=op["roles"],
    profile_rank_medians=medians, training_window_variability=variability,
    cr2_formal_native_cache= native, training_profile_counts_per_rank_role=profile_counts,
    limitation="Offline records and trace verification only; server runtime libraries are not reloaded or rehashed.")
(Path(__file__).parent / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
print("Audit complete: original results unchanged; target_validated =", train["target_validated"], flush=True)
