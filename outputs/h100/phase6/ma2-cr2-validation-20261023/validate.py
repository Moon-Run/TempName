"""Collect CR2 CPU/compile evidence; never starts or claims a GPU run."""
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[3]
sys.path.insert(0, str(REPO / "tools/h100/phase6"))
import ma2_cr2_build as builder
import ma2_cr2_fullstep as fullstep
import ma2_cr2_sources as sources


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


root = OUT / "final/build"
record = builder.verify(root)
compiled = read(OUT / "final/runtime-check/validation.json")
assert compiled["completed"] and not compiled["gpu_used"] and not compiled["linked_runtime"]
assert compiled["builder_inputs"] == record["inputs"]
assert compiled["build_manifest_sha256"] == sha(root / "manifest.json")
assert compiled["cr2_manifest_sha256"] == sha(root / builder.MANIFEST)
assert all(row["host_mapping"] and row["runtime_object"] for row in compiled["checks"].values())
assert compiled["checks"]["CR2"]["ths_wrapper_object"] and compiled["checks"]["CR2"]["pybind_object"]
for name, digest in compiled["changed_sources"].items():
    assert sha(root / "CR2" / name) == digest
assert len(compiled["changed_sources"]) == 6
assert (root / "CR2/src/gemm_rs/ma2_c_decode.cuh").read_bytes() == (root / "CR/src/gemm_rs/ma2_c_decode.cuh").read_bytes()
assert (root / "CR2/src/gemm_rs/taco_codec.cuh").read_bytes() == (root / "CR/src/gemm_rs/taco_codec.cuh").read_bytes()

log = (OUT / "cpu-tests.log").read_text()
assert re.search(r"Ran 34 tests in .*\n\nOK\s*$", log), "Incomplete CPU regression log"
preview = (OUT / "launch-preview.txt").read_text()
assert "64 windows, 3200 formal steps" in preview and "PLAN ONLY" in preview
preview_root = Path(preview.splitlines()[0].removeprefix("Batch: "))
assert not preview_root.exists(), "Command preview created a batch directory"

batch = REPO / "logs/h100/phase6/ma2r-c-ablation-2x-50steps-20261023-104402-375258"
profiles = {}
for arm in ("CR", "C", "R"):
    path = batch / "fullstep/results" / f"profile-{arm}" / "rank0-profile.json.gz"
    with gzip.open(path, "rt") as handle:
        events = json.load(handle)["traceEvents"]
    profiles[arm] = dict(source=str(path.relative_to(REPO)), sha256=sha(path),
                         result=fullstep.trace_projection_records(events, arm))

tools = ("ma2_cr2_build.py", "ma2_cr2_sources.py", "ma2_cr2_bench.py", "ma2_cr2_fullstep.py",
         "detach_ma2_cr2.py", "test_ma2_cr2.py", "test_ma2_cr2_sources.py")
result = dict(
    completed=True, date="2026-10-23", gpu_used=False, gpu_validated=False,
    linked_runtime=False, performance_target_validated=False,
    protocol=builder.PROTOCOL, arms=list(builder.VARIANTS),
    formal_windows=64, formal_optimizer_steps=3200,
    cpu_tests_passed=34, cpu_tests_sha256=sha(OUT / "cpu-tests.log"),
    compile_validation_sha256=sha(OUT / "final/runtime-check/validation.json"),
    source_files={str((REPO / "tools/h100/phase6" / name).relative_to(REPO)):
                  sha(REPO / "tools/h100/phase6" / name) for name in tools},
    control_sources_unchanged=True,
    source_file_counts={arm:len(hashes) for arm,hashes in record["variants"].items()},
    changed_cr2_sources=compiled["changed_sources"],
    source_manifest_sha256=sha(root / builder.MANIFEST),
    old_profile_parser_compatibility=profiles, preview_has_no_side_effects=True,
    limitation="Source identity, CPU contracts and real SM90/native object compilation only. "
               "Old control profiles are parser compatibility evidence, not CR2 GPU execution or performance.")
(OUT / "validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print("Validated: 34 CPU tests; four runtime objects; CR2 wrapper and pybind objects; frozen controls; old trace compatibility.")
