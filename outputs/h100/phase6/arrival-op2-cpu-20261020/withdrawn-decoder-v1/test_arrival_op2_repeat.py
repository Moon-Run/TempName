"""CPU-only checks for op2 launch ordering, repeat isolation and target gating."""
import contextlib
import copy
import io
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import arrival_op2 as entry
import arrival_op2_build as builder
import arrival_op2_protocol as protocol
import arrival_op2_report as reporter
import detach_arrival_op2 as batch
from common import HERE, read, sha, write


class BatchTests(unittest.TestCase):
    def args(self, root, trials=2):
        return batch.arguments(["--out", str(root), "--trials", str(trials)])

    def info(self, root):
        return dict(
            protocol=protocol.PROTOCOL, run_id="op2/"+root.name,
            executable=True, source_only=False, execution_timing=dict(protocol.TIMING),
            arrival_options=dict(protocol.ARRIVAL_OPTIONS),
            schedule=protocol.schedule(6, rounds=2), timing_windows=36,
            communication=dict(environment={"NCCL_NVLS_ENABLE": "0"}, nvls_request="0"),
            config={}, calibration_hardware={}, megatron="same", megatron_commit="same",
            candidate_source_commit="same", build_manifest={}, build_files={},
            controllers={}, files={"scripts/worker.py": "same"},
            variants={p: protocol.definition(p) for p in protocol.POLICIES},
        )

    def report(self, root, point=2.0, lower=1.6):
        info = read(root/"submission.json")
        gate = dict(completed=True, protocol=protocol.PROTOCOL,
                    execution_timing=dict(protocol.TIMING),
                    arrival_options=dict(protocol.ARRIVAL_OPTIONS),
                    submission_sha256=sha(root/"submission.json"),
                    communication=info["communication"])
        write(root/"preflight.json", gate)
        summary = {"comparisons": {
            protocol.D+"/"+protocol.A: {"percent": point, "ci95": [lower, point+0.4]},
            protocol.R+"/"+protocol.D: {"percent": 0.0, "ci95": [-0.2, 0.2]},
            protocol.R+"/"+protocol.A: {"percent": point, "ci95": [lower, point+0.4]},
        }}
        acceptance = reporter.acceptance(summary)
        return dict(
            protocol=protocol.PROTOCOL, run_id=info["run_id"], completed=True,
            correctness_passed=True, execution_timing=dict(protocol.TIMING),
            arrival_options=dict(protocol.ARRIVAL_OPTIONS),
            formal_optimizer_steps=1800, timing_windows=36, all_samples=[{}]*36,
            round_count=2, blocks_per_round=6, timing_rank_records=288,
            preflight_rank_records=48, profile_traces=24, candidate_shape_mappings=6,
            communication=info["communication"], measurement_identity={"same_runtime": True},
            role_plan_sha256={"same_plans": True},
            arrival_schedule={"same_order": True}, acceptance_threshold=1.5,
            primary_comparisons=protocol.PRIMARY, comparisons=protocol.COMPARISONS,
            rounds={key: copy.deepcopy(summary) for key in ("1", "2", "combined")},
            acceptance_by_round={key: copy.deepcopy(acceptance) for key in ("1", "2", "combined")},
            acceptance=acceptance, evidence={"preflight_sha256": sha(root/"preflight.json")},
        )

    def fake_invoke(self, calls, mutate=None, gain=None):
        def invoke(command, log, env):
            label = log.stem
            calls.append(label)
            self.assertEqual(Path(command[1]).name, "arrival_op2.py")
            self.assertEqual(env["NCCL_NVLS_ENABLE"], "0")
            if label.startswith("compare"):
                root = Path(command[3])
                if label.endswith("-prepare"):
                    root.mkdir()
                    write(root/"submission.json", self.info(root))
                    (root/"scripts").mkdir()
                    write(root/"scripts/config.json", {"role_plan_sha256": {"same_plans": True}})
                elif label.endswith("-controller"):
                    write(root/"all-state.json", {"status": "completed"})
                elif label.endswith("-report"):
                    values = (2.0, 1.6) if gain is None else gain[root.name]
                    write(root/"report.json", self.report(root, *values))
            else:
                root = log.parent
            if mutate:
                mutate(label, root, command)
        return invoke

    def run_fake(self, args, calls, mutate=None, gain=None):
        with mock.patch.object(batch, "resolve_nccl"), \
                mock.patch.object(batch.launcher, "invoke",
                                  side_effect=self.fake_invoke(calls, mutate, gain)), \
                mock.patch.object(batch.reporter, "verify_inputs",
                                  side_effect=lambda root: read(root/"submission.json")), \
                contextlib.redirect_stdout(io.StringIO()):
            return batch.supervise(args)

    def test_preview_does_not_create_directories_or_spawn_processes(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(os, "fork", side_effect=AssertionError("fork")), \
                mock.patch.object(subprocess, "run", side_effect=AssertionError("subprocess")), \
                mock.patch.object(batch, "resolve_nccl", side_effect=AssertionError("NCCL lookup")), \
                mock.patch.object(builder, "verify", side_effect=AssertionError("build lookup")), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            root = Path(tmp)/"absent"
            batch.main(["--out", str(root)])
            self.assertFalse(root.exists())
            self.assertIn("3600 formal steps", output.getvalue())
            self.assertNotIn("compare3", output.getvalue())

    def test_decoder_gpu_gate_precedes_library_build_and_two_balanced_trials(self):
        args = self.args(Path("/tmp/op2-command-only"))
        tasks = batch.commands(args)
        labels = [label for label, _ in tasks]
        self.assertEqual(labels[:3], ["decoder-check", "build-prepare", "compile"])
        self.assertIn("--execute", tasks[0][1])
        self.assertEqual(labels.count("compile"), 1)
        for trial in ("compare1", "compare2"):
            self.assertEqual([label for label in labels if label.startswith(trial)],
                             [trial+"-"+stage for stage in ("prepare", "dry-run", "controller", "report")])
        schedule = protocol.schedule(6, rounds=2)
        for rep in (1, 2):
            for position in range(3):
                policies = [row["policy"] for row in schedule
                            if row["round"] == rep and row["index"] % 3 == position]
                self.assertEqual({p: policies.count(p) for p in protocol.POLICIES},
                                 {p: 2 for p in protocol.POLICIES})
        args.op2_build = Path("/tmp/existing-op2-build")
        reused = [label for label, _ in batch.commands(args)]
        self.assertEqual(reused[0], "decoder-check")
        self.assertNotIn("compile", reused)
        self.assertNotIn("build-prepare", reused)

    def test_two_confirmations_complete_only_after_all_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls = []
            state = self.run_fake(self.args(Path(tmp)), calls)
            self.assertEqual(calls[:3], ["decoder-check", "build-prepare", "compile"])
            self.assertEqual(calls[-1], "compare2-report")
            self.assertEqual(state["status"], "completed")
            self.assertTrue(state["target_validated"])
            self.assertEqual((state["timing_windows"], state["timing_rank_records"],
                              state["formal_optimizer_steps"]), (72, 576, 3600))

    def test_missing_target_is_not_execution_failure_and_no_run_is_dropped(self):
        for bad in ((-0.1, -0.3), (1.7, 1.49)):
            with self.subTest(gain=bad), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                calls = []
                state = self.run_fake(self.args(root), calls,
                                      gain={"compare1": (2.0, 1.6), "compare2": bad})
                self.assertEqual(calls[-1], "compare2-report")
                self.assertEqual(state["status"], "completed")
                self.assertFalse(state["target_validated"])
                self.assertEqual(state["acceptance_status"], "not_validated_1.5_percent")

    def test_failed_or_incomplete_first_comparison_stops_before_second(self):
        for stage in ("decoder-check", "compare1-controller", "compare1-report"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                calls = []
                def corrupt(label, folder, command):
                    if label != stage:
                        return
                    if label.endswith("-report"):
                        result = read(folder/"report.json")
                        result["all_samples"].pop()
                        write(folder/"report.json", result)
                    else:
                        raise subprocess.CalledProcessError(1, command)
                with self.assertRaises((ValueError, subprocess.CalledProcessError)):
                    self.run_fake(self.args(root), calls, mutate=corrupt)
                self.assertEqual(calls[-1], stage)
                self.assertNotIn("compare2-prepare", calls)
                self.assertEqual(read(root/"batch-state.json")["status"], "failed")

    def test_repeat_input_drift_or_duplicate_run_id_stops_before_gpu(self):
        for key, value in (("run_id", "op2/compare1"),
                           ("files", {"scripts/worker.py": "changed"}),
                           ("build_files", {"changed": "hash"})):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                calls = []
                def corrupt(label, folder, command):
                    if label == "compare2-prepare":
                        info = read(folder/"submission.json")
                        info[key] = value
                        write(folder/"submission.json", info)
                with self.assertRaises(ValueError):
                    self.run_fake(self.args(root), calls, mutate=corrupt)
                self.assertEqual(calls[-1], "compare2-dry-run")
                self.assertNotIn("compare2-controller", calls)
                self.assertEqual(read(root/"batch-state.json")["status"], "failed")

    def test_identity_drift_after_second_report_cannot_validate_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calls = []
            def corrupt(label, folder, command):
                if label == "compare2-report":
                    result = read(folder/"report.json")
                    result["measurement_identity"] = {"different": True}
                    write(folder/"report.json", result)
            with self.assertRaises(ValueError):
                self.run_fake(self.args(root), calls, mutate=corrupt)
            self.assertEqual(read(root/"batch-state.json")["status"], "failed")

    def test_four_trial_run_cannot_reuse_the_second_run_id_for_third(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calls = []
            def corrupt(label, folder, command):
                if label == "compare3-prepare":
                    info = read(folder/"submission.json")
                    info["run_id"] = "op2/compare2"
                    write(folder/"submission.json", info)
            with self.assertRaises(ValueError):
                self.run_fake(self.args(root, trials=4), calls, mutate=corrupt)
            self.assertEqual(calls[-1], "compare3-dry-run")
            self.assertNotIn("compare3-controller", calls)
            self.assertEqual(read(root/"batch-state.json")["status"], "failed")

    def test_detach_readiness_failure_is_recorded_and_propagated(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(batch.launcher, "detach", side_effect=RuntimeError("readiness timeout")), \
                contextlib.redirect_stdout(io.StringIO()):
            parent = Path(tmp)/"parent"
            parent.mkdir()
            (parent/"ablation4-manifest.json").write_text("{}")
            toolkit = Path(tmp)/"cuda"
            (toolkit/"bin").mkdir(parents=True)
            (toolkit/"bin/nvcc").write_text("synthetic")
            root = Path(tmp)/"new"
            with self.assertRaisesRegex(RuntimeError, "readiness timeout"):
                batch.main(["--out", str(root), "--build", str(parent),
                            "--build-cuda-home", str(toolkit), "--execute"])
            state = read(root/"batch-state.json")
            self.assertEqual(state["status"], "failed")
            self.assertIn("readiness timeout", state["error"])

    def test_forged_cached_target_cannot_rescue_negative_primary_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calls = []
            def corrupt(label, folder, command):
                if label == "compare1-report":
                    result = read(folder/"report.json")
                    result["rounds"]["combined"]["comparisons"][protocol.D+"/"+protocol.A] = {
                        "percent": -0.5, "ci95": [-0.8, -0.2]}
                    write(folder/"report.json", result)
            with self.assertRaises(ValueError):
                self.run_fake(self.args(root), calls, mutate=corrupt)
            self.assertNotIn("compare2-prepare", calls)

    def test_malformed_cached_acceptance_is_rejected(self):
        corruptions = (
            {"passed": "false"},
            {"candidate": protocol.R},
            {"threshold_percent": 0.0},
            {"ci95": [2.0, 1.0]},
        )
        for change in corruptions:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                calls = []
                def corrupt(label, folder, command):
                    if label == "compare1-report":
                        result = read(folder/"report.json")
                        result["acceptance"].update(change)
                        write(folder/"report.json", result)
                with self.assertRaises(ValueError):
                    self.run_fake(self.args(root), calls, mutate=corrupt)
                self.assertNotIn("compare2-prepare", calls)

    def test_compile_only_decoder_check_cannot_authorize_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"validation.json"
            write(path, dict(protocol=protocol.PROTOCOL, completed=True, gpu_executed=False,
                             correctness_passed=False, measured={"bitwise_equal": False}))
            with self.assertRaisesRegex(ValueError, "GPU decoder bitwise"):
                entry.verify_decoder_check(path)

    def test_decoder_launch_patch_retains_original_encoder_and_scalar_decoder(self):
        source = (HERE/"fused_v2"/builder.RUNTIME).read_text()
        changed = builder.decoder_source(source)
        marker = 'extern "C" int taco_decode_reduce'
        before = source.split(marker, 1)[0]
        expected_prefix = before.replace('#include "gemm_rs/taco_codec.cuh"',
                                        '#include "gemm_rs/taco_codec.cuh"\n'
                                        '#include "gemm_rs/arrival_op2_decoder.cuh"')
        self.assertEqual(changed.split(marker, 1)[0], expected_prefix)
        self.assertIn("return taco_decode_reduce_op2(c, local, output, stream);", changed)
        with self.assertRaises(ValueError):
            builder.decoder_source(source.replace("taco_groups(c), 128", "taco_groups(c), 256"))


if __name__ == "__main__":
    unittest.main()
