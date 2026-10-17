"""CPU invariants; all timing fixtures below are explicitly synthetic rehearsal."""
import copy
import importlib.util
from pathlib import Path
import unittest


SPEC = importlib.util.spec_from_file_location("phase7_planner_under_test", Path(__file__).with_name("planner.py"))
planner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(planner)


def small_shape():
    return dict(role="attention", world_size=2, M=512, N=256, K_local=128, tile_m=128, tile_n=128)


def evidence_for(shape=None, chunk_tiles=2):
    shape = small_shape() if shape is None else shape
    return dict(schema_version=1, shape=shape, chunk_tiles=chunk_tiles,
                provenance=dict(run_id="synthetic-unit-test", runtime_sha256="a" * 64,
                                input_sha256="b" * 64, measurement_kind="rehearsal",
                                clock_domain="receiver_relative", sampling_enabled=True),
                ordering_observations=[], interventions=[])


def add_ordering(evidence, split="train", block="ordering", reverse=True):
    shape, size = evidence["shape"], evidence["chunk_tiles"]
    chunks = planner.geometry(shape, size)["chunks_per_destination"]
    for destination in range(shape["world_size"]):
        for chunk in range(chunks):
            value = 10 + chunk if reverse else 10 - chunk
            evidence["ordering_observations"].append(dict(split=split, block_id=block,
                destination=destination, chunk_id=chunk, arrival_us=[value] * shape["world_size"]))


def add_trial(evidence, candidate, pairs, gain, *, split="train", block=None,
              error=0.01, ordering="identity", recalibrated=True):
    shape, size = evidence["shape"], evidence["chunk_tiles"]
    order = planner.make_plan(shape, chunk_tiles=size, ordering=ordering, evidence=evidence)["order_sha256"]
    g = planner.geometry(shape, size)
    arrivals = [[[100.] * shape["world_size"] for _ in range(g["chunks_per_destination"])]
                for _ in range(shape["world_size"])]
    evidence["interventions"].append(dict(split=split, block_id=block or f"{candidate}-{split}",
        candidate_id=candidate, order_sha256=order, recalibrated_after_order=recalibrated,
        mask=pairs, baseline_operator_us=100., candidate_operator_us=100. - gain,
        baseline_arrivals_us=copy.deepcopy(arrivals), candidate_arrivals_us=copy.deepcopy(arrivals),
        encode_us=7., decode_us=8., schedule_us=4., max_abs_error=error))


def target(evidence, **kwargs):
    return planner.make_plan(evidence["shape"], chunk_tiles=evidence["chunk_tiles"],
                             mode="targeted", evidence=evidence, budget_per_source=[2, 2],
                             max_abs_error=0.1, **kwargs)


def legacy_entry(role="attention"):
    shape = planner.default_shape(role)
    g = planner.geometry(shape, 8)
    mask = [[0] * g["tiles"] for _ in range(shape["world_size"])]
    mask[0][64] = 1
    mask[1][0] = 1
    coords = []
    for source in range(shape["world_size"]):
        coords.append([((source + offset) % shape["world_size"]) * g["tiles_per_destination"] + within
                       for within in range(g["tiles_per_destination"]) for offset in range(shape["world_size"])])
    return dict(M=shape["M"], N=shape["N"], K_local=shape["K_local"], tiles=g["tiles"], mask=mask, coords=coords)


class GeometryTests(unittest.TestCase):
    def test_default_real_shapes_and_chunk_coverage(self):
        for role, k in (("attention", 512), ("mlp", 2048)):
            shape = planner.default_shape(role)
            self.assertEqual(shape["K_local"], k)
            for size in (1, 2, 4, 8, 16, 32, 64):
                g = planner.geometry(shape, size)
                self.assertEqual(g["tiles_per_destination"], 64)
                covered = [tile for destination in range(8) for chunk in range(64 // size)
                           for tile in planner.chunk_physical_tiles(shape, size, destination, chunk)]
                self.assertEqual(covered, list(range(512)))

    def test_invalid_shape_and_granularity(self):
        for shape in (dict(small_shape(), M=513), dict(small_shape(), world_size=True),
                      dict(small_shape(), typo=1), dict(small_shape(), tile_m=64)):
            with self.assertRaises(ValueError):
                planner.make_plan(shape, chunk_tiles=2)
        with self.assertRaises(ValueError):
            planner.make_plan(small_shape(), chunk_tiles=3)

    def test_shared_order_and_staggered_destinations(self):
        evidence = evidence_for()
        add_ordering(evidence)
        plan = planner.make_plan(small_shape(), chunk_tiles=2, ordering="shared", evidence=evidence)
        self.assertEqual(plan["chunk_order"], [[1, 0], [1, 0]])
        for source, schedule in enumerate(plan["source_schedules"]):
            self.assertEqual([item["destination"] for item in schedule], [source, (source + 1) % 2] * 2)
            for destination in range(2):
                self.assertEqual([item["chunk_id"] for item in schedule if item["destination"] == destination], [1, 0])

    def test_partial_destination_keeps_identity(self):
        evidence = evidence_for()
        add_ordering(evidence)
        evidence["ordering_observations"] = evidence["ordering_observations"][1:]
        plan = planner.make_plan(small_shape(), chunk_tiles=2, ordering="shared", evidence=evidence)
        self.assertEqual(plan["chunk_order"], [[0, 1], [1, 0]])

    def test_forged_schedule_and_local_mask_rejected(self):
        plan = planner.make_plan(small_shape(), chunk_tiles=2)
        bad = copy.deepcopy(plan)
        bad["source_schedules"][0][0]["chunk_id"] = 1
        with self.assertRaises(ValueError):
            planner.validate_plan(bad)
        bad = copy.deepcopy(plan)
        bad["quantize_mask"][0][0] = True
        with self.assertRaises(ValueError):
            planner.validate_plan(bad)


class SelectionTests(unittest.TestCase):
    def test_zero_and_all_remote_controls(self):
        zero = planner.make_plan(small_shape(), chunk_tiles=2)
        full = planner.make_plan(small_shape(), chunk_tiles=2, mode="all_remote")
        self.assertEqual(zero["selected_per_source"], [0, 0])
        self.assertEqual(full["selected_per_source"], [4, 4])
        self.assertFalse(any(full["quantize_mask"][0][:4]))
        self.assertTrue(all(full["quantize_mask"][0][4:]))

    def test_no_evidence_requires_explicit_bf16(self):
        with self.assertRaisesRegex(ValueError, "measured training interventions"):
            target(evidence_for())

    def test_measured_joint_mask_beats_zero_gain_singletons(self):
        evidence = evidence_for()
        add_trial(evidence, "source0-only", [[0, 4]], 0)
        add_trial(evidence, "source1-only", [[1, 0]], 0)
        add_trial(evidence, "joint", [[0, 4], [1, 0]], 3)
        plan = target(evidence)
        self.assertEqual(plan["selection"]["candidate_id"], "joint")
        # 19 us of diagnostic stage times must not be subtracted a second time.
        self.assertEqual(plan["selection"]["training_median_gain_us"], 3)
        self.assertEqual(plan["selected_per_source"], [1, 1])
        self.assertFalse(plan["performance_validated"])
        self.assertEqual(plan["evidence_kind"], "rehearsal")

    def test_independent_mask_gains_are_never_combined(self):
        evidence = evidence_for()
        add_trial(evidence, "a", [[0, 4]], 2)
        add_trial(evidence, "b", [[1, 0]], 3)
        plan = target(evidence)
        self.assertEqual(plan["selected_per_source"], [0, 1])
        self.assertEqual(plan["selection"]["candidate_id"], "b")

    def test_budget_error_and_positive_net_gain_constraints(self):
        evidence = evidence_for()
        add_trial(evidence, "overbudget", [[0, 4], [0, 5], [0, 6]], 20)
        add_trial(evidence, "unsafe-error", [[0, 4]], 10, error=0.2)
        add_trial(evidence, "negative", [[1, 0]], -2)
        add_trial(evidence, "eligible", [[0, 5]], 1)
        plan = target(evidence)
        self.assertEqual(plan["selection"]["candidate_id"], "eligible")
        self.assertTrue(plan["quantize_mask"][0][5])
        self.assertFalse(plan["quantize_mask"][0][4])

    def test_measured_no_gain_uses_zero_fallback(self):
        evidence = evidence_for()
        add_trial(evidence, "slower", [[0, 4]], -1)
        plan = target(evidence)
        self.assertEqual(plan["selected_per_source"], [0, 0])
        self.assertIn("no eligible", plan["selection"]["rationale"])

    def test_reordering_does_not_rebind_physical_masks(self):
        evidence = evidence_for()
        add_ordering(evidence)
        add_trial(evidence, "physical-tile-4", [[0, 4]], 2, ordering="shared")
        plan = target(evidence, ordering="shared")
        self.assertEqual(plan["chunk_order"][1], [1, 0])
        self.assertTrue(plan["quantize_mask"][0][4])
        self.assertFalse(plan["quantize_mask"][0][6])

    def test_old_order_and_not_recalibrated_are_rejected(self):
        for changed_order in (True, False):
            evidence = evidence_for()
            add_ordering(evidence)
            add_trial(evidence, "old", [[0, 4]], 3, recalibrated=changed_order)
            with self.assertRaisesRegex(ValueError, "remeasured"):
                target(evidence, ordering="shared" if changed_order else "identity")

    def test_random_is_reproducible_matched_and_remote(self):
        evidence = evidence_for()
        add_trial(evidence, "target", [[0, 4], [0, 5], [1, 0]], 3)
        selected = target(evidence)
        random1 = planner.make_plan(small_shape(), chunk_tiles=2, mode="random_matched", matched_plan=selected, seed=42)
        random2 = planner.make_plan(small_shape(), chunk_tiles=2, mode="random_matched", matched_plan=selected, seed=42)
        self.assertEqual(random1, random2)
        self.assertEqual(random1["selected_per_source"], selected["selected_per_source"])
        self.assertEqual(random1["selection"]["numerical_status"], "not_measured")
        self.assertEqual(random1["evidence_kind"], "rehearsal")
        self.assertEqual(random1["evidence_sha256"], selected["evidence_sha256"])
        self.assertEqual(random1["matched_plan_sha256"], planner.plan_digest(selected))
        self.assertFalse(any(random1["quantize_mask"][0][:4]))


class EvidenceTests(unittest.TestCase):
    def test_heldout_and_final_cannot_change_fitted_order_or_mask(self):
        evidence = evidence_for()
        add_ordering(evidence)
        add_trial(evidence, "a", [[0, 4]], 2, ordering="shared")
        expected = target(evidence, ordering="shared")
        add_ordering(evidence, split="heldout", block="hold-order", reverse=False)
        add_trial(evidence, "held-favorite", [[1, 0]], 99, split="heldout", ordering="shared")
        add_trial(evidence, "final-favorite", [[1, 1]], 98, split="final", ordering="shared")
        actual = target(evidence, ordering="shared")
        self.assertEqual(expected["chunk_order"], actual["chunk_order"])
        self.assertEqual(expected["quantize_mask"], actual["quantize_mask"])
        self.assertEqual(expected["fit_block_ids"], actual["fit_block_ids"])

    def test_cross_split_block_leakage_rejected(self):
        evidence = evidence_for()
        add_trial(evidence, "a", [[0, 4]], 2, block="same")
        add_trial(evidence, "a", [[0, 4]], 9, split="heldout", block="same")
        with self.assertRaisesRegex(ValueError, "more than one data split"):
            target(evidence)

    def test_unknown_fields_nan_duplicates_absolute_clocks_rejected(self):
        good = evidence_for()
        add_trial(good, "a", [[0, 4]], 2)
        variants = []
        bad = copy.deepcopy(good)
        bad["interventions"][0]["secret_score"] = 1
        variants.append(bad)
        bad = copy.deepcopy(good)
        bad["interventions"][0]["candidate_operator_us"] = float("nan")
        variants.append(bad)
        bad = copy.deepcopy(good)
        bad["interventions"].append(copy.deepcopy(bad["interventions"][0]))
        variants.append(bad)
        bad = copy.deepcopy(good)
        bad["provenance"]["clock_domain"] = "gpu_absolute"
        variants.append(bad)
        for bad in variants:
            with self.assertRaises(ValueError):
                target(bad)

    def test_frozen_evaluation_preserves_plan_and_rejects_refitting(self):
        evidence = evidence_for()
        add_trial(evidence, "a", [[0, 4]], 2)
        plan = target(evidence)
        frozen = planner.plan_digest(plan)
        independent = evidence_for()
        add_trial(independent, "a-eval", [[0, 4]], -1, split="heldout", block="new-paired-block")
        report = planner.evaluate_frozen_plan(plan, independent, frozen_plan_sha256=frozen)
        self.assertEqual(report["median_gain_us"], -1)
        self.assertEqual(planner.plan_digest(plan), frozen)
        self.assertFalse(report["performance_validated"])
        independent["interventions"][0]["block_id"] = plan["fit_block_ids"][0]
        with self.assertRaisesRegex(ValueError, "reuses"):
            planner.evaluate_frozen_plan(plan, independent, frozen_plan_sha256=frozen)

    def test_final_evaluation_cannot_promote_rehearsal(self):
        evidence = evidence_for()
        add_trial(evidence, "a", [[0, 4]], 2)
        plan = target(evidence)
        final = evidence_for()
        add_trial(final, "a", [[0, 4]], 2, split="final", block="independent-final")
        with self.assertRaisesRegex(ValueError, "GPU measurement"):
            planner.evaluate_frozen_plan(plan, final, frozen_plan_sha256=planner.plan_digest(plan), split="final")

    def test_plan_cannot_claim_validated_performance(self):
        plan = planner.make_plan(small_shape(), chunk_tiles=2)
        plan["performance_validated"] = True
        with self.assertRaisesRegex(ValueError, "cannot validate"):
            planner.validate_plan(plan)

    def test_boolean_counts_and_forged_provenance_rejected(self):
        evidence = evidence_for()
        add_trial(evidence, "a", [[0, 4]], 2)
        plan = target(evidence)
        bad = copy.deepcopy(plan)
        bad["selected_per_source"][0] = True
        with self.assertRaises(ValueError):
            planner.validate_plan(bad)
        bad = copy.deepcopy(plan)
        bad["provenance"]["runtime_sha256"] = "missing"
        with self.assertRaises(ValueError):
            planner.validate_plan(bad)


class LegacyImportTests(unittest.TestCase):
    def test_import_preserves_mask_and_source_entry(self):
        entry = legacy_entry()
        before = copy.deepcopy(entry)
        identity = planner.import_phase6_plan(entry, "attention")
        shared = planner.import_phase6_plan(entry, "attention", ordering="shared")
        self.assertEqual(entry, before)
        self.assertEqual(identity["quantize_mask"], shared["quantize_mask"])
        self.assertEqual(identity["selected_per_source"], [1, 1, 0, 0, 0, 0, 0, 0])
        self.assertEqual(identity["provenance"]["source_entry_sha256"], planner.digest(entry))
        self.assertFalse(shared["recalibrated"])
        self.assertEqual(shared["evidence_kind"], "legacy_proxy")
        self.assertFalse(shared["performance_validated"])
        self.assertEqual(shared["chunk_order"], [list(reversed(range(8)))] * 8)

    def test_control_modes_preserve_shared_order(self):
        legacy = planner.import_phase6_plan(legacy_entry(), "attention", ordering="shared")
        before = planner.plan_digest(legacy)
        for mode in ("bf16", "all_remote", "random_matched"):
            derived = planner.with_mask_mode(legacy, mode, seed=9)
            self.assertEqual(derived["chunk_order"], legacy["chunk_order"])
            self.assertEqual(derived["order_sha256"], legacy["order_sha256"])
            self.assertEqual(derived["selected_per_source"], [0] * 8 if mode == "bf16" else
                             ([448] * 8 if mode == "all_remote" else legacy["selected_per_source"]))
        self.assertEqual(planner.plan_digest(legacy), before)

    def test_import_rejects_local_quantization_role_and_bad_coords(self):
        entry = legacy_entry()
        with self.assertRaises(ValueError):
            planner.import_phase6_plan(entry, "mlp")
        entry["mask"][0][0] = 1
        with self.assertRaisesRegex(ValueError, "local"):
            planner.import_phase6_plan(entry, "attention")
        entry = legacy_entry()
        entry["coords"][0][0] = entry["coords"][0][1]
        with self.assertRaisesRegex(ValueError, "bijections"):
            planner.import_phase6_plan(entry, "attention", ordering="shared")


if __name__ == "__main__":
    unittest.main()
