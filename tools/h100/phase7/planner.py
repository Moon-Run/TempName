"""CPU-only Phase 7 plans: shared chunks, physical masks, measured interventions.

This module never infers operator speedup from an arrival timestamp.  Targeted
selection chooses among *whole measured masks*, including joint interventions,
and never adds independently measured gains.  A generated plan is not a validated
performance claim.  See ``validate_evidence`` for the strict JSON input schema.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any


SCHEMA_VERSION = 1
PROTOCOL = "phase7-coordinated-chunks-v1"
MODES = ("bf16", "all_remote", "targeted", "random_matched", "legacy")
SPLITS = ("train", "heldout", "final")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _keys(value: Any, expected: set[str], label: str) -> None:
    require(isinstance(value, dict), f"{label} must be an object")
    require(set(value) == expected,
            f"{label} keys differ: missing={sorted(expected-set(value))}, "
            f"unknown={sorted(set(value)-expected)}")


def _integer(value: Any, label: str, lower: int = 0) -> int:
    require(type(value) is int and value >= lower, f"{label} must be an integer >= {lower}")
    return value


def _finite(value: Any, label: str, lower: float | None = 0.0) -> float:
    require(type(value) in (int, float) and math.isfinite(value), f"{label} must be finite")
    require(lower is None or value >= lower, f"{label} must be >= {lower}")
    return float(value)


def _text(value: Any, label: str) -> None:
    require(isinstance(value, str) and bool(value.strip()), f"{label} must be nonempty text")


def digest(value: Any) -> str:
    """Canonical JSON content hash; NaN and infinity are deliberately rejected."""
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def _sha(value: Any, label: str) -> None:
    require(isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value), f"{label} must be a SHA256")


def _validate_provenance(provenance: Any) -> None:
    _keys(provenance, {"run_id", "runtime_sha256", "input_sha256", "measurement_kind",
                       "clock_domain", "sampling_enabled"}, "provenance")
    _text(provenance["run_id"], "run_id")
    _sha(provenance["runtime_sha256"], "runtime_sha256")
    _sha(provenance["input_sha256"], "input_sha256")
    require(provenance["measurement_kind"] in ("gpu", "rehearsal"), "Unknown measurement kind")
    require(provenance["clock_domain"] == "receiver_relative", "Absolute cross-GPU clocks are unsupported")
    require(type(provenance["sampling_enabled"]) is bool, "sampling_enabled must be boolean")


def default_shape(role: str) -> dict[str, Any]:
    require(role in ("attention", "mlp"), "Unknown role")
    return dict(role=role, world_size=8, M=2048, N=4096,
                K_local=512 if role == "attention" else 2048, tile_m=128, tile_n=128)


def validate_shape(shape: Any) -> None:
    _keys(shape, {"role", "world_size", "M", "N", "K_local", "tile_m", "tile_n"}, "shape")
    require(shape["role"] in ("attention", "mlp"), "Unknown shape role")
    for key in ("world_size", "M", "N", "K_local", "tile_m", "tile_n"):
        _integer(shape[key], f"shape.{key}", 1)
    require(shape["world_size"] >= 2, "A distributed plan requires at least two ranks")
    require(shape["tile_m"] == shape["tile_n"] == 128, "Only complete 128x128 tiles are supported")
    require(shape["M"] % (shape["world_size"] * shape["tile_m"]) == 0,
            "M must contain complete equal destination row partitions")
    require(shape["N"] % shape["tile_n"] == 0, "N must contain complete tiles")


def geometry(shape: dict[str, Any], chunk_tiles: int) -> dict[str, int]:
    validate_shape(shape)
    _integer(chunk_tiles, "chunk_tiles", 1)
    tiles = shape["M"] // shape["tile_m"] * (shape["N"] // shape["tile_n"])
    per = tiles // shape["world_size"]
    require(chunk_tiles in (1, 2, 4, 8, 16, 32, 64), "Unsupported runtime chunk_tiles")
    require(per % chunk_tiles == 0, "chunk_tiles must divide tiles per destination")
    return dict(tiles=tiles, tiles_per_destination=per,
                chunks_per_destination=per // chunk_tiles)


def chunk_physical_tiles(shape: dict[str, Any], chunk_tiles: int,
                         destination: int, chunk: int) -> list[int]:
    g = geometry(shape, chunk_tiles)
    _integer(destination, "destination")
    _integer(chunk, "chunk")
    require(destination < shape["world_size"], "Destination out of range")
    require(chunk < g["chunks_per_destination"], "Chunk out of range")
    start = destination * g["tiles_per_destination"] + chunk * chunk_tiles
    return list(range(start, start + chunk_tiles))


def order_digest(shape: dict[str, Any], chunk_tiles: int, chunk_order: list[list[int]]) -> str:
    return digest(dict(shape=shape, chunk_tiles=chunk_tiles, chunk_order=chunk_order))


def _mask_pairs(pairs: Any, shape: dict[str, Any], g: dict[str, int], label: str) -> list[list[int]]:
    require(isinstance(pairs, list), f"{label} must be an array")
    seen = set()
    for pair in pairs:
        require(isinstance(pair, list) and len(pair) == 2, f"{label}: expected [source,physical_tile]")
        source, tile = pair
        _integer(source, f"{label}.source")
        _integer(tile, f"{label}.physical_tile")
        require(source < shape["world_size"] and tile < g["tiles"], f"{label} out of bounds")
        require(source != tile // g["tiles_per_destination"], f"{label} may only quantize remote tiles")
        require((source, tile) not in seen, f"{label} contains duplicate contributions")
        seen.add((source, tile))
    return sorted(pairs)


def _arrivals(value: Any, shape: dict[str, Any], g: dict[str, int], label: str) -> None:
    world = shape["world_size"]
    require(isinstance(value, list) and len(value) == world, f"{label}: missing destinations")
    for dest in value:
        require(isinstance(dest, list) and len(dest) == g["chunks_per_destination"],
                f"{label}: missing chunks")
        for chunk in dest:
            require(isinstance(chunk, list) and len(chunk) == world, f"{label}: missing sources")
            for arrival in chunk:
                # Signed receiver-relative offsets are legal. Never compare GPU absolute clocks.
                _finite(arrival, label, None)


def validate_evidence(evidence: Any, shape: dict[str, Any], chunk_tiles: int) -> None:
    """Validate provenance and every split before fitting any observations.

    ``ordering_observations`` rows have split/block_id/destination/chunk_id and
    receiver-relative ``arrival_us`` (one per source). ``interventions`` rows are
    paired full-operator trials with measured whole masks, full chunk arrival
    arrays [destination][chunk][source], and diagnostic stage costs. All timings
    share an explicit run/runtime/input identity. Synthetic fixtures must declare
    measurement_kind='rehearsal'; that cannot become a GPU validation result.
    """
    g = geometry(shape, chunk_tiles)
    _keys(evidence, {"schema_version", "shape", "chunk_tiles", "provenance",
                     "ordering_observations", "interventions"}, "evidence")
    require(type(evidence["schema_version"]) is int and evidence["schema_version"] == SCHEMA_VERSION,
            "Unsupported evidence schema")
    validate_shape(evidence["shape"])
    _integer(evidence["chunk_tiles"], "evidence.chunk_tiles", 1)
    require(evidence["shape"] == shape and evidence["chunk_tiles"] == chunk_tiles,
            "Evidence geometry mismatch")
    provenance = evidence["provenance"]
    _validate_provenance(provenance)
    for key in ("ordering_observations", "interventions"):
        require(isinstance(evidence[key], list), f"{key} must be an array")
    splits = {}
    seen = set()
    candidates = {}

    def register(row: dict[str, Any], identity: tuple) -> None:
        require(row["split"] in SPLITS, "Unknown observation split")
        _text(row["block_id"], "block_id")
        require(splits.setdefault(row["block_id"], row["split"]) == row["split"],
                "A paired block cannot occur in more than one data split")
        require(identity not in seen, "Duplicate observation")
        seen.add(identity)

    for row in evidence["ordering_observations"]:
        _keys(row, {"split", "block_id", "destination", "chunk_id", "arrival_us"}, "ordering row")
        _integer(row["destination"], "destination")
        _integer(row["chunk_id"], "chunk_id")
        require(row["destination"] < shape["world_size"] and row["chunk_id"] < g["chunks_per_destination"],
                "Ordering observation coordinate out of range")
        require(isinstance(row["arrival_us"], list) and len(row["arrival_us"]) == shape["world_size"],
                "Ordering observation must contain all sources")
        for value in row["arrival_us"]:
            _finite(value, "arrival_us", None)
        register(row, ("order", row["block_id"], row["destination"], row["chunk_id"]))
    for row in evidence["interventions"]:
        _keys(row, {"split", "block_id", "candidate_id", "order_sha256", "recalibrated_after_order",
                    "mask", "baseline_operator_us", "candidate_operator_us", "baseline_arrivals_us",
                    "candidate_arrivals_us", "encode_us", "decode_us", "schedule_us", "max_abs_error"},
              "intervention row")
        _text(row["candidate_id"], "candidate_id")
        _sha(row["order_sha256"], "order_sha256")
        require(type(row["recalibrated_after_order"]) is bool, "recalibrated_after_order must be boolean")
        pairs = _mask_pairs(row["mask"], shape, g, "intervention mask")
        identity = (row["order_sha256"], pairs)
        require(candidates.setdefault(row["candidate_id"], identity) == identity,
                "A candidate ID cannot refer to different masks or orders")
        for key in ("baseline_operator_us", "candidate_operator_us", "encode_us", "decode_us", "schedule_us", "max_abs_error"):
            _finite(row[key], key)
        require(row["baseline_operator_us"] > 0 and row["candidate_operator_us"] > 0,
                "Full operator durations must be positive")
        _arrivals(row["baseline_arrivals_us"], shape, g, "baseline_arrivals_us")
        _arrivals(row["candidate_arrivals_us"], shape, g, "candidate_arrivals_us")
        register(row, ("intervention", row["block_id"], row["candidate_id"]))


def _shared_order(shape: dict[str, Any], chunk_tiles: int, ordering: str,
                  evidence: dict[str, Any] | None) -> list[list[int]]:
    g = geometry(shape, chunk_tiles)
    order = [list(range(g["chunks_per_destination"])) for _ in range(shape["world_size"])]
    if ordering == "identity" or evidence is None:
        return order
    observations = evidence["ordering_observations"]
    for destination in range(shape["world_size"]):
        scores = {}
        for chunk in order[destination]:
            values = [max(row["arrival_us"]) for row in observations
                      if row["split"] == "train" and row["destination"] == destination
                      and row["chunk_id"] == chunk]
            if values:
                scores[chunk] = statistics.median(values)
        # A partially observed destination cannot safely rank unseen chunks.
        if len(scores) == g["chunks_per_destination"]:
            order[destination].sort(key=lambda chunk: (-scores[chunk], chunk))
    return order


def _schedules(shape: dict[str, Any], chunk_order: list[list[int]]) -> list[list[dict[str, int]]]:
    world = shape["world_size"]
    count = len(chunk_order[0])
    # Preserve Phase 6 interleaved production: local destination first, then
    # rank-relative destinations. This does not specify the reduction order.
    return [[dict(destination=(source + offset) % world,
                  chunk_id=chunk_order[(source + offset) % world][position])
             for position in range(count) for offset in range(world)] for source in range(world)]


def make_plan(shape: dict[str, Any], *, chunk_tiles: int = 32, ordering: str = "identity",
              mode: str = "bf16", evidence: dict[str, Any] | None = None,
              budget_per_source: list[int] | None = None, max_abs_error: float | None = None,
              seed: int = 0, matched_plan: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a static plan; no files, GPU work, or held-out fitting occur here.

    For ``targeted`` the budget and error threshold are required. Only complete
    measured candidate masks passing both are eligible; an empty BF16 fallback
    is valid. ``random_matched`` samples all remote physical tiles, preserving
    per-source counts, and explicitly does not inherit the target's error result.
    """
    g = geometry(shape, chunk_tiles)
    require(ordering in ("identity", "shared"), "Unknown ordering")
    require(mode in MODES and mode != "legacy", "Use import_phase6_plan for legacy masks")
    _integer(seed, "seed")
    if evidence is not None:
        validate_evidence(evidence, shape, chunk_tiles)
    world = shape["world_size"]
    remote = g["tiles"] - g["tiles_per_destination"]
    budgets = [remote] * world if budget_per_source is None else list(budget_per_source)
    require(len(budgets) == world, "Missing per-source budgets")
    for value in budgets:
        _integer(value, "budget")
        require(value <= remote, "Budget exceeds remote physical tile count")
    if max_abs_error is not None:
        _finite(max_abs_error, "max_abs_error")
    if mode == "targeted":
        require(budget_per_source is not None and max_abs_error is not None,
                "targeted requires explicit per-source budgets and max_abs_error")
        require(evidence is not None and any(row["split"] == "train" for row in evidence["interventions"]),
                "targeted requires post-order measured training interventions; use explicit bf16 otherwise")
    require((matched_plan is not None) == (mode == "random_matched"),
            "matched_plan is required only for random_matched")
    order = _shared_order(shape, chunk_tiles, ordering, evidence)
    order_sha = order_digest(shape, chunk_tiles, order)
    mask = [[False] * g["tiles"] for _ in range(world)]
    fit_blocks = set()
    if ordering == "shared" and evidence is not None:
        fit_blocks.update(row["block_id"] for row in evidence["ordering_observations"] if row["split"] == "train")
    selection = dict(candidate_id=None, training_median_gain_us=None, training_pairs=0,
                     numerical_status="not_measured", rationale="zero-mask control")
    if mode == "all_remote":
        for source in range(world):
            for tile in range(g["tiles"]):
                mask[source][tile] = tile // g["tiles_per_destination"] != source
        require(all(budget >= remote for budget in budgets), "all_remote exceeds requested budget")
        selection["rationale"] = "full-remote quantization control; quality requires independent measurement"
    elif mode == "targeted":
        candidates = {}
        for row in [] if evidence is None else evidence["interventions"]:
            if row["split"] != "train":
                continue
            fit_blocks.add(row["block_id"])
            # Old BF16/source-order measurements cannot silently fit a new order.
            require(row["order_sha256"] == order_sha and row["recalibrated_after_order"],
                    "Targeted train intervention must be remeasured after the exact frozen order")
            candidates.setdefault(row["candidate_id"], []).append(row)
        eligible = []
        for candidate, rows in candidates.items():
            pairs = rows[0]["mask"]
            counts = [sum(pair[0] == source for pair in pairs) for source in range(world)]
            if any(count > budget for count, budget in zip(counts, budgets)):
                continue
            if any(row["max_abs_error"] > max_abs_error for row in rows):
                continue
            gain = statistics.median(row["baseline_operator_us"] - row["candidate_operator_us"] for row in rows)
            if gain > 0:
                eligible.append((gain, candidate, rows))
        if eligible:
            gain, candidate, rows = sorted(eligible, key=lambda item: (-item[0], item[1]))[0]
            for source, tile in rows[0]["mask"]:
                mask[source][tile] = True
            selection = dict(candidate_id=candidate, training_median_gain_us=gain,
                             training_pairs=len(rows), numerical_status="training_constraint_passed",
                             rationale="best measured complete mask; stage costs already included in operator time")
        else:
            selection["rationale"] = "no eligible positive measured net gain; explicit zero-mask fallback"
    elif mode == "random_matched":
        validate_plan(matched_plan)
        require(matched_plan["shape"] == shape and matched_plan["chunk_tiles"] == chunk_tiles,
                "Random control geometry mismatch")
        require(matched_plan["chunk_order"] == order, "Random control must use the same frozen chunk order")
        require(evidence is None or matched_plan["evidence_sha256"] == digest(evidence),
                "Random control must retain its target's fitting evidence")
        rng = random.Random(seed)
        for source, count in enumerate(matched_plan["selected_per_source"]):
            require(count <= budgets[source], "Matched random count exceeds budget")
            eligible_tiles = [tile for tile in range(g["tiles"]) if tile // g["tiles_per_destination"] != source]
            for tile in rng.sample(eligible_tiles, count):
                mask[source][tile] = True
        fit_blocks.update(matched_plan["fit_block_ids"])
        selection["rationale"] = "seeded remote mask with per-source counts matched; numerical quality unmeasured"
    evidence_kind = "none" if evidence is None else evidence["provenance"]["measurement_kind"]
    evidence_sha = None if evidence is None else digest(evidence)
    provenance = None if evidence is None else dict(evidence["provenance"])
    if mode == "random_matched":
        # Counts themselves were selected using the target evidence. Preserve
        # that lineage, including rehearsal status, even without fresh arrivals.
        evidence_kind = matched_plan["evidence_kind"]
        evidence_sha = matched_plan["evidence_sha256"]
        provenance = None if matched_plan["provenance"] is None else dict(matched_plan["provenance"])
    plan = dict(schema_version=SCHEMA_VERSION, protocol=PROTOCOL, shape=dict(shape), chunk_tiles=chunk_tiles,
                ordering=ordering, chunk_order=order, order_sha256=order_sha,
                source_schedules=_schedules(shape, order), quantization_mode=mode,
                quantize_mask=mask, selected_per_source=[sum(row) for row in mask],
                budget_per_source=budgets, max_abs_error=max_abs_error, seed=seed,
                selection=selection, fit_block_ids=sorted(fit_blocks),
                evidence_sha256=evidence_sha, provenance=provenance,
                matched_plan_sha256=None if matched_plan is None else plan_digest(matched_plan),
                evidence_kind=evidence_kind, recalibrated=mode == "targeted", performance_validated=False)
    validate_plan(plan)
    return plan


def validate_plan(plan: Any) -> None:
    _keys(plan, {"schema_version", "protocol", "shape", "chunk_tiles", "ordering", "chunk_order",
                 "order_sha256", "source_schedules", "quantization_mode", "quantize_mask",
                 "selected_per_source", "budget_per_source", "max_abs_error", "seed", "selection",
                 "fit_block_ids", "evidence_sha256", "provenance", "matched_plan_sha256",
                 "evidence_kind", "recalibrated", "performance_validated"}, "plan")
    require(type(plan["schema_version"]) is int and plan["schema_version"] == SCHEMA_VERSION
            and plan["protocol"] == PROTOCOL, "Unsupported plan protocol/schema")
    shape = plan["shape"]
    g = geometry(shape, plan["chunk_tiles"])
    world = shape["world_size"]
    require(plan["ordering"] in ("identity", "shared"), "Unknown plan ordering")
    order = plan["chunk_order"]
    require(isinstance(order, list) and len(order) == world, "Missing destination chunk order")
    for row in order:
        require(isinstance(row, list) and all(type(v) is int for v in row)
                and sorted(row) == list(range(g["chunks_per_destination"])), "Chunk order must be bijective")
    if plan["ordering"] == "identity":
        require(all(row == list(range(g["chunks_per_destination"])) for row in order), "Identity ordering changed")
    require(plan["order_sha256"] == order_digest(shape, plan["chunk_tiles"], order), "Order digest mismatch")
    schedules = plan["source_schedules"]
    require(isinstance(schedules, list) and len(schedules) == world, "Missing source schedules")
    for row in schedules:
        require(isinstance(row, list), "Source schedule must be an array")
        for item in row:
            _keys(item, {"destination", "chunk_id"}, "schedule item")
            _integer(item["destination"], "schedule.destination")
            _integer(item["chunk_id"], "schedule.chunk_id")
    require(schedules == _schedules(shape, order), "Schedules violate destination interleaving/shared order")
    require(plan["quantization_mode"] in MODES, "Unknown plan quantization mode")
    mask = plan["quantize_mask"]
    require(isinstance(mask, list) and len(mask) == world, "Missing mask sources")
    for source, row in enumerate(mask):
        require(isinstance(row, list) and len(row) == g["tiles"] and all(type(v) is bool for v in row),
                "Mask must contain one boolean per physical tile")
        require(not any(row[source*g["tiles_per_destination"]:(source+1)*g["tiles_per_destination"]]),
                "Local contributions cannot be quantized")
    counts = [sum(row) for row in mask]
    require(isinstance(plan["selected_per_source"], list)
            and all(type(value) is int for value in plan["selected_per_source"])
            and plan["selected_per_source"] == counts, "Reported mask counts differ")
    require(isinstance(plan["budget_per_source"], list) and len(plan["budget_per_source"]) == world, "Missing budgets")
    remote = g["tiles"] - g["tiles_per_destination"]
    for count, budget in zip(counts, plan["budget_per_source"]):
        _integer(budget, "budget")
        require(count <= budget <= remote, "Invalid mask budget")
    if plan["quantization_mode"] == "bf16":
        require(not any(counts), "BF16 control must have a zero mask")
    if plan["quantization_mode"] == "all_remote":
        require(all(count == remote for count in counts), "all_remote is incomplete")
    if plan["max_abs_error"] is not None:
        _finite(plan["max_abs_error"], "max_abs_error")
    require(plan["quantization_mode"] != "targeted" or plan["max_abs_error"] is not None,
            "Targeted plan must retain numerical constraint")
    _integer(plan["seed"], "seed")
    _keys(plan["selection"], {"candidate_id", "training_median_gain_us", "training_pairs",
                               "numerical_status", "rationale"}, "selection")
    selection = plan["selection"]
    _integer(selection["training_pairs"], "training_pairs")
    _text(selection["rationale"], "rationale")
    require(selection["numerical_status"] in ("not_measured", "training_constraint_passed"), "Unknown numerical status")
    if selection["candidate_id"] is not None:
        _text(selection["candidate_id"], "candidate_id")
        _finite(selection["training_median_gain_us"], "training_median_gain_us")
        require(plan["quantization_mode"] == "targeted" and selection["training_pairs"] > 0
                and selection["training_median_gain_us"] > 0, "Invalid targeted selection evidence")
    else:
        require(selection["training_median_gain_us"] is None and selection["training_pairs"] == 0,
                "Unselected plan cannot report a fitted gain")
    blocks = plan["fit_block_ids"]
    require(isinstance(blocks, list) and all(isinstance(v, str) and v for v in blocks)
            and blocks == sorted(set(blocks)), "Invalid fit block IDs")
    require(plan["evidence_kind"] in ("none", "gpu", "rehearsal", "legacy_proxy"), "Unknown evidence kind")
    if plan["evidence_kind"] == "none":
        require(plan["evidence_sha256"] is None and plan["provenance"] is None, "Missing evidence identity")
    elif plan["evidence_kind"] == "legacy_proxy":
        _sha(plan["evidence_sha256"], "evidence_sha256")
        _keys(plan["provenance"], {"measurement_kind", "source_entry_sha256", "limitation"}, "legacy provenance")
        require(plan["provenance"]["measurement_kind"] == "legacy_proxy", "Invalid legacy measurement kind")
        require(plan["provenance"]["source_entry_sha256"] == plan["evidence_sha256"], "Legacy source digest differs")
        _text(plan["provenance"]["limitation"], "legacy limitation")
    else:
        _sha(plan["evidence_sha256"], "evidence_sha256")
        provenance = plan["provenance"]
        _validate_provenance(provenance)
        require(provenance["measurement_kind"] == plan["evidence_kind"],
                "Plan provenance differs from evidence kind")
    require((plan["matched_plan_sha256"] is not None) == (plan["quantization_mode"] == "random_matched"),
            "Invalid matched control identity")
    if plan["matched_plan_sha256"] is not None:
        _sha(plan["matched_plan_sha256"], "matched_plan_sha256")
    require(type(plan["recalibrated"]) is bool, "recalibrated must be boolean")
    require(not plan["recalibrated"] or plan["evidence_kind"] in ("gpu", "rehearsal"),
            "Recalibration requires explicit post-order evidence")
    require(plan["quantization_mode"] != "legacy" or plan["evidence_kind"] == "legacy_proxy",
            "Legacy mode requires preserved source entry")
    require(plan["performance_validated"] is False, "Plan construction cannot validate performance")


def plan_digest(plan: dict[str, Any]) -> str:
    validate_plan(plan)
    return digest(plan)


def with_mask_mode(plan: dict[str, Any], mode: str, seed: int = 0) -> dict[str, Any]:
    """Derive a control at exactly the same frozen order without mutating input.

    ``random_matched`` preserves source counts of the supplied plan. Its random
    mask is not claimed to satisfy any inherited numerical measurement.
    """
    validate_plan(plan)
    require(mode in ("legacy", "bf16", "all_remote", "random_matched"), "Unsupported control mode")
    _integer(seed, "seed")
    if mode == "legacy":
        require(plan["quantization_mode"] == "legacy", "Cannot recover a legacy mask from a derived control")
        return json.loads(json.dumps(plan))
    result = json.loads(json.dumps(plan))
    shape = plan["shape"]
    g = geometry(shape, plan["chunk_tiles"])
    world = shape["world_size"]
    mask = [[False] * g["tiles"] for _ in range(world)]
    rng = random.Random(seed)
    for source in range(world):
        remote = [tile for tile in range(g["tiles"]) if tile // g["tiles_per_destination"] != source]
        selected = remote if mode == "all_remote" else (
            rng.sample(remote, plan["selected_per_source"][source]) if mode == "random_matched" else [])
        for tile in selected:
            mask[source][tile] = True
    result.update(quantization_mode=mode, quantize_mask=mask, selected_per_source=[sum(row) for row in mask],
                  budget_per_source=[g["tiles"] - g["tiles_per_destination"]] * world,
                  max_abs_error=None, seed=seed, recalibrated=False,
                  matched_plan_sha256=plan_digest(plan) if mode == "random_matched" else None,
                  selection=dict(candidate_id=None, training_median_gain_us=None, training_pairs=0,
                                 numerical_status="not_measured", rationale=f"{mode} control derived at unchanged frozen order"))
    validate_plan(result)
    return result


def import_phase6_plan(entry: dict[str, Any], role: str, *, chunk_tiles: int = 8,
                       ordering: str = "identity", mode: str = "legacy", seed: int = 0) -> dict[str, Any]:
    """Preserve a frozen Phase 6 physical mask; optionally use a slot-order proxy.

    The legacy entry is content-hashed and never modified. Source ``coords``
    map logical execution positions to physical tiles. For each destination,
    descending max-over-sources chunk-last-slot creates one shared chunk order.
    This is a legacy proxy, not a measured post-reordering arrival model.
    """
    require(isinstance(entry, dict), "Phase 6 entry must be an object")
    shape = default_shape(role)
    for key, value in (("M", shape["M"]), ("N", shape["N"]), ("K_local", shape["K_local"])):
        require(type(entry.get(key)) is int and entry[key] == value, f"Phase 6 {key} differs from requested role")
    require(ordering in ("identity", "shared"), "Unknown imported ordering")
    g = geometry(shape, chunk_tiles)
    world = shape["world_size"]
    require(entry.get("tiles") == g["tiles"], "Phase 6 tile count differs")
    raw_mask = entry.get("mask")
    require(isinstance(raw_mask, list) and len(raw_mask) == world, "Phase 6 mask missing source ranks")
    mask = []
    for source, row in enumerate(raw_mask):
        require(isinstance(row, list) and len(row) == g["tiles"]
                and all(type(value) in (bool, int) and value in (0, 1) for value in row), "Invalid Phase 6 physical mask")
        require(not any(row[source*g["tiles_per_destination"]:(source+1)*g["tiles_per_destination"]]),
                "Phase 6 mask quantizes a local contribution")
        mask.append([bool(value) for value in row])
    order = [list(range(g["chunks_per_destination"])) for _ in range(world)]
    if ordering == "shared":
        coords = entry.get("coords")
        require(isinstance(coords, list) and len(coords) == world, "Shared legacy order requires coords for every source")
        positions = []
        for row in coords:
            require(isinstance(row, list) and all(type(value) is int for value in row)
                    and sorted(row) == list(range(g["tiles"])), "Phase 6 coords must be physical-tile bijections")
            position = [0] * g["tiles"]
            for slot, tile in enumerate(row):
                position[tile] = slot
            positions.append(position)
        for destination in range(world):
            def score(chunk: int) -> int:
                tiles = chunk_physical_tiles(shape, chunk_tiles, destination, chunk)
                return max(position[tile] for position in positions for tile in tiles)
            order[destination].sort(key=lambda chunk: (-score(chunk), chunk))
    result = make_plan(shape, chunk_tiles=chunk_tiles)
    source_sha = digest(entry)
    result.update(ordering=ordering, chunk_order=order, order_sha256=order_digest(shape, chunk_tiles, order),
                  source_schedules=_schedules(shape, order), quantization_mode="legacy", quantize_mask=mask,
                  selected_per_source=[sum(row) for row in mask], evidence_kind="legacy_proxy",
                  evidence_sha256=source_sha, recalibrated=False,
                  provenance=dict(measurement_kind="legacy_proxy", source_entry_sha256=source_sha,
                                  limitation="Physical mask preserved; shared order uses legacy logical-slot proxy, not post-order arrival calibration."),
                  selection=dict(candidate_id=None, training_median_gain_us=None, training_pairs=0,
                                 numerical_status="not_measured", rationale="frozen Phase 6 physical mask preserved for structural ablation"))
    validate_plan(result)
    return result if mode == "legacy" else with_mask_mode(result, mode, seed)


def evaluate_frozen_plan(plan: dict[str, Any], evidence: dict[str, Any], *,
                         frozen_plan_sha256: str, split: str = "heldout") -> dict[str, Any]:
    """Report independent matching trials without modifying/refitting the plan.

    Final evaluation requires GPU provenance and sampling disabled. This is a
    descriptive report, not a statistical declaration of validated speedup.
    """
    require(split in ("heldout", "final"), "Evaluation cannot consume training data")
    require(plan_digest(plan) == frozen_plan_sha256, "Frozen plan digest changed")
    validate_evidence(evidence, plan["shape"], plan["chunk_tiles"])
    prior = plan["provenance"]
    if prior is not None and "runtime_sha256" in prior:
        require(evidence["provenance"]["runtime_sha256"] == prior["runtime_sha256"],
                "Evaluation runtime differs from fitted runtime")
    if split == "final":
        require(evidence["provenance"]["measurement_kind"] == "gpu"
                and evidence["provenance"]["sampling_enabled"] is False,
                "Final evaluation requires GPU measurement with sampling disabled")
        require(plan["evidence_kind"] != "rehearsal", "Rehearsal-fitted plans need genuine GPU calibration first")
    pairs = sorted([source, tile] for source, row in enumerate(plan["quantize_mask"])
                   for tile, enabled in enumerate(row) if enabled)
    rows = [row for row in evidence["interventions"] if row["split"] == split
            and sorted(row["mask"]) == pairs and row["order_sha256"] == plan["order_sha256"]
            and row["recalibrated_after_order"]]
    require(rows, "No independent measurements of this exact frozen plan")
    require(not (set(plan["fit_block_ids"]) & {row["block_id"] for row in rows}),
            "Evaluation reuses a fitting block")
    require(len({row["block_id"] for row in rows}) == len(rows), "Duplicate evaluation paired blocks")
    gains = [row["baseline_operator_us"] - row["candidate_operator_us"] for row in rows]
    return dict(frozen_plan_sha256=frozen_plan_sha256, split=split, paired_blocks=len(rows),
                block_ids=sorted(row["block_id"] for row in rows), median_gain_us=statistics.median(gains),
                paired_gains_us=gains, worst_max_abs_error=max(row["max_abs_error"] for row in rows),
                measurement_kind=evidence["provenance"]["measurement_kind"], performance_validated=False,
                interpretation="descriptive independent observations; no significance or model-quality claim")


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--role", choices=("attention", "mlp"), required=True)
    parser.add_argument("--chunk-tiles", type=int, default=32)
    parser.add_argument("--ordering", choices=("identity", "shared"), default="identity")
    parser.add_argument("--mode", choices=[mode for mode in MODES if mode != "legacy"], default="bf16")
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--matched-plan", type=Path)
    parser.add_argument("--budget-per-source", type=int)
    parser.add_argument("--max-abs-error", type=float)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    shape = default_shape(args.role)
    plan = make_plan(shape, chunk_tiles=args.chunk_tiles, ordering=args.ordering, mode=args.mode,
                     evidence=None if args.evidence is None else json.loads(args.evidence.read_text()),
                     matched_plan=None if args.matched_plan is None else json.loads(args.matched_plan.read_text()),
                     budget_per_source=None if args.budget_per_source is None else [args.budget_per_source] * shape["world_size"],
                     max_abs_error=args.max_abs_error, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation keeps frozen plans reproducible.
    with args.out.open("x") as handle:
        json.dump(plan, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(dict(plan=str(args.out), sha256=plan_digest(plan),
                          selected_per_source=plan["selected_per_source"], performance_validated=False)))


if __name__ == "__main__":
    main()
