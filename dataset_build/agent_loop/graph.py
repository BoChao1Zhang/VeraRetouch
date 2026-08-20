"""LangGraph source thread, nested global branches, repair, join, and commit."""
from __future__ import annotations

import hashlib
import itertools
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .artifacts import ArtifactStore
from .candidates import (
    LutCatalog, allocate_role_packets, build_local_packets, build_mask_bank,
    mask_summary, palette_summary, region_descriptor,
)
from .config import AgentLoopConfig
from .models import (
    BranchOutput, GlobalBranchState, LOCAL_DELTA_E_TARGETS, LocalLeafInput, MainState,
    local_target_center, luma_capped, target_center,
)
from .persistence import AuditStore
from .prompts import global_request, local_request
from .render import (
    LocalDirectionProbe, MaskReachProbe, RenderError, StrengthCalibrator,
    measure_subject_headroom,
)
from .source_reach import configured_lut_loader
from .responses import CachedResponsesClient
from .scheduler import TerraLane, TerraLimiter, TerraRouter
from .source_annotations import source_content_hash, validate_source_annotation
from .validator import ChainValidator


@dataclass(slots=True)
class AgentServices:
    config: AgentLoopConfig
    artifacts: ArtifactStore
    audit: AuditStore
    terra: CachedResponsesClient
    catalog: LutCatalog
    calibrator: StrengthCalibrator
    validator: ChainValidator | None
    limiter: TerraLimiter
    landing: Any | None = None
    router: TerraRouter | None = None


def stable_id(prefix: str, *parts: object) -> str:
    payload = "\x1f".join(map(str, parts)).encode("utf-8")
    return f"{prefix}_" + hashlib.sha256(payload).hexdigest()[:24]


def _source_run_revision(config: AgentLoopConfig, pass_index: int) -> str:
    return f"{config.thread_revision}:pass-{int(pass_index)}"


def terra_lane(services: AgentServices, route_key: str) -> TerraLane:
    """Fixed per-source lane. Never route per request: the hash carries the identity."""
    if services.router is None:
        return TerraLane(
            index=0, identity=services.config.terra.identity,
            endpoint=services.config.terra, client=services.terra,
            limiter=services.limiter,
        )
    return services.router.lane_for(str(route_key))


def _request_with_limit(
    services: AgentServices, source_key: str, stage: str, spec: Any,
    validate_extra: Any | None = None, lane: TerraLane | None = None,
):
    lane = lane or terra_lane(services, source_key)
    try:
        with lane.limiter.slot(source_key, stage, spec.prompt_cache_key):
            result = lane.client.request(spec, validate_extra)
        lane.limiter.observe(None)
        return result
    except Exception as exc:
        error = getattr(exc, "error_type", None) or getattr(exc, "code", None) \
            or type(exc).__name__
        lane.limiter.observe(str(error))
        raise


def _record_request_context(
    services: AgentServices, state: Mapping[str, Any], result: Any, stage: str
) -> None:
    services.audit.record_request_context(
        result.request_hash, services.config.campaign_id,
        str(state["source_sha256"]), stage, bool(result.cache_hit),
    )


def _prepare_node(services: AgentServices):
    def prepare(state: MainState) -> dict[str, Any]:
        source_path = str(state["source_path"])
        subject_path = str(state["subject_path"])
        source_sha = str(state.get("source_sha256") or source_content_hash(source_path))
        subject_sha = str(
            state.get("subject_sha256") or source_content_hash(subject_path)
        )
        pass_index = int(state.get("pass_index", 0))
        if pass_index < 0:
            raise ValueError("pass_index must be non-negative")
        thread_id = str(
            state.get("thread_id")
            or services.config.thread_id(source_sha, subject_sha, pass_index)
        )
        annotation = validate_source_annotation(
            state.get("source_annotation"), {
                **state, "source_sha256": source_sha, "subject_sha256": subject_sha,
            }
        )
        diagnosis = dict(annotation["diagnosis"])
        preset_reach = dict(annotation["preset_reach"])
        source_ref, render_ref = services.artifacts.normalize_source_images(
            source_path, retention="audit"
        )
        annotation_ref = services.artifacts.put_json(annotation, retention="audit")
        from PIL import Image

        with Image.open(services.artifacts.path_for(render_ref)) as image:
            render_size = image.size
        mask_diagnostics: list[dict[str, Any]] = []
        masks = build_mask_bank(
            subject_path, render_size=render_size,
            source_id=f"{state['source_id']}:pass-{pass_index}",
            prompt_revision=services.config.prompt_revision, artifacts=services.artifacts,
            include_background=True, diagnostics=mask_diagnostics,
        )
        palette = palette_summary(services.artifacts.path_for(source_ref))
        services.audit.record_source_start({
            "campaign_id": services.config.campaign_id, "source_sha256": source_sha,
            "prompt_revision": _source_run_revision(services.config, pass_index),
            "thread_id": thread_id,
            "source_id": state["source_id"],
        })
        return {
            "source_sha256": source_sha, "subject_sha256": subject_sha,
            "pass_index": pass_index,
            "thread_id": thread_id,
            "source_artifact": source_ref.to_dict(),
            "source_render_artifact": render_ref.to_dict(),
            "source_annotation_artifact": annotation_ref.to_dict(),
            "diagnosis": diagnosis, "preset_reach": preset_reach,
            "palette": palette, "masks": masks,
            "mask_diagnostics": mask_diagnostics, "branches": [],
        }

    return prepare


def _shortlist_node(services: AgentServices):
    def shortlist(state: MainState) -> dict[str, Any]:
        payload = services.catalog.global_shortlist(
            state["diagnosis"], state["palette"], str(state.get("scene") or "unknown"),
            services.config.catalog, state["preset_reach"],
            source_sha256=str(state["source_sha256"]),
        )
        ref = services.artifacts.put_json(payload, retention="audit")
        return {
            "global_shortlist": payload["by_major"],
            "global_shortlist_artifact": ref.to_dict(),
        }

    return shortlist


def _global_propose_node(services: AgentServices):
    def propose(state: MainState) -> dict[str, Any]:
        lane = terra_lane(services, str(state["source_sha256"]))
        spec = global_request(
            lane.endpoint, state["source_artifact"], state["diagnosis"],
            state["global_shortlist"],
            min_proposals=services.config.min_global_proposals,
            max_proposals=services.config.max_global_proposals,
        )

        def validate(parsed: dict[str, Any]) -> str | None:
            major = str(parsed.get("major") or "")
            rows = list(state["global_shortlist"].get(major) or [])
            proposals = parsed.get("proposals") or []
            if not rows:
                return "global_major_outside_shortlist"
            if len(proposals) < services.config.min_global_proposals:
                return "global_count_below_config_minimum"
            indices = [row.get("row_index") for row in proposals]
            if any(not isinstance(index, int) or isinstance(index, bool)
                   or not 0 <= index < len(rows) for index in indices):
                return "global_row_index_out_of_range"
            if len(set(indices)) != len(indices):
                return "global_row_index_not_distinct"
            for proposal in proposals:
                row = rows[int(proposal["row_index"])]
                if proposal.get("bin") not in set(row.get("achievable_bins") or []):
                    return "global_strength_bin_unreachable"
            bins = {row.get("bin") for row in proposals}
            if len(proposals) >= 2 and len(bins) < 2:
                return "global_strength_bin_diversity"
            return None

        result = _request_with_limit(
            services, state["thread_id"], "global_propose", spec, validate, lane
        )
        _record_request_context(services, state, result, "global_propose")
        parsed = result.response["parsed"]
        major = str(parsed["major"])
        rows = list(state["global_shortlist"].get(major) or [])
        proposals = _enrich_global_proposals(
            parsed["proposals"][:services.config.max_global_proposals], rows
        )
        for proposal in proposals:
            services.audit.record_proposal_audit({
                "campaign_id": services.config.campaign_id,
                "source_sha256": str(state["source_sha256"]),
                "branch_id": stable_id(
                    "global", state["source_sha256"], proposal["proposal_id"],
                    proposal["preset_id"], proposal["strength_bin"],
                ),
                "level": "global", "preset_id": proposal["preset_id"],
                "scorer_top1": int(proposal["scorer_rank_offered"] == 0),
                "scorer_top3": int(proposal["scorer_rank_offered"] < 3),
                "scorer_top1_raw": int(proposal["scorer_rank_raw"] == 0),
                "direction_cosine": None,
            })
        return {"selected_major": major, "global_proposals": proposals}

    return propose


def _enrich_global_proposals(
    proposals: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Resolve pick-by-index answers into the internal preset-keyed proposal."""
    result = []
    for proposal in proposals:
        index = int(proposal["row_index"])
        row = rows[index]
        strength_bin = str(proposal["bin"])
        result.append({
            "proposal_id": f"g{index}-{strength_bin}",
            "preset_id": str(row["preset_id"]),
            "strength_bin": strength_bin,
            "row_index": index,
            "reason_codes": list(proposal.get("reason_codes") or []),
            "scorer_rank_raw": int(row.get("scorer_rank_raw", -1)),
            "scorer_rank_offered": int(row.get("scorer_rank_offered", -1)),
        })
    return result


def _global_branch_graph(services: AgentServices):
    config = services.config

    def calibrate_global(state: GlobalBranchState) -> dict[str, Any]:
        proposal = state["global_proposal"]
        branch_id = stable_id(
            "global", state["source_sha256"], proposal["proposal_id"],
            proposal["preset_id"], proposal["strength_bin"],
        )
        try:
            rendered = services.calibrator.calibrate_global(
                source_sha256=state["source_sha256"], branch_id=branch_id,
                input_artifact=state["source_render_artifact"],
                preset_id=str(proposal["preset_id"]),
                strength_bin=str(proposal["strength_bin"]),
            )
            return {"global_render": rendered, "branch_status": "global_rendered"}
        except RenderError as exc:
            return {"branch_status": "global_rejected",
                    "branch_reject_reason": f"render:{exc.code}"}

    def route_after_global(state: GlobalBranchState) -> str:
        return "build_local_packet" if state.get("branch_status") == "global_rendered" \
            else "finalize_global_branch"

    def _intent_packets(
        state: GlobalBranchState, masks: Sequence[Mapping[str, Any]],
        headroom: Mapping[str, Any], exclude: Iterable[str] = (),
    ) -> dict[str, Any]:
        # B8 item 4 (R1): the mask-conditioned reach gate is measured on global_after,
        # the same image the local calibration starts from.
        probe = MaskReachProbe(
            services.artifacts, services.catalog,
            input_artifact=state["global_render"]["artifact"],
            source_sha256=str(state["source_sha256"]),
            loader=configured_lut_loader(services.config.databuild_config),
        )
        # B11 item 2 (R7.1): the online direction prefilter reads the same pair of
        # images the local prompt now shows, source -> global_after.
        directions = LocalDirectionProbe(
            services.artifacts,
            source_artifact=state["source_render_artifact"],
            global_after_artifact=state["global_render"]["artifact"],
            diagnosis=state["diagnosis"],
        )
        built = build_local_packets(
            services.catalog, masks,
            exclude={str(state["global_proposal"]["preset_id"]), *map(str, exclude)},
            diagnosis=state["diagnosis"], palette=state["palette"],
            scene=str(state.get("scene") or "unknown"),
            source_sha256=str(state["source_sha256"]),
            global_fingerprint=_global_fingerprint(services.catalog, state),
            headroom=headroom, mask_reach=probe, direction_probe=directions,
        )
        # Pre-registered criteria, runtime assertions: both gates must have run.
        if not built.get("mask_reach_applied"):
            raise RuntimeError("mask_reach_gate_not_wired")
        if not built.get("direction_prefilter_applied"):
            raise RuntimeError("direction_prefilter_not_wired")
        return built

    def build_local_packet(state: GlobalBranchState) -> dict[str, Any]:
        first, remaining, role_note = allocate_role_packets(
            state["masks"], str(state["global_proposal"]["proposal_id"]),
            packet_size=config.max_local_proposals,
        )
        headroom = measure_subject_headroom(
            services.artifacts, state["global_render"]["artifact"],
            first[0]["subject_artifact"],
        )
        built = _intent_packets(state, first, headroom)
        result = {
            "local_packet": first, "unused_masks": remaining,
            "role_packet_note": role_note, "subject_headroom": dict(headroom),
            "intent_supply": _intent_supply(built["packets"]),
            "local_intent_packets": built["packets"],
            "local_packet_notes": built["notes"],
            "local_shortlist": built["rows"],
            "local_shortlist_deficits": built["quota_deficits"],
            # B11 item 2 audit column (R7.1): per (mask, intent) domain size, prefilter
            # survivors and reach drops of the online retrieval.
            "local_retrieval": built["retrieval"],
            "repair_count": 0,
        }
        if not built["packets"]:
            result["branch_status"] = "global_rejected"
            result["branch_reject_reason"] = "no_intent_packet"
        return result

    def route_after_packet(state: GlobalBranchState) -> str:
        return "finalize_global_branch" \
            if state.get("branch_status") == "global_rejected" else "local_propose_batch"

    def local_propose_batch(state: GlobalBranchState) -> dict[str, Any]:
        proposal = state["global_proposal"]
        lane = terra_lane(services, str(state["source_sha256"]))
        spec = local_request(
            lane.endpoint, state["source_render_artifact"],
            state["global_render"]["artifact"], state["diagnosis"],
            _public_global(proposal), state["global_render"]["metrics"],
            state["local_shortlist"],
            [mask_summary(row) for row in state["local_packet"]],
            max_proposals=config.max_local_proposals,
            packets=state["local_intent_packets"],
        )
        error = _validate_local_response(
            state, state["local_packet"], state["local_shortlist"],
            state["local_intent_packets"],
        )
        result = _request_with_limit(
            services, state["thread_id"], "local_propose", spec, error, lane
        )
        _record_request_context(services, state, result, "local_propose")
        return {"local_proposals": _enrich_local_proposals(
            result.response["parsed"]["proposals"], state["local_packet"],
            state["local_shortlist"], state["local_intent_packets"],
            int(state.get("repair_count", 0)),
        )}

    def route_local_batch(state: GlobalBranchState):
        if not state.get("local_proposals"):
            return "finalize_global_branch"
        return [Send("calibrate_local_render", _leaf_input(state, row))
                for row in state["local_proposals"]]

    def calibrate_local_render(state: LocalLeafInput) -> dict[str, Any]:
        proposal = state["local_proposal"]
        parent_id = stable_id(
            "global", state["source_sha256"], state["global_proposal"]["proposal_id"],
            state["global_proposal"]["preset_id"], state["global_proposal"]["strength_bin"],
        )
        branch_id = stable_id(
            "local", parent_id, state.get("repair_count", 0), proposal["proposal_id"],
            proposal["preset_id"], proposal["mask_id"],
        )
        intent = str(proposal.get("intent", ""))
        p99_luma = state.get("subject_headroom", {}).get("p99_luma")
        subject_p99_luma = float(p99_luma) if isinstance(p99_luma, (int, float)) \
            and not isinstance(p99_luma, bool) else None
        base = {
            "branch_id": branch_id, "global_branch_id": parent_id,
            "proposal": _public_local(proposal),
            "repair_count": int(state.get("repair_count", 0)),
            "global_strength_bin": state["global_proposal"]["strength_bin"],
            "mask_family": proposal["mask"]["family"],
            "mask_role": proposal.get("mask_role", "subject"),
            "intent": intent,
            "intent_variant": proposal.get("intent_variant", ""),
            "visible_region": proposal.get("visible_region", ""),
            # B8 item 4 audit column (R1.3).
            "mask_reach_de": proposal.get("mask_reach_de"),
            # B7 item 2 audit column.
            "luma_capped": luma_capped(intent, subject_p99_luma)
            if intent in LOCAL_DELTA_E_TARGETS else False,
        }
        services.audit.record_proposal_audit({
            "campaign_id": config.campaign_id,
            "source_sha256": str(state["source_sha256"]),
            "branch_id": branch_id, "level": "local",
            "preset_id": str(proposal["preset_id"]),
            "scorer_top1": None, "scorer_top3": None,
            "direction_cosine": _direction_cosine(
                services.catalog, str(state["global_proposal"]["preset_id"]),
                str(proposal["preset_id"]),
            ),
        })
        try:
            rendered = services.calibrator.calibrate_local(
                source_sha256=state["source_sha256"], branch_id=branch_id,
                input_artifact=state["global_render"]["artifact"],
                preset_id=str(proposal["preset_id"]), mask=proposal["mask"],
                strength_bin=str(proposal["strength_bin"]), intent=intent,
                subject_p99_luma=subject_p99_luma,
            )
        except RenderError as exc:
            leaf = {**base, "status": "render_rejected",
                    "reject_reason": f"render:{exc.code}", "validation": {"passed": False,
                    "defects": [{"defect_code": "render_failure", "location": "image",
                                 "evidence": exc.code, "confidence": 1.0}]}}
            _record_local_branch(services, state, leaf)
            return {"leaves": [leaf]}
        if not config.validator_enabled:
            leaf = {
                **base, "status": "validator_skipped", "render": rendered,
                "validation": {
                    "passed": None, "skipped": True, "mode": "disabled", "defects": [],
                },
            }
            _record_local_branch(services, state, leaf)
            return {"leaves": [leaf]}
        if services.validator is None:
            raise RuntimeError("validator_enabled_but_unavailable")
        validation = services.validator.validate(
            source_sha256=state["source_sha256"], branch_id=branch_id,
            source=state["source_artifact"],
            global_after=state["global_render"]["artifact"],
            final_after=rendered["artifact"],
            # Validator prose stays the human-readable hint; the machine-side
            # `visible_region` key on the leaf is the low-granularity region key.
            assignment={"visible_region": proposal["mask"].get("center_hint"),
                        "direction": proposal["mask"].get("direction"),
                        "mask_role": proposal.get("mask_role", "subject"),
                        "intent": proposal.get("intent", ""),
                        "repair_count": int(state.get("repair_count", 0))},
        )
        status = "validator_pass" if validation["passed"] else "validator_failed"
        leaf = {**base, "status": status, "render": rendered,
                "validation": validation}
        _record_local_branch(services, state, leaf)
        return {"leaves": [leaf]}

    def assess_local_batch(state: GlobalBranchState) -> dict[str, Any]:
        if any(_leaf_accepted(row) for row in state.get("leaves", [])):
            return {"branch_status": "formal_global"}
        entire_batch_failed = bool(state.get("leaves")) and not any(
            _leaf_accepted(row) for row in state.get("leaves", [])
        )
        if entire_batch_failed and int(state.get("repair_count", 0)) \
                < config.max_local_repairs and state.get("unused_masks"):
            return {"branch_status": "needs_repair"}
        return {"branch_status": "global_rejected",
                "branch_reject_reason": "no_accepted_local"}

    def route_assessment(state: GlobalBranchState) -> str:
        return {
            "formal_global": "finalize_global_branch",
            "needs_repair": "local_repair",
            "global_rejected": "finalize_global_branch",
        }[state["branch_status"]]

    def local_repair(state: GlobalBranchState) -> dict[str, Any]:
        # Row indices are positions in the flattened intent shortlist, and the repair
        # rebuilds that shortlist, so already tried presets are excluded by preset ID
        # instead of by index.
        used_presets = {
            str(row.get("proposal", {}).get("preset_id", ""))
            for row in state.get("leaves", [])
        } - {""}
        packet = list(state["unused_masks"])[:config.max_local_proposals]
        used_mask_ids = {row["mask_id"] for row in packet}
        remaining = [row for row in state["unused_masks"] if row["mask_id"] not in used_mask_ids]
        defects = [
            defect for leaf in state.get("leaves", [])
            for defect in leaf.get("validation", {}).get("defects", [])
        ]
        repair = {
            "first_proposal_ids": [row.get("proposal", {}).get("proposal_id")
                                   for row in state.get("leaves", [])],
            "defects": defects,
            # B6 item 9: preset IDs never enter the model context. The already-tried
            # presets are excluded retrieval-side by `_intent_packets(exclude=...)`.
            "excluded_mask_ids": sorted(
                row.get("proposal", {}).get("mask_id", "") for row in state.get("leaves", [])
            ),
            "unused_candidates_only": True,
        }
        built = _intent_packets(
            state, packet, state.get("subject_headroom", {}), exclude=used_presets
        )
        if not built["packets"]:
            return {
                "repair_count": int(state.get("repair_count", 0)) + 1,
                "local_packet": packet, "unused_masks": remaining,
                "local_proposals": [], "branch_status": "global_rejected",
                "branch_reject_reason": "no_intent_packet",
            }
        rows = built["rows"]
        supply = _intent_supply(built["packets"])
        lane = terra_lane(services, str(state["source_sha256"]))
        spec = local_request(
            lane.endpoint, state["source_render_artifact"],
            state["global_render"]["artifact"], state["diagnosis"],
            _public_global(state["global_proposal"]), state["global_render"]["metrics"],
            rows,
            [mask_summary(row) for row in packet], max_proposals=config.max_local_proposals,
            packets=built["packets"], repair=repair,
        )
        error = _validate_local_response(state, packet, rows, built["packets"])
        result = _request_with_limit(
            services, state["thread_id"], "local_repair", spec, error, lane
        )
        _record_request_context(services, state, result, "local_repair")
        return {
            "repair_count": int(state.get("repair_count", 0)) + 1,
            "local_packet": packet, "unused_masks": remaining,
            "local_shortlist": rows, "local_intent_packets": built["packets"],
            "local_packet_notes": built["notes"], "intent_supply": supply,
            "local_shortlist_deficits": built["quota_deficits"],
            "local_retrieval": built["retrieval"],
            "local_proposals": _enrich_local_proposals(
                result.response["parsed"]["proposals"], packet, rows,
                built["packets"], int(state.get("repair_count", 0)) + 1,
            ),
        }

    def finalize_global_branch(state: GlobalBranchState) -> dict[str, Any]:
        proposal = state["global_proposal"]
        branch_id = stable_id(
            "global", state["source_sha256"], proposal["proposal_id"],
            proposal["preset_id"], proposal["strength_bin"],
        )
        passed = [row for row in state.get("leaves", []) if _leaf_accepted(row)]
        status = "formal_global" if passed else "global_rejected"
        branch = {
            "branch_id": branch_id, "status": status, "proposal": proposal,
            "global_render": state.get("global_render"),
            "leaves": list(state.get("leaves", [])),
            "reject_reason": state.get("branch_reject_reason", "") if not passed else "",
            "local_shortlist_deficits": list(state.get("local_shortlist_deficits", [])),
            "local_intent_packets": list(state.get("local_intent_packets", [])),
            "local_packet_notes": list(state.get("local_packet_notes", [])),
            "local_retrieval": list(state.get("local_retrieval", [])),
            "role_packet_note": dict(state.get("role_packet_note", {})),
            "subject_headroom": dict(state.get("subject_headroom", {})),
            "intent_supply": int(state.get("intent_supply", 0)),
        }
        services.audit.record_branch({
            "branch_id": branch_id, "campaign_id": config.campaign_id,
            "source_sha256": state["source_sha256"], "parent_id": None,
            "level": "global", "status": status, "proposal": proposal, "result": branch,
        })
        return {"branches": [branch]}

    graph = StateGraph(GlobalBranchState, output_schema=BranchOutput)
    graph.add_node("calibrate_global_render", calibrate_global)
    graph.add_node("build_local_packet", build_local_packet)
    graph.add_node("local_propose_batch", local_propose_batch)
    graph.add_node("calibrate_local_render", calibrate_local_render)
    graph.add_node("assess_local_batch", assess_local_batch)
    graph.add_node("local_repair", local_repair)
    graph.add_node("finalize_global_branch", finalize_global_branch)
    graph.add_edge(START, "calibrate_global_render")
    graph.add_conditional_edges("calibrate_global_render", route_after_global)
    graph.add_conditional_edges("build_local_packet", route_after_packet)
    graph.add_conditional_edges("local_propose_batch", route_local_batch)
    graph.add_edge("calibrate_local_render", "assess_local_batch")
    graph.add_conditional_edges("assess_local_batch", route_assessment)
    graph.add_conditional_edges("local_repair", route_local_batch)
    graph.add_edge("finalize_global_branch", END)
    return graph.compile()


def _intent_supply(packets: Sequence[Mapping[str, Any]]) -> int:
    """Audit column: how many distinct intents this branch could offer at all."""
    return len({str(row["intent"]) for row in packets})


def _global_fingerprint(
    catalog: LutCatalog, state: Mapping[str, Any]
) -> dict[str, float]:
    return catalog.get(str(state["global_proposal"]["preset_id"])).fingerprint()


def _validate_local_response(
    state: GlobalBranchState, packet: Sequence[Mapping[str, Any]],
    shortlist: Sequence[Mapping[str, Any]],
    intent_packets: Sequence[Mapping[str, Any]] = (),
):
    allowed_masks = {str(row["mask_id"]) for row in packet}
    rows = list(shortlist)
    offered = {
        (str(row["mask_id"]), str(row["intent"])):
        {int(index) for index in row["row_indices"]} for row in intent_packets
    }
    global_preset = str(state["global_proposal"]["preset_id"])

    def validate(parsed: dict[str, Any]) -> str | None:
        proposals = parsed.get("proposals") or []
        masks = [str(row.get("mask_id")) for row in proposals]
        if len(set(masks)) != len(masks):
            return "local_sibling_mask_duplicate"
        if any(str(row.get("mask_id")) not in allowed_masks for row in proposals):
            return "local_mask_outside_packet"
        indices = [row.get("row_index") for row in proposals]
        if any(not isinstance(index, int) or isinstance(index, bool)
               or not 0 <= index < len(rows) for index in indices):
            return "local_row_index_out_of_range"
        keys = [(str(row.get("mask_id")), str(row.get("intent"))) for row in proposals]
        if any(key not in offered for key in keys):
            return "local_intent_not_offered"
        if any(int(row["row_index"]) not in offered[key]
               for row, key in zip(proposals, keys)):
            return "local_row_index_outside_intent_packet"
        # Contract C3: the submitted sibling set must cover at least two intents.
        # B6 item 8: when the branch only supplies one intent the requirement drops
        # to that supply instead of rejecting an answer that cannot be given.
        required = min(2, len({intent for _mask, intent in offered}))
        if len(proposals) >= 2 and len({key[1] for key in keys}) < required:
            return "local_intent_diversity"
        # Defense in depth: `build_local_packets(exclude=(global preset,))` already
        # drops the parent preset, so this can only fire if that wiring regresses.
        if any(str(rows[int(index)]["preset_id"]) == global_preset for index in indices):
            return "global_local_preset_equal"
        for proposal in proposals:
            row = rows[int(proposal["row_index"])]
            if "achievable_bins" not in row:
                continue
            if str(proposal.get("bin")) not in set(row.get("achievable_bins") or []):
                return "local_strength_bin_unreachable"
        return None

    return validate


def _enrich_local_proposals(
    proposals: Sequence[Mapping[str, Any]], packet: Sequence[Mapping[str, Any]],
    shortlist: Sequence[Mapping[str, Any]],
    intent_packets: Sequence[Mapping[str, Any]] = (), repair_count: int = 0,
) -> list[dict[str, Any]]:
    masks = {str(row["mask_id"]): dict(row) for row in packet}
    rows = list(shortlist)
    variants = {str(row["intent"]): str(row["intent_variant"]) for row in intent_packets}
    result = []
    for proposal in proposals:
        index = int(proposal["row_index"])
        row = rows[index]
        mask = masks[str(proposal["mask_id"])]
        intent = str(proposal.get("intent") or "")
        result.append({
            "proposal_id": f"l{repair_count}-{index}-{mask['mask_id']}",
            "preset_id": str(row["preset_id"]),
            "mask_id": str(mask["mask_id"]),
            "strength_bin": str(proposal["bin"]),
            "row_index": index,
            "intent": intent,
            "intent_variant": variants.get(intent, intent),
            "mask_role": str(mask.get("role") or "subject"),
            "reason_codes": list(proposal.get("reason_codes") or []),
            "scorer_rank_raw": int(row.get("scorer_rank_raw", -1)),
            "scorer_rank_offered": int(row.get("scorer_rank_offered", -1)),
            "visible_region": region_descriptor(mask),
            # B8 item 4 audit column: mask-conditioned reachable dE of this row.
            "mask_reach_de": row.get("mask_reach_de"),
            "mask": mask,
        })
    return result


def _public_global(proposal: Mapping[str, Any]) -> dict[str, Any]:
    """What the local stage sees about the committed global edit (no preset IDs)."""
    return {
        "bin": proposal.get("strength_bin"),
        "reason_codes": list(proposal.get("reason_codes") or []),
    }


def _direction_cosine(catalog: LutCatalog, global_preset: str, local_preset: str) -> float | None:
    try:
        left = catalog.get(global_preset).direction_vector()
        right = catalog.get(local_preset).direction_vector()
    except Exception:  # audit-only column must never break a chain
        return None
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    dot = sum(a * b for a, b in zip(left, right))
    return round(dot / (left_norm * right_norm), 6)


def _leaf_input(state: GlobalBranchState, proposal: Mapping[str, Any]) -> LocalLeafInput:
    keys = (
        "source_id", "source_sha256", "thread_id", "source_artifact",
        "source_render_artifact", "diagnosis", "selected_major", "global_proposal",
        "global_render", "repair_count",
    )
    result: LocalLeafInput = {key: state[key] for key in keys if key in state}  # type: ignore[literal-required]
    result["local_proposal"] = dict(proposal)
    # B7 item 2: the luminance_pop soft cap reads `p99_luma` off the branch headroom.
    result["subject_headroom"] = dict(state.get("subject_headroom") or {})
    return result


def _public_local(proposal: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in proposal.items() if key != "mask"}


def _record_local_branch(
    services: AgentServices, state: Mapping[str, Any], leaf: Mapping[str, Any]
) -> None:
    services.audit.record_branch({
        "branch_id": leaf["branch_id"], "campaign_id": services.config.campaign_id,
        "source_sha256": state["source_sha256"], "parent_id": leaf["global_branch_id"],
        "level": "local", "status": leaf["status"],
        "proposal": leaf["proposal"], "result": dict(leaf),
    })


def _leaf_accepted(leaf: Mapping[str, Any]) -> bool:
    return leaf.get("status") in {"validator_pass", "validator_skipped"}


def select_committed_leaves(
    branches: Sequence[Mapping[str, Any]], *, minimum: int = 2, maximum: int = 6
) -> tuple[list[dict[str, Any]], list[str]]:
    formal_by_id = {
        str(row["branch_id"]): row for row in branches
        if row.get("status") == "formal_global" and row.get("branch_id")
    }
    formal = list(formal_by_id.values())
    leaf_by_id = {
        str(leaf["branch_id"]): dict(leaf)
        for branch in formal for leaf in branch.get("leaves", [])
        if _leaf_accepted(leaf) and leaf.get("branch_id")
    }
    leaves = list(leaf_by_id.values())
    reasons = []
    if len(leaves) < minimum:
        reasons.append(f"committed_leaves_lt_{minimum}")
    if len({leaf["global_branch_id"] for leaf in leaves}) < 2:
        reasons.append("formal_globals_lt_2")
    if len({leaf["global_strength_bin"] for leaf in leaves}) < 2:
        reasons.append("global_strength_bins_lt_2")
    if reasons:
        return [], reasons

    def deviation(leaf: Mapping[str, Any]) -> float:
        local = float(leaf["render"]["metrics"]["delta_e"])
        global_render = next(
            branch["global_render"] for branch in formal
            if branch["branch_id"] == leaf["global_branch_id"]
        )
        global_de = float(global_render["metrics"]["delta_e"])
        local_bin = str(leaf.get("proposal", {}).get("strength_bin") or "natural")
        # B7 item 1: the local target center is the intent's own ladder, shifted down
        # one bin when this leaf was served under the luminance soft cap (item 2).
        intent = str(leaf.get("intent") or "")
        if intent not in LOCAL_DELTA_E_TARGETS:
            raise ValueError(f"committed leaf carries unknown local intent: {intent!r}")
        center = local_target_center(
            intent, local_bin, capped=bool(leaf.get("luma_capped", False))
        )
        return abs(local - center) \
            + abs(global_de - target_center(leaf["global_strength_bin"]))

    pairs = [
        pair for pair in itertools.combinations(leaves, 2)
        if pair[0]["global_branch_id"] != pair[1]["global_branch_id"]
        and pair[0]["global_strength_bin"] != pair[1]["global_strength_bin"]
    ]
    # B6 item 2: the committed set must carry at least one subject-role leaf. The
    # anchor pair is where it is enforced, since it is always part of the result.
    # Sources whose accepted leaves contain no subject leaf at all (background
    # infeasible / subject side all rejected) are exempt.
    subject_pairs = [
        pair for pair in pairs
        if any(str(row.get("mask_role") or "subject") == "subject" for row in pair)
    ]
    if subject_pairs:
        pairs = subject_pairs
    first = min(pairs, key=lambda pair: (
        sum(deviation(row) for row in pair),
        tuple(sorted(str(row["branch_id"]) for row in pair)),
    ))
    selected = list(first)
    selected_ids = {row["branch_id"] for row in selected}
    while len(selected) < min(maximum, len(leaves)):
        families = {row.get("mask_family") for row in selected}
        intents = {row.get("intent") for row in selected}
        presets = {row["proposal"].get("preset_id") for row in selected}
        regions = {row.get("visible_region") for row in selected}
        globals_ = {row["global_branch_id"] for row in selected}
        bins = {row["global_strength_bin"] for row in selected}

        def rank(row: Mapping[str, Any]):
            diversity = (
                int(row.get("intent") not in intents)
                + int(row.get("mask_family") not in families)
                + int(row["proposal"].get("preset_id") not in presets)
                + int(row.get("visible_region") not in regions)
                + int(row["global_branch_id"] not in globals_)
                + int(row["global_strength_bin"] not in bins)
            )
            return (-diversity, deviation(row), str(row["branch_id"]))

        remaining = [row for row in leaves if row["branch_id"] not in selected_ids]
        if not remaining:
            break
        candidate = min(remaining, key=rank)
        selected.append(candidate)
        selected_ids.add(candidate["branch_id"])
    selected.sort(key=lambda row: str(row["branch_id"]))
    return selected, []


def _commit_select_node(services: AgentServices):
    def commit_select(state: MainState) -> dict[str, Any]:
        selected, reasons = select_committed_leaves(
            state.get("branches", []), minimum=services.config.min_committed_leaves,
            maximum=services.config.max_committed_leaves,
        )
        selected = [
            {**row, "winner_confidence":
             "normal" if row.get("validation", {}).get("passed") is True else "low"}
            for row in selected
        ]
        selected_by_id = {row["branch_id"]: row for row in selected}
        selected_ids = set(selected_by_id)
        for branch in state.get("branches", []):
            for leaf in branch.get("leaves", []):
                if not _leaf_accepted(leaf):
                    continue
                status = "committed" if leaf["branch_id"] in selected_ids else "cap_excluded"
                updated = {
                    **selected_by_id.get(leaf["branch_id"], leaf),
                    "commit_status": status,
                }
                services.audit.record_branch({
                    "branch_id": leaf["branch_id"],
                    "campaign_id": services.config.campaign_id,
                    "source_sha256": state["source_sha256"],
                    "parent_id": leaf["global_branch_id"], "level": "local",
                    "status": leaf["status"], "proposal": leaf["proposal"],
                    "result": updated,
                })
        return {
            "committed_leaves": selected,
            "terminal_status": "accepted" if selected else "source_rejected",
            "reject_reasons": reasons,
        }

    return commit_select


def _route_terminal(state: MainState) -> str:
    return "commit_tree" if state.get("terminal_status") == "accepted" else "source_reject"


def _promote(services: AgentServices, ref: Mapping[str, Any]) -> None:
    services.artifacts.promote(dict(ref))


def _promote_tree_inputs(
    services: AgentServices, state: Mapping[str, Any], protected_sha256: set[str],
) -> None:
    refs: list[Mapping[str, Any]] = []
    for key in ("source_annotation_artifact", "global_shortlist_artifact"):
        ref = state.get(key)
        if isinstance(ref, Mapping):
            refs.append(ref)
    for mask in state.get("masks", []):
        for key in ("alpha_artifact", "subject_artifact"):
            ref = mask.get(key)
            if isinstance(ref, Mapping):
                refs.append(ref)
    for ref in refs:
        _promote(services, ref)
        protected_sha256.add(str(ref["sha256"]))


def _render_refs(render: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    if not render:
        return []
    refs = []
    artifact = render.get("artifact")
    if isinstance(artifact, Mapping):
        refs.append(artifact)
    applied = render.get("applied_alpha_artifact") \
        or (render.get("parameters") or {}).get("applied_alpha_artifact")
    if isinstance(applied, Mapping):
        refs.append(applied)
    return refs


def _discard_uncommitted_renders(
    services: AgentServices, branches: Sequence[Mapping[str, Any]],
    protected_sha256: set[str],
) -> None:
    refs = []
    for branch in branches:
        refs.extend(_render_refs(branch.get("global_render")))
        for leaf in branch.get("leaves", []):
            refs.extend(_render_refs(leaf.get("render")))
    for ref in refs:
        digest = str(ref["sha256"])
        if digest in protected_sha256:
            continue
        # C1b item 12: `protected_sha256` only knows this source's committed rows, but
        # artifacts are content-addressed and dedupe across sources and passes, so an
        # identical render committed by another source used to be unlinked here. The
        # claim is now taken first: `mark_artifact_purged` flips `quarantine` ->
        # `purged` in one conditional statement and returns False for anything that any
        # committed row still holds (`record_artifact` makes `accepted` sticky). Only
        # the winner of that claim unlinks the blob.
        if services.audit.mark_artifact_purged(digest):
            services.artifacts.discard(dict(ref))


def _commit_tree_node(services: AgentServices):
    def commit_tree(state: MainState) -> dict[str, Any]:
        selected_by_id = {
            row["branch_id"]: row for row in state["committed_leaves"]
        }
        selected_ids = set(selected_by_id)
        committed_branches = {
            row["global_branch_id"] for row in state["committed_leaves"]
        }
        _promote(services, state["source_artifact"])
        _promote(services, state["source_render_artifact"])
        protected_sha256 = {
            str(state["source_artifact"]["sha256"]),
            str(state["source_render_artifact"]["sha256"]),
        }
        _promote_tree_inputs(services, state, protected_sha256)
        tree_branches = []
        for branch in state.get("branches", []):
            branch_copy = dict(branch)
            if branch["branch_id"] in committed_branches and branch.get("global_render"):
                _promote(services, branch["global_render"]["artifact"])
                protected_sha256.add(str(branch["global_render"]["artifact"]["sha256"]))
            leaves = []
            for leaf in branch.get("leaves", []):
                copy = dict(leaf)
                if _leaf_accepted(leaf):
                    copy["commit_status"] = "committed" if leaf["branch_id"] in selected_ids \
                        else "cap_excluded"
                if leaf["branch_id"] in selected_ids:
                    copy["winner_confidence"] = selected_by_id[
                        leaf["branch_id"]
                    ]["winner_confidence"]
                    _promote(services, leaf["render"]["artifact"])
                    protected_sha256.add(str(leaf["render"]["artifact"]["sha256"]))
                    applied = leaf["render"].get("applied_alpha_artifact") \
                        or leaf["render"].get("parameters", {}).get("applied_alpha_artifact")
                    if applied:
                        _promote(services, applied)
                        protected_sha256.add(str(applied["sha256"]))
                    _promote(services, _mask_ref(state, leaf))
                leaves.append(copy)
            branch_copy["leaves"] = leaves
            tree_branches.append(branch_copy)
        _discard_uncommitted_renders(
            services, state.get("branches", []), protected_sha256
        )
        manifest = {
            "schema": "local-retouch-edit-tree-v1",
            "campaign_id": services.config.campaign_id,
            "prompt_revision": services.config.thread_revision,
            "thread_id": state["thread_id"], "source_id": state["source_id"],
            "pass_index": int(state.get("pass_index", 0)),
            "source_sha256": state["source_sha256"],
            "source_artifact": state["source_artifact"],
            "source_annotation_artifact": state.get("source_annotation_artifact"),
            "diagnosis": state["diagnosis"], "style_major": state["selected_major"],
            "mask_bank": state["masks"],
            "mask_diagnostics": list(state.get("mask_diagnostics", [])),
            "global_shortlist_artifact": state["global_shortlist_artifact"],
            "branches": tree_branches,
            "committed_leaf_ids": sorted(selected_ids),
            "config": services.config.sanitized_dict(),
        }
        tree_ref = services.artifacts.put_json(manifest, retention="accepted")
        counts = {
            "global_proposals": len(state.get("global_proposals", [])),
            "formal_globals": sum(
                branch.get("status") == "formal_global"
                for branch in state.get("branches", [])
            ),
            "validator_pass_leaves": sum(
                bool(leaf.get("validation", {}).get("passed", False))
                for branch in state.get("branches", []) for leaf in branch.get("leaves", [])
            ),
            "validator_skipped_leaves": sum(
                leaf.get("status") == "validator_skipped"
                for branch in state.get("branches", []) for leaf in branch.get("leaves", [])
            ),
            "committed_leaves": len(selected_ids),
        }
        services.audit.record_source_finish(
            services.config.campaign_id, state["source_sha256"],
            _source_run_revision(services.config, int(state.get("pass_index", 0))),
            "accepted", counts, tree_ref.to_dict(),
        )
        return {"tree_artifact": tree_ref.to_dict()}

    return commit_tree


def _mask_ref(state: MainState, leaf: Mapping[str, Any]) -> Mapping[str, Any]:
    mask_id = leaf["proposal"]["mask_id"]
    return next(row["alpha_artifact"] for row in state["masks"] if row["mask_id"] == mask_id)


def _source_reject_node(services: AgentServices):
    def reject(state: MainState) -> dict[str, Any]:
        manifest = {
            "schema": "local-retouch-edit-tree-v1",
            "terminal_status": "source_rejected", "thread_id": state["thread_id"],
            "source_id": state["source_id"], "source_sha256": state["source_sha256"],
            "pass_index": int(state.get("pass_index", 0)),
            "reject_reasons": state.get("reject_reasons", []),
            "source_artifact": state.get("source_artifact"),
            "source_annotation_artifact": state.get("source_annotation_artifact"),
            "diagnosis": state.get("diagnosis"), "mask_bank": state.get("masks", []),
            "global_shortlist_artifact": state.get("global_shortlist_artifact"),
            "branches": state.get("branches", []),
        }
        ref = services.artifacts.put_json(manifest, retention="audit")
        protected = {
            str(item["sha256"]) for item in (
                state.get("source_artifact"), state.get("source_render_artifact")
            ) if isinstance(item, Mapping)
        }
        _discard_uncommitted_renders(
            services, state.get("branches", []), protected
        )
        services.audit.record_source_finish(
            services.config.campaign_id, state["source_sha256"],
            _source_run_revision(services.config, int(state.get("pass_index", 0))),
            "source_rejected",
            {"global_proposals": len(state.get("global_proposals", [])),
             "committed_leaves": 0,
             "reject_reasons": list(state.get("reject_reasons", []))}, ref.to_dict(),
        )
        return {"tree_artifact": ref.to_dict()}

    return reject


def build_graph(
    services: AgentServices, *, checkpointer: Any = None,
    interrupt_after: Sequence[str] | None = None,
):
    # B11 item 1 (R7.1): the local round cannot be built without the segmented
    # fingerprint table, so a config that forgot to mount it fails at graph build, not
    # three nodes later on the first mask.
    if not services.catalog.segment_fingerprints_mounted:
        raise RuntimeError("segment_fingerprints_not_mounted")
    global_branch = _global_branch_graph(services)

    def fan_out_globals(state: MainState):
        common = {
            key: state[key] for key in (
                "source_id", "source_sha256", "thread_id", "source_artifact",
                "source_render_artifact", "diagnosis", "masks", "selected_major",
                "global_shortlist", "preset_reach", "palette", "scene",
            ) if key in state
        }
        return [Send("global_branch", {**common, "global_proposal": proposal, "leaves": []})
                for proposal in state["global_proposals"]]

    graph = StateGraph(MainState)
    graph.add_node("prepare_source", _prepare_node(services))
    graph.add_node("build_global_shortlist", _shortlist_node(services))
    graph.add_node("global_propose_batch", _global_propose_node(services))
    graph.add_node("global_branch", global_branch)
    graph.add_node("commit_select", _commit_select_node(services))
    graph.add_node("commit_tree", _commit_tree_node(services))
    graph.add_node("source_reject", _source_reject_node(services))
    graph.add_edge(START, "prepare_source")
    graph.add_edge("prepare_source", "build_global_shortlist")
    graph.add_edge("build_global_shortlist", "global_propose_batch")
    graph.add_conditional_edges("global_propose_batch", fan_out_globals)
    graph.add_edge("global_branch", "commit_select")
    graph.add_conditional_edges("commit_select", _route_terminal)
    graph.add_edge("commit_tree", END)
    graph.add_edge("source_reject", END)
    return graph.compile(
        checkpointer=checkpointer, interrupt_after=list(interrupt_after or ()),
        name="local_retouch_agent_loop",
    )


def run_source(
    graph: Any, config: AgentLoopConfig, source: Mapping[str, Any], *, resume: bool = False
) -> dict[str, Any]:
    source_sha = str(source.get("source_sha256") or source_content_hash(str(source["source_path"])))
    subject_sha = str(
        source.get("subject_sha256")
        or source_content_hash(str(source["subject_path"]))
    )
    pass_index = int(source.get("pass_index", 0))
    if pass_index < 0:
        raise ValueError("pass_index must be non-negative")
    thread_id = config.thread_id(source_sha, subject_sha, pass_index)
    invocation = {"configurable": {"thread_id": thread_id}}
    snapshot = None
    try:
        snapshot = graph.get_state(invocation)
    except (AttributeError, ValueError):
        pass
    checkpoint_values = dict(getattr(snapshot, "values", None) or {})
    if checkpoint_values.get("tree_artifact") and checkpoint_values.get(
        "terminal_status"
    ) in {"accepted", "source_rejected"}:
        return checkpoint_values
    snapshot_config = dict(getattr(snapshot, "config", None) or {})
    configurable = dict(snapshot_config.get("configurable") or {})
    has_checkpoint = bool(checkpoint_values) or bool(configurable.get("checkpoint_id"))
    if has_checkpoint:
        result = graph.invoke(None, config=invocation)
    else:
        initial = dict(source)
        initial.update({
            "source_sha256": source_sha, "subject_sha256": subject_sha,
            "pass_index": pass_index, "thread_id": thread_id, "branches": [],
        })
        result = graph.invoke(initial, config=invocation)
    return dict(result)


__all__ = [
    "AgentServices", "build_graph", "run_source", "select_committed_leaves",
    "source_content_hash", "stable_id",
]
