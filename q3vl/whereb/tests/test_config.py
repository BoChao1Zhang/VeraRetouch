"""Every frozen number checked against the protocol text it was quoted from."""

from __future__ import annotations

import torch

from q3vl.whereb import config as C


def test_eight_arms_are_the_protocol_5_3_table():
    assert C.ARM_IDS == ("W01", "W02", "W03", "W04", "W05", "W06", "W07", "W08")
    assert C.ARMS["W01"] == ("MC8-Joint", "band")
    assert C.ARMS["W02"] == ("MC8-Joint", "cband12")
    assert C.ARMS["W03"] == ("MC16-Joint", "band")
    assert C.ARMS["W04"] == ("MC16-Joint", "cband12")
    assert C.ARMS["W05"] == ("MC16-SplitHead", "band")
    assert C.ARMS["W06"] == ("MC16-SplitHead", "cband12")
    assert C.ARMS["W07"] == ("MC16-DualCanvas", "band")
    assert C.ARMS["W08"] == ("MC16-DualCanvas", "cband12")
    # the cartesian product, no arm dropped or added
    assert {(s, r) for s, r in C.ARMS.values()} == {
        (s, r) for s in C.STRUCTURE_IDS for r in C.READOUTS
    }


def test_structures_are_the_protocol_5_2_table():
    assert C.STRUCTURES["MC8-Joint"] == {"canvas": 8, "streams": 1, "pools": 1}
    assert C.STRUCTURES["MC16-Joint"] == {"canvas": 16, "streams": 1, "pools": 1}
    assert C.STRUCTURES["MC16-SplitHead"] == {"canvas": 16, "streams": 1, "pools": 2}
    assert C.STRUCTURES["MC16-DualCanvas"] == {"canvas": 16, "streams": 2, "pools": 2}
    assert 8 * 8 == 64 and 16 * 16 == 256


def test_connector_shape_is_512_6_8_2048():
    c = C.ConnectorConfig()
    assert (c.dim, c.n_blocks, c.n_heads, c.ffn) == (512, 6, 8, 2048)
    assert c.text_dim == 2560          # Qwen3-VL-4B text hidden (config.json)
    assert c.vision_dim == 1024        # vision hidden, merger-pre (config.json)


def test_loss_weights_and_schedule_are_protocol_5_5():
    assert (C.MASK_IOU_W, C.MASK_BCE_W, C.MASK_BF1_W) == (1.00, 0.25, 0.10)
    assert C.STAGE1_FRACTION == 0.30
    assert C.STAGE1_WEIGHTS == {"s": 1.00, "curve": 1.00, "dir": 0.10}
    assert C.STAGE2_WEIGHTS == {"s": 0.25, "curve": 0.25, "dir": 0.05}
    assert C.MASK_WEIGHT == 1.00
    z = torch.linspace(C.CURVE_Z_LO, C.CURVE_Z_HI, C.CURVE_Z_N)
    assert z.numel() == 257 and float(z[0]) == -3.0 and float(z[-1]) == 3.0


def test_optimisation_is_protocol_10_3():
    assert C.LEARNING_RATE == 2.0e-4
    assert C.WEIGHT_DECAY == 0.01
    assert C.WARMUP_RATIO == 0.03
    assert C.SCHEDULER == "cosine"
    assert C.MAX_GRAD_NORM == 1.0
    assert C.PRECISION == "bf16"
    assert C.EFFECTIVE_BATCH == 32
    assert C.EPOCHS == 1.0
    assert C.EVAL_STEPS == C.SAVE_STEPS == 500


def test_gates_are_the_amendment_a5_table():
    """Protocol 5.6 as revised by amendment A-5 (2026-08-05 red lines).

    The full table is asserted in test_a5_criteria.py; this pins the two things
    A-5 is about, so a drift here fails in the config test too.
    """
    keys = [k for k, _, _ in C.GATES]
    assert not any("auc" in k.lower() for k in keys), keys
    assert "grid_boundary_f1_vs_oracle_ratio" in keys
    assert "center_prior_delta_hard_iou" in keys
    assert "center_prior_delta_hard_iou_p" in keys
    assert len(C.GATES) == 10
    first = [k for k, _ in C.SELECTION_ORDER][:3]
    assert first == ["local_soft_iou_median", "grid_boundary_f1", "local_soft_iou_p10"]


def test_grad_accum_keeps_effective_batch_32():
    assert C.TrainConfig(micro_batch=4).grad_accum() == 8
    assert C.TrainConfig(micro_batch=2).grad_accum() == 16
    assert C.TrainConfig(micro_batch=8).grad_accum() == 4


def test_arm_config_exposes_structure_properties():
    a = C.arm_config("W07")
    assert (a.structure, a.readout, a.canvas, a.n_streams, a.n_pools) == (
        "MC16-DualCanvas", "band", 16, 2, 2
    )


def test_where_context_boundary_covers_the_measured_corpus():
    # measured on V_where+V_what+T_final (2711 records): local max 79 body tokens
    # + <where> + </where> = 81 <= 96, and the generation budget is larger still.
    assert C.WHERE_CONTEXT_MAX_TOKENS >= 81
    assert C.GEN_MAX_NEW_TOKENS > C.WHERE_CONTEXT_MAX_TOKENS
