"""EPR-028 R1 (G4D) **runner-layer** tests -- CPU only, no GPU process is started.

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=/home/bc/VeraRetouch \\
    /home/bc/envs/q3vl_sft/bin/python -m pytest q3vl/whatb/tests/test_g4d.py -q

Spec: ``experiments/prs/EPR-028_4d-gaussian-conditional-slice/PROPOSAL_R1.md``.
The carrier's own maths (the conditional parameterisation, the log-domain
forward, the gradcheck, step-0 identity, the A2/A3 bit-equality at step 0, the
per-arm ``compose_headline`` contract, ``r_line`` on a single batch) lives in
``test_g4d_r1.py`` and is deliberately **not** repeated here.

What this file pins, all of it in ``scripts/run_g4d_arm.py``
-----------------------------------------------------------
* **the paired-anchor sampler** (R1 §8.1): every colour carries exactly four
  ``s`` anchors, ``0`` and ``1`` are always among them, the two interior ones
  are complementary, and ``n_pairs_s == L * Q * 4``;
* **mining granularity**: hard mining selects whole colour groups.  Selecting on
  the flattened ``(B, Q*4)`` error would split a group and silently drop the
  ``s = 0`` / ``s = 1`` endpoints that ``R_line`` and the endpoint supervision
  depend on;
* **``R_line`` costs no forward** and is (up to float rounding) zero on ``A1``,
  which is linear in ``s`` by construction;
* **the oracle condition** (R1 §6): its embedding is in the optimiser, in the
  generator's lr group, and receives gradient (from step 1 -- the zero-initialised
  output layers make step 0's gradient legitimately zero);
* **the image formation is per arm** (R1 §5): with an all-zero field ``A1``'s
  headline is bit-exactly the input and ``A2``/``A3``'s is **not forced** to be
  -- that colour leakage at ``s ~ 0`` is what R0's outer ``mix_alpha`` hid;
* **the gate ladder** (R1 §9): the three presets resolve to the spec's numbers;
* **the NaN re-check** (R1 §10): a non-finite ``L_rec`` stops the run with a
  non-zero rc and lands ``void_reason`` on disk;
* **the first-row column set**: ``R_line`` is in it and R0's ``L_m4d`` is gone.
"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import pytest
import torch

from q3vl.whatb import queries
from q3vl.whatb.arms import g4d
from q3vl.whatb.lutdata import apply_lut_volume, mix_alpha
from q3vl.whatb.scripts import run_g4d_arm as runner
from q3vl.whatb.splits import DATASET_ROOT

RUNNER_FILE = Path(runner.__file__)
MOUNTED = (DATASET_ROOT / "splits" / "train.index.jsonl").exists()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _args(*extra: str):
    return runner.build_arg_parser().parse_args(["--out", "/tmp/not-written",
                                                 *extra])


def _arm(mode: str, *, n: int = 6, seed: int = 0) -> g4d.G4DArm:
    """A small arm whose non-final generator layers come from a pinned stream."""
    torch.manual_seed(seed)
    return g4d.G4DArm(g4d.G4DConfig(mode=mode, n_gauss=n, cond_dim=8, hidden=16,
                                    init_seed=20260810))


def _sampler(q: int, seed: int = 3) -> queries.QuerySampler:
    return queries.QuerySampler(seed=seed, q=q)


# --------------------------------------------------------------------------- #
# 1. the paired-anchor sampler (R1 §8.1)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("b,q", [(1, 5), (3, 7), (4, 1)])
def test_every_colour_carries_exactly_the_four_s_anchors(b: int, q: int) -> None:
    x, s = runner.paired_anchor_batch(_sampler(q), b, q, device=torch.device("cpu"))
    assert x.shape == (b, q * runner.S_ANCHORS, 3)
    assert s.shape == (b, q * runner.S_ANCHORS)
    assert x.numel() // 3 == s.numel() == b * q * runner.S_ANCHORS

    xg = x.reshape(b, q, runner.S_ANCHORS, 3)
    sg = s.reshape(b, q, runner.S_ANCHORS)
    # the colour is the SAME across the four anchors -- that is what makes the
    # path supervision low-variance and R_line free
    assert torch.equal(xg, xg[:, :, :1, :].expand_as(xg))
    # the two endpoints are exact, not approximate
    assert torch.equal(sg[:, :, 0], torch.zeros(b, q))
    assert torch.equal(sg[:, :, 1], torch.ones(b, q))
    # the two interior anchors are complementary and strictly inside (0, 1)
    u, v = sg[:, :, 2], sg[:, :, 3]
    assert torch.allclose(u + v, torch.ones(b, q), atol=0, rtol=0)
    assert bool((u > 0).all()) and bool((u < 1).all())


def test_n_pairs_s_is_L_times_Q_times_four() -> None:
    b, q = 3, 11
    _, s = runner.paired_anchor_batch(_sampler(q), b, q, device=torch.device("cpu"))
    assert int(s.numel()) == b * q * runner.S_ANCHORS == b * q * 4
    assert runner.S_ANCHORS == 4


def test_the_sampler_uses_its_private_stream_not_the_global_one() -> None:
    """The where-side N1 lesson: a new draw must not shift any other decision."""
    torch.manual_seed(1234)
    before = torch.rand(4)
    torch.manual_seed(1234)
    runner.paired_anchor_batch(_sampler(16), 2, 16, device=torch.device("cpu"))
    after = torch.rand(4)
    assert torch.equal(before, after)


# --------------------------------------------------------------------------- #
# 2. hard mining is per colour group (R1 §8.1)
# --------------------------------------------------------------------------- #
def test_mining_keeps_or_drops_a_colour_group_whole() -> None:
    b, q = 2, 8
    torch.manual_seed(0)
    err = torch.rand(b, q * runner.S_ANCHORS)
    keep = runner.select_hard_colour_groups(err, 0.5, b, q)
    assert keep.shape == (b, q * runner.S_ANCHORS)
    groups = keep.reshape(b, q, runner.S_ANCHORS)
    per_group = groups.sum(-1)
    assert set(per_group.reshape(-1).tolist()) <= {0, runner.S_ANCHORS}, \
        "a selected colour must keep all four of its s anchors and an " \
        "unselected one none -- top-k on the flat (B, Q*4) error splits the group"

    # and the groups it keeps are exactly the top-r by ANCHOR-MEAN error
    group_err = err.reshape(b, q, runner.S_ANCHORS).mean(-1).reshape(-1)
    want = queries.select_hard(group_err, 0.5)
    got = groups[:, :, 0].reshape(-1).nonzero().reshape(-1)
    assert sorted(want.tolist()) == sorted(got.tolist())


def test_mining_ratio_zero_selects_nothing_and_one_selects_everything() -> None:
    b, q = 2, 5
    err = torch.rand(b, q * runner.S_ANCHORS)
    assert not bool(runner.select_hard_colour_groups(err, 0.0, b, q).any())
    assert bool(runner.select_hard_colour_groups(err, 1.0, b, q).all())


def test_mining_never_drops_an_endpoint_on_a_selected_colour() -> None:
    """The failure this granularity exists to stop, stated as an assertion."""
    b, q = 3, 6
    torch.manual_seed(7)
    err = torch.rand(b, q * runner.S_ANCHORS)
    _, s = runner.paired_anchor_batch(_sampler(q), b, q, device=torch.device("cpu"))
    keep = runner.select_hard_colour_groups(err, 0.4, b, q)
    kept_s = s.reshape(b, q, 4)[keep.reshape(b, q, 4)[:, :, 0]]
    if kept_s.numel():
        assert bool((kept_s[:, 0] == 0).all()) and bool((kept_s[:, 1] == 1).all())


# --------------------------------------------------------------------------- #
# 3. R_line (R1 §4.4) -- zero extra forward, and zero on A1
# --------------------------------------------------------------------------- #
def test_r_line_is_zero_on_a1_and_reads_the_anchors_of_the_same_batch() -> None:
    b, q = 2, 32
    arm = _arm("A1").to(dtype=torch.float64)
    x, s = runner.paired_anchor_batch(_sampler(q), b, q,
                                      device=torch.device("cpu"),
                                      dtype=torch.float64)
    params = arm.theta(torch.randn(b, 8, dtype=torch.float64))
    y = arm.transform(params, x, s)
    line = runner.r_line_from_anchors(y, s, b, q)
    # A1 is f(x,s) = (1-s) x + s T(x): structurally on the chord.
    assert float(line.detach()) < 1e-12

    # and the chord is built from the batch's OWN anchors, so no second forward
    yg = y.reshape(b, q, 4, 3)
    sg = s.reshape(b, q, 4)
    manual = 0.5 * (g4d.r_line(yg[:, :, 2], yg[:, :, 0], yg[:, :, 1], sg[:, :, 2])
                    + g4d.r_line(yg[:, :, 3], yg[:, :, 0], yg[:, :, 1], sg[:, :, 3]))
    assert torch.equal(line, manual)


def test_r_line_is_positive_on_a3_with_a_live_cross_term() -> None:
    b, q = 2, 24
    arm = _arm("A3").to(dtype=torch.float64)
    with torch.no_grad():                       # beta is zero-initialised
        arm.generator.head_beta[-1].bias.add_(0.4)
        arm.generator.head_color[-1].bias.add_(0.2)
    x, s = runner.paired_anchor_batch(_sampler(q), b, q,
                                      device=torch.device("cpu"),
                                      dtype=torch.float64)
    y = arm.transform(arm.theta(torch.randn(b, 8, dtype=torch.float64)), x, s)
    assert float(runner.r_line_from_anchors(y, s, b, q).detach()) > 0.0


# --------------------------------------------------------------------------- #
# 4. the oracle condition (R1 §2.2 / §6)
# --------------------------------------------------------------------------- #
def test_oracle_store_is_indexed_by_lut_id_and_refuses_a_control() -> None:
    store = runner.OracleConditionStore(["b", "a", "c"], cond_dim=8, seed=1)
    assert store.lut_ids == ["b", "a", "c"]     # insertion order, not re-sorted
    z = store.get(["a", "c", "a"])
    assert z.shape == (3, 8)
    assert torch.equal(z[0], z[2]) and not torch.equal(z[0], z[1])
    with pytest.raises(KeyError, match="outside this run's oracle pool"):
        store.get(["nope"])
    with pytest.raises(KeyError, match="negative controls"):
        store.get(["a"], "shuffle")
    facts = store.facts()
    assert facts["n_lut"] == 3 and facts["cond_dim"] == 8
    assert facts["n_params"] == 3 * 8 and facts["publishable"] is False


def test_oracle_embedding_is_in_the_optimiser_and_gets_gradient() -> None:
    args = _args("--base-lr", "1e-3", "--pi-lr-scale", "0.1")
    arm = _arm("A3", n=4)
    store = runner.OracleConditionStore(["l0", "l1", "l2"], cond_dim=8, seed=2)
    groups = runner.build_param_groups(arm, store, args)
    by_name = {g["name"]: g for g in groups}

    assert "cond_oracle" in by_name, "the oracle embedding must be trained"
    # R1 §6: the generator's lr, NOT the pi(z_color) scale
    assert by_name["cond_oracle"]["lr"] == args.lr == 1e-3
    assert by_name["pi"]["lr"] == 1e-4
    ids = {id(p) for p in by_name["cond_oracle"]["params"]}
    assert ids == {id(store.embedding.weight)}

    opt = torch.optim.Adam(groups)
    lut_ids = ["l0", "l1", "l2"]
    x, s = runner.paired_anchor_batch(_sampler(8), 3, 8, device=torch.device("cpu"))
    y = g4d.target_4d(x, s, torch.rand_like(x))

    def _step() -> float:
        opt.zero_grad(set_to_none=True)
        y_hat, aux = arm.transform(arm.theta(store.get(lut_ids)), x, s,
                                   return_aux=True)
        loss, _ = g4d.total_loss(y_hat, y, aux.opacity)
        loss.backward()
        g = store.embedding.weight.grad
        val = 0.0 if g is None else float(g.abs().sum())
        opt.step()
        return val

    # the zero-initialised output layers give dL/dh = W_last^T delta = 0, so the
    # condition legitimately sees no gradient on step 0 and gradient after it.
    assert _step() == 0.0
    assert _step() > 0.0


def test_oracle_batch_source_sees_the_whole_pool_when_it_is_small() -> None:
    store = runner.OracleConditionStore([f"l{i}" for i in range(4)], cond_dim=8)
    one = runner.OracleBatchSource(store, b_samples=4, seed=0)
    lut_ids, rows, z = one.batch(torch.device("cpu"))
    assert lut_ids == store.lut_ids and rows is None and z.shape == (4, 8)
    assert one.facts()["whole_pool_every_step"] is True

    big = runner.OracleConditionStore([f"l{i}" for i in range(10)], cond_dim=8)
    src = runner.OracleBatchSource(big, b_samples=4, seed=0)
    seen = {tuple(src.batch(torch.device("cpu"))[0]) for _ in range(3)}
    assert len(seen) > 1, "a pool larger than the batch must actually rotate"
    assert src.facts()["whole_pool_every_step"] is False


# --------------------------------------------------------------------------- #
# 5. the image formation is per arm (R1 §5)
# --------------------------------------------------------------------------- #
def test_a0_and_a1_headlines_coincide_on_the_same_carrier() -> None:
    """R1 §5: ``A1``'s ``f(I,S) = I + S[T(I) - I]`` and ``A0``'s outer mix agree.

    They are the same formula, so with the same generator weights the two
    headlines must be **bit** identical; that is the "A0 / A1 are magnitude
    references, not the paper question" claim, checked rather than asserted.
    """
    a0, a1 = _arm("A0", seed=5), _arm("A1", seed=5)
    a1.load_state_dict(a0.state_dict())          # A0 / A1 share their structure
    img = torch.rand(1, 3, 6, 7)
    field = torch.rand(1, 1, 6, 7)
    z = torch.randn(1, 8)

    t_img = a0.apply_to_image(a0.theta(z), img, field)          # A0 ignores S
    head0 = g4d.compose_headline(img, field, t_img, mode="A0")
    f_img = a1.apply_to_image(a1.theta(z), img, field)          # A1 ate S
    head1 = g4d.compose_headline(img, field, f_img, mode="A1")
    assert torch.equal(head0, head1)


@pytest.mark.parametrize("mode", ["A0", "A1", "A2", "A3"])
def test_a_zero_field_pins_a1_to_the_input_and_leaves_a2_a3_free(mode: str) -> None:
    """R1 §5's second consequence: ``E_out`` is no longer zero by construction.

    R0 composed every arm through one outer ``mix_alpha`` with the GT alpha, so
    ``S = 0`` pixels were **forced** equal to the input and the colour leakage
    A2/A3 can have at ``s ~ 0`` was invisible.  Here the formation is per arm.
    """
    arm = _arm(mode)
    with torch.no_grad():                        # a non-trivial global branch
        arm.generator.head_global[-1].bias[9:12] += 0.25
    img = torch.rand(1, 3, 5, 5)
    zero = torch.zeros(1, 1, 5, 5)
    f_img = arm.apply_to_image(arm.theta(torch.randn(1, 8)), img, zero)
    head = g4d.compose_headline(img, zero, f_img, mode=mode)
    if mode in ("A0", "A1"):
        # A0: mix_alpha's endpoint snapping.  A1: the same snapping, inside f.
        assert torch.equal(head, img)
    else:
        assert not torch.equal(head, img), (
            f"{mode}: a conditional arm must not be forced to the identity at "
            "S = 0 -- that is exactly the leakage this experiment measures")


def test_compose_headline_needs_a_mode_at_every_runner_call_site() -> None:
    """AST: no ``compose_headline`` call in the runner may omit ``mode=``."""
    tree = ast.parse(RUNNER_FILE.read_text(encoding="utf-8"), filename=str(RUNNER_FILE))
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "compose_headline"]
    assert calls, "the runner is supposed to form the headline somewhere"
    bad = [c.lineno for c in calls
           if not any(k.arg == "mode" for k in c.keywords)]
    assert bad == [], (f"compose_headline without mode= at lines {bad}: R0's "
                       "single outer compositor is the double-alpha bug (§0.1)")


# --------------------------------------------------------------------------- #
# 6. the gate ladder (R1 §9)
# --------------------------------------------------------------------------- #
POOL = [f"rcp_{i:05d}" for i in range(3149)]


@pytest.mark.parametrize("stage,n_lut,b,q,steps,board", [
    ("G1", 1, 1, 2048, 2000, False),
    ("G2", 32, 32, 512, 4000, False),
    ("G3", 3149, 256, 2048, 18760, True),
])
def test_gate_stage_presets_are_the_spec_table(stage: str, n_lut: int, b: int,
                                               q: int, steps: int,
                                               board: bool) -> None:
    got = runner.resolve_gate_stage(_args("--gate-stage", stage), POOL)
    assert got["stage"] == stage
    assert got["n_lut_pool"] == n_lut == len(got["lut_ids"])
    assert (got["b_samples"], got["q_colors"], got["s_anchors"]) == (b, q, 4)
    assert got["total_steps"] == steps
    assert got["needs_eval_board"] is board
    assert got["colors_per_step"] == got["n_pairs_s"] == b * q * 4
    # every gate stage is an E[lut_id] run: never published, always flagged
    assert got["published"] is False and got["oracle_reference"] is True


def test_the_three_gate_stages_hit_the_spec_colour_counts() -> None:
    counts = {s: runner.resolve_gate_stage(_args("--gate-stage", s),
                                           POOL)["colors_per_step"]
              for s in ("G1", "G2", "G3")}
    assert counts == {"G1": 8_192, "G2": 65_536, "G3": 2_097_152}


def test_g1_defaults_to_the_first_lut_and_rejects_one_outside_the_bucket() -> None:
    got = runner.resolve_gate_stage(_args("--gate-stage", "G1"), POOL)
    assert got["lut_ids"] == [sorted(POOL)[0]]
    named = runner.resolve_gate_stage(
        _args("--gate-stage", "G1", "--g1-lut-id", POOL[17]), POOL)
    assert named["lut_ids"] == [POOL[17]]
    with pytest.raises(SystemExit, match="not in the training LUT bucket"):
        runner.resolve_gate_stage(
            _args("--gate-stage", "G1", "--g1-lut-id", "rcp_missing"), POOL)


def test_g2_pool_is_seed_determined_and_g1_g2_do_not_ask_for_a_board() -> None:
    a = runner.resolve_gate_stage(_args("--gate-stage", "G2", "--seed", "1"), POOL)
    b = runner.resolve_gate_stage(_args("--gate-stage", "G2", "--seed", "1"), POOL)
    c = runner.resolve_gate_stage(_args("--gate-stage", "G2", "--seed", "2"), POOL)
    assert a["lut_ids"] == b["lut_ids"] != c["lut_ids"]
    assert a["needs_eval_board"] is False
    assert runner.GATE_STAGES["G1"]["mining"] is False   # R1 §9-G1: no mining


# --------------------------------------------------------------------------- #
# 7. the NaN re-check (R1 §10)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_a_non_finite_l_rec_stops_the_run_with_a_non_zero_rc(tmp_path, bad) -> None:
    with pytest.raises(SystemExit) as exc:
        runner.assert_l_rec_finite(bad, step=4000, where="quick_eval@step4000",
                                   run_dir=tmp_path)
    assert exc.value.code == runner.RC_NONFINITE != 0
    rec = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert rec["void_reason"] == "L_rec is not finite"
    assert rec["published"] is False and rec["step"] == 4000
    assert rec["where"] == "quick_eval@step4000" and rec["rc"] == runner.RC_NONFINITE


def test_a_finite_l_rec_passes_and_an_absent_one_is_not_a_nan(tmp_path) -> None:
    assert runner.assert_l_rec_finite(0.25, step=1, where="w", run_dir=tmp_path) == 0.25
    # "not measured" is a different failure and belongs to assert_criteria_ran
    assert runner.assert_l_rec_finite(None, step=1, where="w", run_dir=tmp_path) is None
    assert not (tmp_path / "metrics.json").exists()


@pytest.mark.skipif(not MOUNTED, reason="sft2seg splits not mounted")
def test_an_injected_nan_l_rec_kills_a_real_run(tmp_path, monkeypatch) -> None:
    """End to end: the run stops at the quick eval, not at the board."""
    real = g4d.total_loss

    def _nan(*a, **kw):
        loss, cols = real(*a, **kw)
        cols["L_rec"] = float("nan")
        return loss, cols

    monkeypatch.setattr(g4d, "total_loss", _nan)
    out = tmp_path / "nan"
    with pytest.raises(SystemExit) as exc:
        runner.main(["--gate-stage", "G1", "--glut4d-mode", "A1",
                     "--glut4d-n", "4", "--steps", "4", "--quick-eval-every", "2",
                     "--reg-grid", "5", "--device", "cpu", "--out", str(out)])
    assert exc.value.code == runner.RC_NONFINITE
    rec = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert rec["void"] is True and rec["void_reason"] == "L_rec is not finite"
    assert rec["step"] == 2, "it must die at the first quick eval, not at the board"
    assert not (out / "best.pt").exists()


def test_every_quick_eval_rechecks_not_only_the_first(tmp_path) -> None:
    """The hole R1 §10 names: the check must not sit behind ``first_quick``."""
    src = RUNNER_FILE.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(RUNNER_FILE))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "quick_eval")
    calls = [n for n in ast.walk(fn)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "assert_l_rec_finite"]
    assert calls, "quick_eval must re-check L_rec"
    guards = [n for n in ast.walk(fn)
              if isinstance(n, ast.If)
              and "first_quick" in ast.dump(n.test)]
    guarded = {c.lineno for g in guards for c in ast.walk(g)
               if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
               and c.func.id == "assert_l_rec_finite"}
    assert {c.lineno for c in calls} - guarded, (
        "at least one L_rec re-check must run on EVERY quick eval, not only "
        "inside the first_quick branch (R1 §10)")


# --------------------------------------------------------------------------- #
# 8. G1 / G2 start with nothing but luts.npz
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not MOUNTED, reason="sft2seg splits not mounted")
@pytest.mark.parametrize("stage,mode", [("G1", "A1"), ("G2", "A3")])
def test_gate_dry_run_needs_no_z_cache_no_image_no_pred_field(tmp_path, stage,
                                                              mode) -> None:
    out = tmp_path / f"{stage}{mode}"
    rc = runner.main(["--dry-run", "--gate-stage", stage, "--glut4d-mode", mode,
                      "--device", "cpu", "--out", str(out)])
    assert rc == 0
    setup = json.loads((out / "run_setup.json").read_text(encoding="utf-8"))
    assert setup["cond"] == "oracle"
    assert setup["flags"]["z_cache"] is None
    assert setup["flags"]["pred_field_dir"] is None
    assert setup["conditions"]["kind"] == "oracle"
    assert setup["colorspan_assertion"]["ran"] is False
    gate = setup["gate_stage"]
    assert gate["stage"] == stage
    assert gate["published"] is False and gate["oracle_reference"] is True
    assert gate["needs_eval_board"] is False
    assert (setup["measured"]["colors_per_step"]
            == gate["colors_per_step"]
            == gate["b_samples"] * gate["q_colors"] * 4)
    pre = json.loads((out / "config" / "loss_preregistration.json")
                     .read_text(encoding="utf-8"))
    assert pre["mode"] == mode and pre["loss_level"] == 1


# --------------------------------------------------------------------------- #
# 9. the first-row column set
# --------------------------------------------------------------------------- #
def test_first_row_has_r_line_and_no_l_m4d() -> None:
    args = _args()
    cols = g4d.step_columns(runner.resolve_loss_level(args),
                            mode=args.glut4d_mode,
                            r_line=bool(args.lam_line))
    assert "R_line" in cols
    assert "L_m4d" not in cols, "R1 §4.4 deleted L_m4d; R_line replaces it"
    assert "L_s4d" not in cols and "L_img" not in cols   # off, so not computed
    for c in ("L_rec", "n_colors", "n_pairs_s", "gnorm", "null_mass_mean",
              "cholesky_info_nonzero", "tau_p50", "beta_absmean",
              "opacity_p50"):
        assert c in cols, c


def test_lam_line_zero_takes_r_line_off_the_row() -> None:
    args = _args("--lam-line", "0")
    cols = g4d.step_columns(runner.resolve_loss_level(args),
                            mode=args.glut4d_mode, r_line=bool(args.lam_line))
    assert "R_line" not in cols


def test_a1_row_carries_no_tau_or_beta_column() -> None:
    cols = g4d.step_columns(1, mode="A1", r_line=True)
    assert "R_line" in cols
    assert not any(c.startswith("tau_") for c in cols)
    assert "beta_absmean" not in cols


# --------------------------------------------------------------------------- #
# 10. the runner's own surface
# --------------------------------------------------------------------------- #
def test_runner_defaults_are_the_r1_block() -> None:
    args = _args()
    assert args.glut4d_mode == "A3" and args.glut4d_n == 48
    assert args.glut4d_marg_norm == "peak"          # R1 §4.2
    assert args.cond == "oracle" and args.gate_stage == "none"
    assert args.lam_line == g4d.LAMBDA_LINE == 0.1
    assert (args.w_img, args.alpha_s) == (0.0, 0.0)
    assert (args.lam_hc, args.lam_sparse) == (10.0, 0.001)
    assert runner.resolve_loss_level(args) == 1     # R1 §4.4 pure L1
    assert runner.effective_lambdas(args) == (0.0, 0.0)
    assert args.clamp == "two" and args.reg_points == 256
    assert (args.img_pixels, args.img_batch) == (768, 4)
    assert args.batch_split == "32x256" and args.epochs == 40
    assert args.cond_dim == 64 and args.gen_width == 128 and args.lr == 1e-3
    assert args.pi_lr_scale == 0.1 and args.seed == 20260810
    assert runner.FROZEN["headline_formation"] == (
        "per arm (R1 §5): A0 = I + S * [T(I) - I]; A1/A2/A3 = f(I, S)")
    assert runner.parse_batch_split(args.batch_split) == (32, 256)


def test_the_deleted_r0_flags_are_gone() -> None:
    opts = {o for a in runner.build_arg_parser()._actions for o in a.option_strings}
    for dead in ("--glut4d-rot-sign", "--glut4d-rot-flip", "--alpha-m"):
        assert dead not in opts, f"{dead} belongs to R0's deleted carrier"
    for live in ("--lam-line", "--cond", "--gate-stage", "--img-pixels"):
        assert live in opts, live


def test_carrier_glut3d_selects_the_plain_3d_arm() -> None:
    src = RUNNER_FILE.read_text(encoding="utf-8")
    assert 'args.glut4d_mode = "A0"' in src
    assert "maskblend" not in src, "R0's mode names are gone (R1 §3)"


def test_runner_volume_helper_round_trips_the_identity_transform() -> None:
    grid = queries.uniform_grid(9)
    vol = runner.values_to_volume(grid, 9)          # identity LUT as a volume
    x = torch.rand(64, 3)
    assert torch.allclose(apply_lut_volume(vol, x), x, atol=1e-6)


def test_no_cpu_round_trip_on_any_metric_path() -> None:
    """AST, not grep: a ``.cpu()`` in prose is fine, one in code is the 0.296 IoU."""
    tree = ast.parse(RUNNER_FILE.read_text(encoding="utf-8"),
                     filename=str(RUNNER_FILE))
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "cpu":
                bad.append(node.lineno)
            if node.func.attr == "to" and any(
                    isinstance(a, ast.Constant) and a.value == "cpu"
                    for a in node.args):
                bad.append(f"{node.lineno} (.to('cpu'))")
    assert bad == [], f"CUDA and CPU break top-k ties differently: {bad}"


def test_l_s4d_uses_random_points_not_the_full_lattice() -> None:
    """R1 §8.3: ``17^4`` per step is 1.5e13 combinations over the horizon."""
    arm = _arm("A3", n=4).to(dtype=torch.float64)
    params = arm.theta(torch.randn(2, 8, dtype=torch.float64))
    gen = torch.Generator().manual_seed(0)
    val = runner.l_s4d_random_points(arm, params, n_points=16, step=1.0 / 16,
                                     generator=gen, device=torch.device("cpu"))
    assert val.shape == () and float(val.detach()) >= 0.0
    # the training path must not call the lattice builder at all
    tree = ast.parse(RUNNER_FILE.read_text(encoding="utf-8"),
                     filename=str(RUNNER_FILE))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "train")
    grid_calls = [n.lineno for n in ast.walk(fn)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                  and n.func.attr == "grid_4d"]
    assert grid_calls == []


def test_the_oracle_waiver_names_exactly_the_g4_only_columns() -> None:
    """R1 §9.1: G1-G3 do not carry the three negative controls or field_pred."""
    waived = set(runner.ORACLE_WAIVED_CRITERIA)
    assert waived == {"N1_shuffle_delta", "N1_shuffle_M",
                      "N2_irrelevant_delta", "N2_irrelevant_M",
                      "N3_const_delta", "N3_const_M", "field_pred"}
    # everything else this arm pre-registers is still required of an oracle board
    still = set(g4d.required_criteria()) - waived
    for key in ("headline_normal_only", "B4_oracle", "loc_in", "loc_band",
                "loc_out", "grid_s0", "grid_s100"):
        assert key in still, key


def test_gate_grid_metrics_reads_the_five_s_columns(monkeypatch) -> None:
    """R1 §9.1's G1 / G2 read-out: function values, no image and no board."""

    class _Bank:
        def apply(self, x, lut_id):              # a deterministic fake LUT
            return (x * 0.5 + 0.25).clamp(0, 1)

    arm = _arm("A2", n=4)
    store = runner.OracleConditionStore(["l0", "l1"], cond_dim=8, seed=0)
    out = runner.gate_grid_metrics(arm, store, _Bank(), ["l0", "l1"],
                                   torch.device("cpu"), grid_n=5)
    assert out["n_lut"] == 2
    for _, key in g4d.S_AXIS_GRID:
        assert out[key]["n"] == 2 and math.isfinite(out[key]["mean"])
    assert out["grid_de00_mean"]["n"] == 10
    # s = 0 is the identity target, so it is the easiest column by construction
    assert out["grid_s0"]["mean"] <= out["grid_s100"]["mean"]


def test_target_is_the_data_law_at_every_anchor() -> None:
    """The training target is ``mix_alpha`` itself, endpoint snapping included."""
    b, q = 2, 6
    x, s = runner.paired_anchor_batch(_sampler(q), b, q, device=torch.device("cpu"))
    lut = torch.rand_like(x)
    y = g4d.target_4d(x, s, lut)
    yg, xg, lg = (t.reshape(b, q, 4, 3) for t in (y, x, lut))
    assert torch.equal(yg[:, :, 0], xg[:, :, 0])     # s = 0 -> the input
    assert torch.equal(yg[:, :, 1], lg[:, :, 1])     # s = 1 -> the LUT
    assert torch.allclose(y, mix_alpha(x, lut, s.unsqueeze(-1)))
