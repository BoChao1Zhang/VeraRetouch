#!/usr/bin/env python3
# 源自 experiments/prs/EPR-051_masked-restore-production/stage0/sprf/train_align_time.py @ 2026-09-06（原件未入 git；原 sha256 见 veraretouch_sprf/PROVENANCE.md），逐字复制/仅改导入与路径解析（EPR-052）。
"""EPR-051 · stage0 · sprf · T-ALIGN-TIMEFILM 训练入口（contract=predicted_lut）。

四要素
  数据    与 C-LUT-XL **同一条**：v3 279 shards；训练固定 clutxl300k 子集
          n=300,000；held-out 只认冻结 id 表，精确 4,560。规则与 sha 全部继承
          backbone_run 的冻结 config（`run_args.json`），X1--X5 逐条重验。
  模型    冻结 `XL_FINALIZED.json` 指定且 sha 闭合的 formal checkpoint
          （SPRF 主干 + edit_enc）；新训一个
          `LatentPredictor`：完整替换 stage-conditioning block 为
          (固定 stage/time code + FiLM + identity zero-init)，其余结构配对不变；输入为
          (冻结 pooled c, 退化图 y 的 β 加权/全局直方图)
          -> 128 维编辑 latent。推理时把预测 latent 顶替 oracle 查表的位置。
  数学    对齐目标 `t_{m} = edit_enc_frozen(inv_table[lut_id_m])`（只是 lut_id 的
          函数，4052×128 一次算完）。
          L = w_latent · SmoothL1(pred, t)  在 **β 非零的阶段**上取均值。
          （w_pixel 预注册为 0：本期只证对齐可行，不做端到端。）
  优化器  AdamW lr 3e-4 / wd 0.01 / betas(0.9,0.999) / cosine / batch 64 / clip 1.0。

断言
  B1 主干确实全冻（`requires_grad` 全 False，且训练前后主干权重逐位不变）
  B2 predictor 拿到非零梯度（「定义了没接线」）
  B3 held-out 隔离：predictor 训练逐 batch 断言 key 不在 held-out
  B4 注入正确性：把**目标** latent（而非预测）喂进去，rollout 必须与 C-LUT oracle
     查表逐位相同 —— 证明注入路径本身没有改变模型语义
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
from veraretouch_sprf import _paths as _P  # EPR-052：替代原 sys.path 注入块
_P.ensure_sys_path()
STAGE0 = _P.STAGE0
REPO = _P.REPO

from veraretouch_sprf.data import train_stage0 as T0             # noqa: E402
import epr050_build_degradation as B  # noqa: E402
from veraretouch_sprf.data import stage_targets as ST            # noqa: E402
from veraretouch_sprf.models import stage_flow as SF               # noqa: E402
from veraretouch_sprf.solver import stage_solver as SS             # noqa: E402
from veraretouch_sprf.models import edit_cond as EC                # noqa: E402
from veraretouch_sprf.models import align_predictor as AP_BASE     # noqa: E402
from veraretouch_sprf.models import align_predictor_time as AP     # noqa: E402
from veraretouch_sprf.data import archive_assets as AA           # noqa: E402
from veraretouch_sprf.train.train_sprf_xl import install_subset  # noqa: E402
from veraretouch_sprf.eval.probes.quick100_eval import MiniCfg     # noqa: E402


XL_EPR = "EPR-051/stage0/sprf/C-LUT-XL"
XL_FINALIZED_SCHEMA = "epr051-xl-finalized-v1"
XL_CONTROLLER_SCHEMA = "epr051-xl-controller-v1"
XL_CANONICAL_SCHEMA = "epr051-xl-canonical-metrics-v1"
XL_COLUMNS = [
    "model", "identity", "exact_solve", "model_midpoint",
    "active_only_headline", "stage_all", "stage_active", "stage_beta1",
    "cycle", "rollout_drift", "mask_leak_max", "mask_bitexact_count",
    "stage_oob_fraction", "stage_oob_max", "condition_number_by_stage",
    "nfe_d_vs_2d", "latency_by_depth", "delta_const", "delta_shuffle",
    "stage_mask_bitexact", "de00", "de00_frac_le5", "de00_identity",
    "de00_identity_frac_le5", "delta_edit_null", "delta_edit_roll",
]
XL_STRATA = ["depth", "geom", "rec_band"]
XL_SPARSE_OK = ["stage_beta1", "stage_mask_bitexact"]
XL_ERR_COLUMNS = {
    "model", "identity", "exact_solve", "model_midpoint",
    "active_only_headline", "stage_all", "stage_active", "stage_beta1",
    "cycle", "de00", "de00_identity",
}
XL_COLUMN_SOURCE = {
    "model": "uint8_asset", "identity": "uint8_asset",
    "exact_solve": "journal_certificate", "model_midpoint": "uint8_asset",
    "active_only_headline": "uint8_asset_active", "stage_all": "teacher_state",
    "stage_active": "teacher_state", "stage_beta1": "teacher_state",
    "cycle": "teacher_state", "rollout_drift": "euler_d",
    "mask_leak_max": "off_union", "mask_bitexact_count": "off_union",
    "stage_oob_fraction": "euler_d", "stage_oob_max": "euler_d",
    "condition_number_by_stage": "fd_jacobian",
    "nfe_d_vs_2d": "euler_vs_midpoint", "latency_by_depth": "wall_clock_ms",
    "delta_const": "cond_control", "delta_shuffle": "cond_control",
    "stage_mask_bitexact": "per_stage_beta0", "de00": "ciede2000_vs_clean",
    "de00_frac_le5": "ciede2000_vs_clean",
    "de00_identity": "ciede2000_vs_clean",
    "de00_identity_frac_le5": "ciede2000_vs_clean",
    "delta_edit_null": "edit_control", "delta_edit_roll": "edit_control",
}
XL_RUNTIME_DEPENDENCIES = {
    "train_sprf", "stage_targets", "stage_flow", "stage_solver",
    "train_stage0", "lutdata", "epr050_build_degradation", "train_sprf_xl",
    "archive_assets", "edit_cond", "eval_sprf_checkpoint",
}
ALIGN_COLUMNS = ["model", "identity", "delta_const", "delta_shuffle",
                 "delta_edit_null", "delta_edit_roll"]
ALIGN_COLUMN_SOURCES = {
    "model": ["uint8_asset"], "identity": ["uint8_asset"],
    "delta_const": ["cond_control"], "delta_shuffle": ["cond_control"],
    "delta_edit_null": ["edit_control"], "delta_edit_roll": ["edit_control"],
}
ALIGN_STRATA = ["depth", "geom", "rec_band"]
ALIGN_EPR = "EPR-051/stage0/sprf/T-ALIGN"
ALIGN_STATE_SHA256 = "a3b9300d0fdebc05347febaebfbbaf4bb09822deb67a9e55c6590c26dfbfc14b"


def die(msg: str):
    raise SystemExit(f"train_align_time: {msg}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_sha256(t: torch.Tensor) -> str:
    """Tensor 内容 + dtype/shape 的稳定 A8 指纹。"""
    x = t.detach().contiguous().cpu()
    h = hashlib.sha256()
    h.update(str(x.dtype).encode())
    h.update(json.dumps(list(x.shape)).encode())
    h.update(x.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def module_sha256(module: nn.Module) -> str:
    """全 state_dict 指纹；B1 不只抽样前几个参数。"""
    h = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        h.update(name.encode())
        h.update(tensor_sha256(value).encode())
    return h.hexdigest()


def named_tensors_sha256(items) -> str:
    """Stable fingerprint for an explicitly selected state_dict subset."""
    h = hashlib.sha256()
    for name, value in sorted(items):
        h.update(name.encode())
        h.update(tensor_sha256(value).encode())
    return h.hexdigest()


def load_torch_hashed(path: Path) -> tuple[dict, str]:
    """同一份字节既做 A8 sha 又反序列化，避免原子替换时路径与对象错配。"""
    raw = path.read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    obj = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    del raw
    return obj, got


def read_stable_json(path: Path, label: str) -> tuple[dict, bytes, str]:
    """连续两次 raw read 必须逐位相同；解析、sha 均只认第一份 bytes。"""
    try:
        raw = path.read_bytes()
        raw2 = path.read_bytes()
    except OSError as exc:
        die(f"formal readiness 缺/不可读 {label}: {path} ({exc})")
    if raw2 != raw:
        die(f"formal readiness {label} 在 stable raw read 间变化: {path}")
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        die(f"formal readiness {label} 不是合法 JSON: {exc}")
    if not isinstance(obj, dict):
        die(f"formal readiness {label} 不是 JSON object")
    return obj, raw, hashlib.sha256(raw).hexdigest()


def read_stable_bytes(path: Path, label: str) -> tuple[bytes, str]:
    try:
        raw = path.read_bytes()
        raw2 = path.read_bytes()
    except OSError as exc:
        die(f"formal readiness 缺/不可读 {label}: {path} ({exc})")
    if raw2 != raw:
        die(f"formal readiness {label} 在 stable raw read 间变化: {path}")
    return raw, hashlib.sha256(raw).hexdigest()


def expect_keys(obj: dict, keys: set[str], label: str) -> None:
    if not isinstance(obj, dict) or set(obj) != keys:
        got = set(obj) if isinstance(obj, dict) else set()
        die(f"formal readiness {label} schema keys mismatch: "
            f"{sorted(got ^ keys)}")


def verified_json_record(record: dict, label: str) -> tuple[dict, bytes, str]:
    expect_keys(record, {"path", "sha256"}, label)
    path = Path(record["path"])
    obj, raw, digest = read_stable_json(path, label)
    if digest != record["sha256"]:
        die(f"formal readiness {label} bytes sha 不闭合")
    return obj, raw, digest


def semantic_json_list(path: Path, label: str) -> tuple[list, str]:
    try:
        raw = path.read_bytes()
        raw2 = path.read_bytes()
        values = json.loads(raw)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        die(f"formal readiness {label} 不可读: {exc}")
    if raw2 != raw or not isinstance(values, list) or not all(
            isinstance(x, str) for x in values):
        die(f"formal readiness {label} 不是稳定字符串列表")
    return values, hashlib.sha256("\n".join(values).encode()).hexdigest()


def finite_number(value, label: str, allow_nan: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        die(f"formal readiness {label} 不是数值")
    if not math.isfinite(float(value)) and not (allow_nan and math.isnan(float(value))):
        die(f"formal readiness {label} 不是有限值")


def validate_xl_heldout(metrics: dict, *, require_contract: bool) -> dict:
    """镜像 frozen finalizer validate_metrics 的 canonical v1 深 schema。"""
    top = {"step", "eval_kind", "metric", "n", "err_columns",
           "scalar_columns", "sparse_ok_columns", "exact_solve_source",
           "overall", "by", "per_sample", "column_missing",
           "column_available", "column_sources", "columns_not_registered"}
    if require_contract:
        top.add("contract")
    expect_keys(metrics, top, "canonical.heldout")
    if (metrics.get("step") != 8000 or metrics.get("eval_kind") != "final" or
            metrics.get("n") != 4560 or
            (require_contract and metrics.get("contract") != "oracle_lut") or
            (not require_contract and "contract" in metrics)):
        die("formal readiness heldout step/final/n/contract 不闭合")
    err = [c for c in XL_COLUMNS if c in XL_ERR_COLUMNS]
    scalar = [c for c in XL_COLUMNS if c not in XL_ERR_COLUMNS]
    metric_text = ("8-bit L-inf over RGB, |x_hat - before| * 255; per-sample "
                   "p50/p95/p99, then the MEDIAN across evaluated samples")
    if (metrics.get("metric") != metric_text or metrics.get("err_columns") != err or
            metrics.get("scalar_columns") != scalar or
            metrics.get("sparse_ok_columns") != XL_SPARSE_OK or
            metrics.get("exact_solve_source") != "journal" or
            metrics.get("columns_not_registered") != {}):
        die("formal readiness heldout metric/ERR-scalar/sparse schema 不闭合")
    overall = metrics.get("overall", {})
    if set(overall) != set(XL_COLUMNS):
        die("formal readiness heldout overall 不等于26列")
    for col in XL_COLUMNS:
        cell = overall[col]
        wanted = {"p50", "p95", "p99"} if col in XL_ERR_COLUMNS else {"value"}
        if not isinstance(cell, dict) or set(cell) != wanted:
            die(f"formal readiness overall.{col} shape 不闭合")
        for field in wanted:
            finite_number(cell[field], f"overall.{col}.{field}", col in XL_SPARSE_OK)
    missing = metrics.get("column_missing")
    available = metrics.get("column_available")
    sources = metrics.get("column_sources")
    for label, mapping in (("missing", missing), ("available", available),
                           ("sources", sources)):
        if not isinstance(mapping, dict) or set(mapping) != set(XL_COLUMNS):
            die(f"formal readiness heldout ledger {label} keys 不闭合")
    for col in XL_COLUMNS:
        miss, avail, src = missing[col], available[col], sources[col]
        if (not isinstance(miss, int) or isinstance(miss, bool) or
                not isinstance(avail, int) or isinstance(avail, bool) or
                miss < 0 or avail < 0 or miss + avail != 4560 or
                src != [XL_COLUMN_SOURCE[col]] or
                (miss and col not in XL_SPARSE_OK)):
            die(f"formal readiness heldout ledger count/source 不闭合: {col}")
    by = metrics.get("by")
    if not isinstance(by, dict) or list(by) != XL_STRATA:
        die("formal readiness heldout strata 不是 depth/geom/rec_band")
    for stratum in XL_STRATA:
        cells = by[stratum]
        if (not isinstance(cells, dict) or not cells or
                sum(x.get("n", -1) for x in cells.values()
                    if isinstance(x, dict)) != 4560):
            die(f"formal readiness heldout {stratum} cell n 不闭合")
        for label, cell in cells.items():
            if (not isinstance(cell, dict) or set(cell) != set(XL_COLUMNS) | {"n"} or
                    not isinstance(cell["n"], int) or isinstance(cell["n"], bool) or
                    cell["n"] <= 0):
                die(f"formal readiness heldout {stratum}/{label} shape 不闭合")
            for col in XL_COLUMNS:
                value = cell[col]
                wanted = {"p50", "p95"} if col in XL_ERR_COLUMNS else {"value"}
                if not isinstance(value, dict) or set(value) != wanted:
                    die(f"formal readiness heldout {stratum}/{label}/{col} shape 不闭合")
                for field in wanted:
                    finite_number(value[field], f"by.{stratum}.{label}.{col}.{field}",
                                  col in XL_SPARSE_OK)
    per_sample = metrics.get("per_sample")
    if not isinstance(per_sample, list) or len(per_sample) != 4560:
        die("formal readiness heldout per_sample != 4560")
    sample_keys = {"id", "depth", "geom", "rec_band", "major", "n_eval_pixels",
                   "union_frac", "alpha_hat_mean", "y_quant_gap", "cols"}
    for i, rec in enumerate(per_sample):
        if (not isinstance(rec, dict) or set(rec) != sample_keys or
                not isinstance(rec.get("cols"), dict) or
                set(rec["cols"]) != set(XL_COLUMNS)):
            die(f"formal readiness heldout per_sample[{i}] metadata/26列不闭合")
        for col, value in rec["cols"].items():
            if not isinstance(value, dict):
                die(f"formal readiness per_sample[{i}].{col} 非object")
            if col in XL_ERR_COLUMNS:
                if value.get("missing") is True:
                    if col not in XL_SPARSE_OK or set(value) != {"missing", "source"}:
                        die(f"formal readiness per_sample[{i}].{col} missing shape 不闭合")
                else:
                    required = {"n", "p50", "p95", "p99", "mean", "max"}
                    if (not required <= set(value) or
                            not set(value) <= required | {"source", "by_stage"} or
                            not isinstance(value["n"], int) or isinstance(value["n"], bool) or
                            value["n"] <= 0):
                        die(f"formal readiness per_sample[{i}].{col} ERR shape 不闭合")
                    for field in required - {"n"}:
                        finite_number(value[field], f"per_sample[{i}].{col}.{field}")
            else:
                if "value" not in value or "source" not in value:
                    die(f"formal readiness per_sample[{i}].{col} scalar shape 不闭合")
                finite_number(value["value"], f"per_sample[{i}].{col}.value",
                              col in XL_SPARSE_OK)
            if ("source" in value and
                    value.get("source") != XL_COLUMN_SOURCE[col]):
                die(f"formal readiness per_sample[{i}].{col} source 不闭合")
    return dict(expected_calls=4560,
                column_calls={c: 4560 for c in XL_COLUMNS},
                column_missing=missing, column_available=available,
                column_sources=sources, strata=XL_STRATA)


def snapshot_formal_checkpoint(record: dict, finalized_path: Path,
                               producer_run: Path,
                               frozen_config_sha: str,
                               frozen_sha256: dict) -> tuple[dict, Path, str]:
    """Formal alias 只打开一次；该 raw 同时供 sha、BytesIO load 和模型。"""
    expect_keys(record, {"path", "sha256", "bytes", "step", "config_sha256",
                         "source_path", "source_sha256",
                         "content_addressed_path"}, "checkpoint")
    alias = Path(record["path"])
    object_path = Path(record["content_addressed_path"])
    term = finalized_path.parent.resolve()
    source_path = Path(record["source_path"])
    if (alias.resolve() != term / "ckpt_formal.pt" or
            object_path.parent.resolve() != term / "checkpoints" or
            object_path.name != f"{record['sha256']}.pt" or
            source_path.resolve() != producer_run.resolve() / "ckpt_last.pt"):
        die("formal readiness checkpoint alias/object/source 路径边界不闭合")
    try:
        with alias.open("rb") as f:
            alias_stat = os.fstat(f.fileno())
            raw = f.read()
        object_stat = object_path.stat()
        source_raw = source_path.read_bytes()
        source_raw2 = source_path.read_bytes()
    except OSError as exc:
        die(f"formal readiness checkpoint 不可读: {exc}")
    if ((alias_stat.st_dev, alias_stat.st_ino, alias_stat.st_size) !=
            (object_stat.st_dev, object_stat.st_ino, object_stat.st_size)):
        die("formal readiness checkpoint alias/object 不是同一 create-only inode")
    digest = hashlib.sha256(raw).hexdigest()
    source_digest = hashlib.sha256(source_raw).hexdigest()
    if (digest != record["sha256"] or digest != record["source_sha256"] or
            source_digest != record["source_sha256"] or source_raw2 != source_raw or
            source_raw != raw or len(source_raw) != record["bytes"] or
            len(raw) != record["bytes"] or alias_stat.st_size != len(raw)):
        die("formal readiness checkpoint raw sha/bytes/source_sha 不闭合")
    del source_raw, source_raw2
    try:
        payload = torch.load(io.BytesIO(raw), map_location="cpu",
                             weights_only=False)
    except Exception as exc:
        die(f"formal readiness checkpoint torch.load 失败: {exc}")
    del raw
    if (not isinstance(payload, dict) or not isinstance(payload.get("model"), dict)
            or payload.get("step") != 8000 or record["step"] != 8000 or
            payload.get("config_sha256") != frozen_config_sha or
            record["config_sha256"] != frozen_config_sha or
            payload.get("frozen_sha256") != frozen_sha256):
        die("formal readiness checkpoint payload step/config/object 不闭合")
    return payload, alias, digest


def roll_active_edits(edits: torch.Tensor, depths: torch.Tensor
                      ) -> tuple[torch.Tensor, dict]:
    """逐样本只错配 active [0:depth)；inactive 尾部逐位不动。"""
    out = edits.clone()
    exceptions = 0
    for i, d0 in enumerate(depths.detach().cpu().tolist()):
        d = int(d0)
        if not 1 <= d <= edits.shape[1]:
            die(f"delta_edit_roll depth={d} 越出 1..{edits.shape[1]}")
        before = edits[i, :d]
        out[i, :d] = torch.roll(before, 1, dims=0)
        if not torch.equal(out[i, d:], edits[i, d:]):
            die("delta_edit_roll 改动了 inactive 尾部")
        # roll 天然是 active 集合的置换；全相等/周期重复时错配可以自然恒等。
        if torch.equal(out[i, :d], before):
            exceptions += 1
    return out, dict(n=int(edits.shape[0]), natural_identity_exceptions=exceptions,
                     active_set_preserved=True, inactive_bitexact=True)


def assert_roll_helper(n_steps: int, device: str) -> dict:
    if n_steps < 6:
        die(f"delta_edit_roll d4/d5/d6 断言要求 n_steps>=6，收到 {n_steps}")
    x = torch.arange(3 * n_steps * 2, device=device).reshape(3, n_steps, 2)
    depths = torch.tensor([4, 5, 6], device=device)
    y, rep = roll_active_edits(x, depths)
    for i, d in enumerate((4, 5, 6)):
        if not torch.equal(y[i, :d], torch.roll(x[i, :d], 1, dims=0)):
            die(f"delta_edit_roll d{d} active 循环断言失败")
        if torch.equal(y[i, :d], x[i, :d]):
            die(f"delta_edit_roll d{d} 合成非重复输入却恒等")
        if not torch.equal(y[i, d:], x[i, d:]):
            die(f"delta_edit_roll d{d} inactive 尾部改变")
    rep["depths_tested"] = [4, 5, 6]
    rep["synthetic_nonidentity"] = True
    return rep


class AlignPixels(Dataset):
    """一条 = (特征 key, 逐阶段手工特征, 逐阶段逆表行号, depth 掩膜)。

    直方图在**整图**上算（不是像素子集）：它是分布统计，子集只会加方差。
    离开 worker 的只有 (K, feat_dim)，不是 (K, P) 的 β 场。
    """

    def __init__(self, samples, blobs, n_steps, grid, inv_src):
        self.samples, self.blobs = samples, blobs
        self.n_steps, self.grid, self.inv = int(n_steps), int(grid), inv_src

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        row = json.loads(self.blobs[s["id"]])
        x0, y = T0.load_pair(s["shard"], row, s["after_asset"])
        af = T0.alpha_fields(row, x0)
        mask = T0.depth_mask(s["depth"], self.n_steps)
        alphas = af.reshape(self.n_steps, -1) * mask.unsqueeze(-1)
        feat = AP.hist_features(y.reshape(-1, 3), alphas, self.grid)
        return dict(feat=feat, rows=self.inv.for_row(row, x0, mask), mask=mask,
                    key=T0.feature_key(s), sid=s["id"],
                    depth=torch.tensor(self.n_steps if s["depth"] is None
                                       else int(s["depth"]), dtype=torch.long))


def collate(b):
    out = {k: torch.stack([x[k] for x in b])
           for k in ("feat", "rows", "mask", "depth")}
    out["key"] = [x["key"] for x in b]
    out["sid"] = [x["sid"] for x in b]
    return out


def predict_latents(predictor: AP.LatentPredictor, c: torch.Tensor,
                    y_full: torch.Tensor, alphas_full: torch.Tensor,
                    grid: int, device: str) -> torch.Tensor:
    """只用生产可见量 c/y/beta，返回 (B,K,128)；不接触 LUT 身份。"""
    if y_full.ndim != 3 or alphas_full.ndim != 3:
        die(f"predicted_lut 输入形状错误: y={tuple(y_full.shape)} "
            f"alphas={tuple(alphas_full.shape)}")
    feats = torch.stack([
        AP.hist_features(y_full[i].detach().cpu(),
                         alphas_full[i].detach().cpu(), grid)
        for i in range(y_full.shape[0])
    ]).to(device)
    return predictor(c.to(device), feats)


@torch.no_grad()
def restore_predicted_lut(model: nn.Module, predictor: AP.LatentPredictor,
                          batch: SS.ProductionBatch, path_mode: str,
                          nfe_per_stage: int, depth: int, grid: int,
                          device: str, contract: str,
                          histogram_y: torch.Tensor | None = None,
                          histogram_alphas: torch.Tensor | None = None,
                          oracle_lut: torch.Tensor | None = None) -> torch.Tensor:
    """T-ALIGN-TIMEFILM 生产入口：只接 ProductionBatch，并显式拒绝 oracle LUT。"""
    if contract != "predicted_lut":
        die(f"生产入口 contract={contract!r}，只接受 'predicted_lut'")
    if oracle_lut is not None:
        die("生产 predicted_lut 入口拒绝 oracle LUT")
    if not isinstance(batch, SS.ProductionBatch):
        die("生产 predicted_lut 入口只接受 ProductionBatch")
    missing = [k for k in SS.PRODUCTION_KEYS if k not in batch]
    if missing:
        die(f"生产 predicted_lut 入口缺键 {missing}")
    c, y = batch["cond"], batch["y"]
    alphas, union = batch["alphas"], batch["union"]
    d, s = batch["depth"], batch["s"]
    hy = y if histogram_y is None else histogram_y
    ha = alphas if histogram_alphas is None else histogram_alphas
    if (histogram_y is None) != (histogram_alphas is None):
        die("histogram_y/histogram_alphas 必须同时给或同时省略")
    if hy.shape[0] != y.shape[0] or ha.shape[:2] != alphas.shape[:2]:
        die("production histogram context 与 batch 的 B/K 不一致")
    edits = predict_latents(predictor, c, hy, ha, grid, device)
    base = model.cond.base(c.to(device))
    ahat = ST.compose_alpha_hat(alphas.to(device))
    out, mtr = SS.rollout_with_metrics(
        model, base, y.to(device), ahat, alphas.to(device), union.to(device),
        d.to(device), s.to(device), path_mode, SS.stages_for(path_mode, depth),
        nfe_per_stage, edits=edits)
    want = SS.stages_for(path_mode, depth) * int(nfe_per_stage)
    if int(mtr["nfe"]) != want:
        die(f"生产 predicted_lut NFE {mtr['nfe']} != {want}")
    return out


@torch.no_grad()
def final_eval(model: nn.Module, predictor: AP.LatentPredictor, held, blobs,
               feats_cache, cfg: T0.Cfg, cd: dict, device: str, n_steps: int,
               grid: int, targets: torch.Tensor, inv: EC.InvLutSource,
               smoke: bool) -> dict:
    """predicted_lut 正式 rollout；final 必须逐一覆盖冻结 held-out 4,560。"""
    if not smoke and len(held) != 4560:
        die(f"T-ALIGN-TIMEFILM final held-out {len(held)} != 4560")
    columns = ["model", "identity", "delta_const", "delta_shuffle",
               "delta_edit_null", "delta_edit_roll"]
    strata = ["depth", "geom", "rec_band"]
    ledger = T0.MetricLedger(columns, strata)
    ledger.arm(len(held))
    partners = ST.shuffle_partner_index(
        [s["id"] for s in held], cd["eval"]["shuffle_salt"], T0.source_id_of)
    n_pix = int(cd["eval"]["pixels_per_sample"])
    if smoke:
        n_pix = min(n_pix, 64)
    path_mode = cd["flow"]["path_mode"]
    nfe = int(cd["solver"]["nfe_per_stage"])
    lo, hi = float(cd["solver"]["clamp_lo"]), float(cd["solver"]["clamp_hi"])
    null_lat = targets[int(model.edit_null_row)].to(device)
    recs, roll_reports = [], []
    predictor.eval(); model.eval()

    def run(base_c, yb, alphas, union, depth, sc, edits, d_int):
        base = model.cond.base(base_c)
        out, _ = SS.rollout_with_metrics(
            model, base, yb, ST.compose_alpha_hat(alphas), alphas, union,
            depth, sc, path_mode, SS.stages_for(path_mode, d_int), nfe,
            clamp_lo=lo, clamp_hi=hi, edits=edits)
        return out

    for i, s in enumerate(held):
        row = json.loads(blobs[s["id"]])
        x0, y = T0.load_pair(s["shard"], row, s["after_asset"])
        mask = T0.depth_mask(s["depth"], n_steps)
        af_full = T0.alpha_fields(row, x0).reshape(n_steps, -1) * mask.unsqueeze(-1)
        y_full = y.reshape(-1, 3)
        npx = int(y_full.shape[0])
        idx = (torch.arange(npx) if n_pix <= 0 else
               torch.arange(0, npx, max(1, npx // n_pix))[:n_pix])
        xb = x0.reshape(-1, 3)[idx].to(device).unsqueeze(0)
        yb = y_full[idx].to(device).unsqueeze(0)
        alphas = af_full[:, idx].to(device).unsqueeze(0)
        union = ST.union_mask_of(alphas)
        d_int = n_steps if s["depth"] is None else int(s["depth"])
        depth = torch.tensor([d_int], device=device)
        sc = torch.tensor([float(row["calib"]["s"])], device=device)
        c = T0.gather_feats(feats_cache, [T0.feature_key(s)], 0, device)
        c_const = torch.zeros_like(c)
        c_shuffle = T0.gather_feats(
            feats_cache, [T0.feature_key(held[partners[i]])], 0, device)
        yf = y_full.unsqueeze(0)
        aff = af_full.unsqueeze(0)
        ed = predict_latents(predictor, c, yf, aff, grid, device)
        ed_const = predict_latents(predictor, c_const, yf, aff, grid, device)
        ed_shuffle = predict_latents(predictor, c_shuffle, yf, aff, grid, device)
        prod_batch = SS.ProductionBatch(dict(
            cond=c, y=yb, alphas=alphas, union=union, depth=depth, s=sc))
        out = restore_predicted_lut(
            model, predictor, prod_batch, path_mode, nfe, d_int, grid, device,
            "predicted_lut", histogram_y=yf, histogram_alphas=aff)
        out_const = run(c_const, yb, alphas, union, depth, sc, ed_const, d_int)
        out_shuffle = run(c_shuffle, yb, alphas, union, depth, sc,
                          ed_shuffle, d_int)
        out_null = run(c, yb, alphas, union, depth, sc,
                       null_lat.view(1, 1, -1).expand(1, n_steps, -1), d_int)
        ed_roll, roll_rep = roll_active_edits(ed, depth)
        roll_reports.append(roll_rep)
        out_roll = run(c, yb, alphas, union, depth, sc, ed_roll, d_int)
        stats = T0.err_stats(T0.linf8(out, xb))
        ident = T0.err_stats(T0.linf8(yb, xb))
        controls = {}
        for col, value in (("delta_const", out_const),
                           ("delta_shuffle", out_shuffle),
                           ("delta_edit_null", out_null),
                           ("delta_edit_roll", out_roll)):
            cs = T0.err_stats(T0.linf8(value, xb))
            controls[col] = dict(value=cs["p50"] - stats["p50"],
                                 control_p50=cs["p50"])
            ledger.note(col, source=("cond_control" if col in
                        ("delta_const", "delta_shuffle") else "edit_control"))
        ledger.note("model", source="uint8_asset")
        ledger.note("identity", source="uint8_asset")
        rows = inv.for_row(row, x0, mask).to(device)
        tgt = targets[rows]
        recs.append(dict(
            id=s["id"], depth=d_int, geom=s["geom"], rec_band=s["rec_band"],
            n_eval_pixels=int(idx.numel()), model=stats, identity=ident,
            latent_cos=float(nn.functional.cosine_similarity(ed[0], tgt).mean()),
            latent_l2=float((ed[0] - tgt).norm(dim=-1).mean()), **controls))
        if (i + 1) % 100 == 0 or i + 1 == len(held):
            print(f"[align-eval] {i+1}/{len(held)}", flush=True)

    def med(col, field):
        vals = [r[col][field] for r in recs]
        return float(np.median(np.asarray(vals, dtype=float)))

    overall = {
        "model": {f: med("model", f) for f in ("p50", "p95", "p99")},
        "identity": {f: med("identity", f) for f in ("p50", "p95", "p99")},
    }
    for col in columns[2:]:
        overall[col] = dict(value=med(col, "value"),
                            control_p50=med(col, "control_p50"))
    overall["latent_cos"] = dict(value=float(np.median(
        [r["latent_cos"] for r in recs])))
    overall["latent_l2"] = dict(value=float(np.median(
        [r["latent_l2"] for r in recs])))
    by = {}
    for stratum in strata:
        by[stratum] = {}
        for key in sorted({str(r[stratum]) for r in recs}):
            cell = [r for r in recs if str(r[stratum]) == key]
            by[stratum][key] = dict(n=len(cell))
    out = dict(n=len(recs), eval_kind="final", contract="predicted_lut",
               columns_registered=columns, overall=overall, by=by,
               model_execution_path="restore_predicted_lut(ProductionBatch)",
               delta_edit_roll_guard=dict(
                   n=sum(r["n"] for r in roll_reports),
                   natural_identity_exceptions=sum(
                       r["natural_identity_exceptions"] for r in roll_reports),
                   active_set_preserved=all(
                       r["active_set_preserved"] for r in roll_reports),
                   inactive_bitexact=all(r["inactive_bitexact"]
                                         for r in roll_reports)),
               per_sample=recs)
    ledger.assert_wired(out)
    out["column_calls"] = ledger.calls
    out["column_sources"] = {k: sorted(v) for k, v in ledger.sources.items()}
    return out


def validate_align_guard_bundle(assertions: dict, *, require_state_sha: bool) -> None:
    """Deep semantic validation shared by matched metrics/train_meta."""
    if not isinstance(assertions, dict):
        die("matched baseline assertions 非object")
    if assertions.get("B1_backbone_frozen") is not True:
        die("matched baseline B1_backbone_frozen != true")
    if require_state_sha and assertions.get("B1_state_sha256") != ALIGN_STATE_SHA256:
        die("matched baseline B1 完整 state sha 不闭合")
    b2 = assertions.get("B2_predictor_grad", {})
    if (not isinstance(b2, dict) or not math.isfinite(float(b2.get("grad_l2", 0))) or
            not float(b2.get("grad_l2", 0)) > 0 or
            b2.get("params_without_grad") != 0):
        die("matched baseline B2 grad/nonmissing 不闭合")
    if assertions.get("B3_heldout_isolated") is not True:
        die("matched baseline B3_heldout_isolated != true")
    b4 = assertions.get("B4_injection_bitexact", {})
    cache_max = b4.get("target_cache_runtime_max_abs")
    if (b4.get("ok") is not True or b4.get("max_abs") != 0 or
            isinstance(cache_max, bool) or not isinstance(cache_max, (int, float)) or
            not math.isfinite(float(cache_max)) or not 0 <= float(cache_max) <= 2e-6):
        die("matched baseline B4 exact/cache-allclose 不闭合")
    prod = assertions.get("production_guard", {})
    if (prod.get("ok") is not True or prod.get("rejects_oracle_lut") is not True or
            prod.get("production_keys") != list(SS.PRODUCTION_KEYS) or
            prod.get("output_shape") != [1, 256, 3]):
        die("matched baseline ProductionBatch/oracle guard 不闭合")
    roll = assertions.get("delta_edit_roll_helper", {})
    if (roll.get("depths_tested") != [4, 5, 6] or
            roll.get("active_set_preserved") is not True or
            roll.get("inactive_bitexact") is not True or
            roll.get("synthetic_nonidentity") is not True):
        die("matched baseline active d4/d5/d6 guard 不闭合")
    seen = assertions.get("one_pass_seen", {})
    if (seen.get("seen") != 300000 or seen.get("expected") != 300000 or
            seen.get("exact") is not True or seen.get("drop_last") is not False or
            seen.get("one_pass_steps") != 4688):
        die("matched baseline one-pass 300k/drop_last=false 不闭合")


def validate_align_heldout(held: dict) -> dict:
    """Validate all 4,560 records and derive the matched headline row."""
    expected_keys = {"n", "eval_kind", "contract", "columns_registered",
                     "overall", "by", "model_execution_path",
                     "delta_edit_roll_guard", "per_sample", "column_calls",
                     "column_sources"}
    expect_keys(held, expected_keys, "matched heldout")
    if (held.get("n") != 4560 or held.get("eval_kind") != "final" or
            held.get("contract") != "predicted_lut" or
            held.get("columns_registered") != ALIGN_COLUMNS or
            held.get("model_execution_path") !=
            "restore_predicted_lut(ProductionBatch)" or
            list(held.get("column_calls", {})) != ALIGN_COLUMNS or
            any(held["column_calls"].get(c) != 4560 for c in ALIGN_COLUMNS) or
            held.get("column_sources") != ALIGN_COLUMN_SOURCES):
        die("matched heldout final/contract/六列/calls/sources 不闭合")
    roll = held.get("delta_edit_roll_guard", {})
    if (set(roll) != {"n", "natural_identity_exceptions",
                      "active_set_preserved", "inactive_bitexact"} or
            roll.get("n") != 4560 or roll.get("active_set_preserved") is not True or
            roll.get("inactive_bitexact") is not True or
            not isinstance(roll.get("natural_identity_exceptions"), int)):
        die("matched heldout delta_edit_roll_guard 不闭合")
    by = held.get("by")
    if not isinstance(by, dict) or list(by) != ALIGN_STRATA:
        die("matched heldout strata 顺序不闭合")
    samples = held.get("per_sample")
    if not isinstance(samples, list) or len(samples) != 4560:
        die("matched heldout per_sample != 4560")
    sample_keys = {"id", "depth", "geom", "rec_band", "n_eval_pixels",
                   "model", "identity", "latent_cos", "latent_l2",
                   "delta_const", "delta_shuffle", "delta_edit_null",
                   "delta_edit_roll"}
    err_keys = {"n", "p50", "p95", "p99", "mean", "max"}
    ctl_keys = {"value", "control_p50"}
    ids, counts = set(), {s: {} for s in ALIGN_STRATA}
    for i, rec in enumerate(samples):
        if not isinstance(rec, dict) or set(rec) != sample_keys:
            die(f"matched heldout per_sample[{i}] schema 不闭合")
        sid = rec.get("id")
        if not isinstance(sid, str) or not sid:
            die(f"matched heldout per_sample[{i}] id 非空字符串守卫失败")
        if rec.get("depth") not in (4, 5, 6):
            die(f"matched heldout per_sample[{i}] depth 不在4/5/6")
        token = (sid, int(rec["depth"]))
        if token in ids:
            die(f"matched heldout per_sample[{i}] (id,depth) 非唯一")
        ids.add(token)
        npx = rec.get("n_eval_pixels")
        if not isinstance(npx, int) or isinstance(npx, bool) or npx <= 0:
            die(f"matched heldout per_sample[{i}] n_eval_pixels 非正整数")
        for col in ("model", "identity"):
            cell = rec[col]
            if not isinstance(cell, dict) or set(cell) != err_keys or cell["n"] != npx:
                die(f"matched heldout per_sample[{i}].{col} schema/n 不闭合")
            for field in err_keys - {"n"}:
                finite_number(cell[field], f"matched[{i}].{col}.{field}")
        for col in ALIGN_COLUMNS[2:]:
            cell = rec[col]
            if not isinstance(cell, dict) or set(cell) != ctl_keys:
                die(f"matched heldout per_sample[{i}].{col} schema 不闭合")
            for field in ctl_keys:
                finite_number(cell[field], f"matched[{i}].{col}.{field}")
        finite_number(rec["latent_cos"], f"matched[{i}].latent_cos")
        finite_number(rec["latent_l2"], f"matched[{i}].latent_l2")
        for stratum in ALIGN_STRATA:
            label = str(rec[stratum])
            counts[stratum][label] = counts[stratum].get(label, 0) + 1
    for stratum in ALIGN_STRATA:
        cells = by[stratum]
        if (not isinstance(cells, dict) or
                cells != {k: {"n": v} for k, v in sorted(counts[stratum].items())} or
                sum(x["n"] for x in cells.values()) != 4560):
            die(f"matched heldout by.{stratum} 未由per_sample闭合")
    overall = held.get("overall")
    expected_overall = {"model", "identity", *ALIGN_COLUMNS[2:],
                        "latent_cos", "latent_l2"}
    if not isinstance(overall, dict) or set(overall) != expected_overall:
        die("matched heldout overall schema 不闭合")
    for col in ("model", "identity"):
        if set(overall[col]) != {"p50", "p95", "p99"}:
            die(f"matched heldout overall.{col} schema 不闭合")
        for field in ("p50", "p95", "p99"):
            derived = float(np.median([x[col][field] for x in samples]))
            if overall[col][field] != derived:
                die(f"matched heldout overall.{col}.{field} 未由per_sample闭合")
    for col in ALIGN_COLUMNS[2:]:
        if set(overall[col]) != ctl_keys:
            die(f"matched heldout overall.{col} schema 不闭合")
        for field in ctl_keys:
            derived = float(np.median([x[col][field] for x in samples]))
            if overall[col][field] != derived:
                die(f"matched heldout overall.{col}.{field} 未由per_sample闭合")
    for col in ("latent_cos", "latent_l2"):
        derived = float(np.median([x[col] for x in samples]))
        if set(overall[col]) != {"value"} or overall[col]["value"] != derived:
            die(f"matched heldout overall.{col} 未由per_sample闭合")
    model = overall["model"]
    return dict(name="T-ALIGN A-frozen", contract="predicted_lut", n=4560,
                step=4688, linf8_p50=float(model["p50"]),
                linf8_p95=float(model["p95"]), linf8_p99=float(model["p99"]))


def reference_row(run: Path, name: str, contract: str) -> dict:
    p = run / "metrics.json"
    m, raw, digest = read_stable_json(p, f"对照 {name}")
    held = m.get("heldout", {})
    actual_contract = m.get("contract", (m.get("edit") or {}).get("contract", "none"))
    if actual_contract != contract:
        die(f"对照 {name} 实际 contract={actual_contract!r} != {contract!r}")
    if int(m.get("n_heldout", -1)) != 4560 or int(held.get("n", -1)) != 4560:
        die(f"对照 {name} held-out 不是 4560")
    if held.get("eval_kind") != "final":
        die(f"对照 {name} heldout.eval_kind != final")
    if (name == "CHAINEND" and
            int(held.get("step", -1)) != int(m.get("steps", -2))):
        die(f"对照 {name} heldout.step 与 metrics.steps 不闭合")
    if (name == "CHAINEND" and
            (m.get("epr") != "EPR-051/stage0/sprf" or
             m.get("arm") != "arm_chain_end" or m.get("smoke") is not False or
             int(m.get("steps", -1)) != int(m.get("effective_max_steps", -2)))):
        die("对照 CHAINEND formal/epr/arm/step 不闭合")
    if (name == "T-ALIGN A-frozen" and
            (m.get("epr") != "EPR-051/stage0/sprf/T-ALIGN" or
             m.get("smoke") is not False or m.get("mode") != "frozen" or
             int(m.get("steps", -1)) != 4688 or
             int(m.get("predictor_params", -1)) != 1713280 or
             int(m.get("n_train", -1)) != 300000)):
        die("对照 T-ALIGN A-frozen 不是冻结 formal matched baseline")
    v = held["overall"]["model"]
    for q in ("p50", "p95", "p99"):
        finite_number(v.get(q), f"{name} overall.model.{q}")
    return dict(name=name, contract=actual_contract, n=4560,
                step=int(m["steps"]),
                linf8_p50=float(v["p50"]), linf8_p95=float(v["p95"]),
                linf8_p99=float(v["p99"]), metrics=str(p),
                metrics_sha256=digest, metrics_bytes=len(raw))


def matched_baseline_row(run: Path, cfg: T0.Cfg,
                         expected_readiness: dict) -> dict:
    """Stable-raw plus full semantic/provenance validation of A-frozen."""
    run = run.resolve()
    metrics_path = run / "metrics.json"
    metrics, metrics_raw, metrics_sha = read_stable_json(
        metrics_path, "matched baseline metrics")
    expected_metrics_sha = cfg.str_("compare", "matched_metrics_sha256")
    if metrics_sha != expected_metrics_sha:
        die("matched baseline metrics stable raw sha 不闭合")
    top = {"epr", "generated", "smoke", "mode", "contract", "n_train",
           "n_heldout", "steps", "predictor_params", "feat_dim", "hist_grid",
           "assertions", "a8", "contract_rows", "heldout", "train_history",
           "wall_seconds"}
    expect_keys(metrics, top, "matched baseline metrics")
    if (metrics.get("epr") != ALIGN_EPR or metrics.get("mode") != "frozen" or
            metrics.get("contract") != "predicted_lut" or
            metrics.get("smoke") is not False or metrics.get("n_train") != 300000 or
            metrics.get("n_heldout") != 4560 or metrics.get("steps") != 4688 or
            metrics.get("predictor_params") != 1713280 or
            metrics.get("feat_dim") != 1032 or metrics.get("hist_grid") != 8):
        die("matched baseline top identity 不闭合")
    validate_align_guard_bundle(metrics.get("assertions"), require_state_sha=True)
    if metrics["assertions"].get("readiness") != expected_readiness:
        die("matched baseline metrics assertions.readiness 不闭合")
    row = validate_align_heldout(metrics.get("heldout"))
    history = metrics.get("train_history")
    if (not isinstance(history, list) or not history or
            history[0].get("step") != 1 or history[-1].get("step") != 4688 or
            any(not isinstance(x.get("step"), int) for x in history) or
            any(a["step"] >= b["step"] for a, b in zip(history, history[1:]))):
        die("matched baseline train_history 1..4688 不闭合")

    provenance_path = run / "provenance.json"
    provenance, provenance_raw, provenance_sha = read_stable_json(
        provenance_path, "matched baseline provenance")
    if provenance_sha != cfg.str_("compare", "matched_provenance_sha256"):
        die("matched baseline provenance stable raw sha 不闭合")
    expect_keys(provenance, {"epr", "contract", "a8", "outputs"},
                "matched baseline provenance")
    if (provenance.get("epr") != ALIGN_EPR or
            provenance.get("contract") != "predicted_lut"):
        die("matched baseline provenance identity 不闭合")
    outputs = provenance.get("outputs")
    output_names = ["predictor.pt", "target_latents.pt", "run_args.json",
                    "train_meta.json", "metrics.json", "subset_report.json"]
    if not isinstance(outputs, dict) or list(outputs) != output_names:
        die("matched baseline provenance outputs schema/order 不闭合")
    for name in output_names:
        path = run / name
        if not path.is_file() or sha256_file(path) != outputs[name]:
            die(f"matched baseline provenance output sha 不闭合: {name}")
    if outputs["metrics.json"] != metrics_sha:
        die("matched baseline provenance metrics sha != stable raw")

    a8 = metrics.get("a8")
    a8_keys = {"source_sha256", "align_config", "backbone_run_args",
               "backbone_config", "backbone_ckpt", "target_latents",
               "subset", "readiness"}
    expect_keys(a8, a8_keys, "matched baseline a8")
    if provenance.get("a8") != a8:
        die("matched baseline metrics/provenance a8 不逐位相同")
    expected_sources = {
        "train_align.py": _P.src("train_align.py"),
        "align_predictor.py": _P.src("align_predictor.py"),
        "archive_assets.py": _P.src("archive_assets.py"),
        "train_sprf_xl.py": _P.src("train_sprf_xl.py"),
        "stage_flow.py": _P.src("stage_flow.py"),
        "stage_solver.py": _P.src("stage_solver.py"),
        "stage_targets.py": _P.src("stage_targets.py"),
        "train_sprf.py": _P.src("train_sprf.py"),
        "train_stage0.py": _P.src("train_stage0.py"),
    }
    if (list(a8.get("source_sha256", {})) != list(expected_sources) or
            any(sha256_file(path) != a8["source_sha256"].get(name)
                for name, path in expected_sources.items())):
        die("matched baseline source file sha 不闭合")
    align_cfg = a8.get("align_config", {})
    align_cfg_path = _P.SPRF_LEGACY / "configs" / "align_frozen.toml"
    if (align_cfg.get("path") != str(align_cfg_path.resolve()) or
            sha256_file(align_cfg_path) != align_cfg.get("sha256")):
        die("matched baseline align_frozen config sha/path 不闭合")
    for label, rec in (("backbone_run_args", a8.get("backbone_run_args", {})),
                       ("backbone_config", a8.get("backbone_config", {})),
                       ("backbone_ckpt", a8.get("backbone_ckpt", {}))):
        path = Path(rec.get("path", ""))
        if not path.is_file() or sha256_file(path) != rec.get("sha256"):
            die(f"matched baseline {label} sha/path 不闭合")
    target = a8.get("target_latents", {})
    if (target.get("path") != str((run / "target_latents.pt").resolve()) or
            target.get("file_sha256") != outputs["target_latents.pt"] or
            target.get("shape") != [4052, 128] or
            target.get("indexed_by") != "lut_id/row"):
        die("matched baseline target_latents provenance 不闭合")
    expected_subset = dict(
        n=300000, sha256=expected_readiness["subset"]["keys_sha256"],
        heldout_n=4560,
        heldout_ids_sha256=expected_readiness["subset"]["heldout_ids_sha256"])
    if a8.get("subset") != expected_subset:
        die("matched baseline a8 subset 300k/4560 sha 不闭合")
    if a8.get("readiness") != expected_readiness:
        die("matched baseline readiness 与当前冻结输入不闭合")

    train_meta, train_meta_raw, train_meta_sha = read_stable_json(
        run / "train_meta.json", "matched baseline train_meta")
    if train_meta_sha != outputs["train_meta.json"]:
        die("matched baseline train_meta sha 不闭合")
    train_top = {"epr", "mode", "contract", "backbone", "backbone_step",
                 "n_train", "steps", "one_pass", "predictor_params", "feat_dim",
                 "hist_grid", "assertions", "config", "history", "wall_seconds"}
    expect_keys(train_meta, train_top, "matched baseline train_meta")
    if (train_meta.get("epr") != ALIGN_EPR or train_meta.get("mode") != "frozen" or
            train_meta.get("contract") != "predicted_lut" or
            train_meta.get("n_train") != 300000 or train_meta.get("steps") != 4688 or
            train_meta.get("one_pass") != 4688 or
            train_meta.get("predictor_params") != 1713280 or
            train_meta.get("feat_dim") != 1032 or train_meta.get("hist_grid") != 8 or
            train_meta.get("history") != history or
            train_meta.get("config") != T0.Cfg(align_cfg_path).d):
        die("matched baseline train_meta identity/config/history 不闭合")
    validate_align_guard_bundle(train_meta.get("assertions"),
                                require_state_sha=False)
    if train_meta["assertions"].get("readiness") != expected_readiness:
        die("matched baseline train_meta readiness 不闭合")

    run_args, run_args_raw, run_args_sha = read_stable_json(
        run / "run_args.json", "matched baseline run_args")
    if (run_args_sha != outputs["run_args.json"] or run_args.get("epr") != ALIGN_EPR or
            run_args.get("mode") != "frozen" or
            run_args.get("contract") != "predicted_lut" or
            run_args.get("n_train") != 300000 or run_args.get("n_heldout") != 4560 or
            run_args.get("a8") != a8 or
            run_args.get("invocation") != {
                "smoke": False, "limit_samples": 0, "stop_after": 0,
                "device": "cuda:0", "out_dir": str(run)}):
        die("matched baseline run_args formal identity/a8 不闭合")
    # run_args is written before training, so only pre-training guards live here.
    ra_assert = run_args.get("assertions", {})
    if (ra_assert.get("B1_backbone_frozen") is not True or
            ra_assert.get("B3_heldout_isolated") is not True or
            ra_assert.get("B4_injection_bitexact") !=
            metrics["assertions"]["B4_injection_bitexact"] or
            ra_assert.get("production_guard") !=
            metrics["assertions"]["production_guard"] or
            ra_assert.get("delta_edit_roll_helper") !=
            metrics["assertions"]["delta_edit_roll_helper"]):
        die("matched baseline run_args pre-training guards 不闭合")

    contract_rows = metrics.get("contract_rows")
    if (not isinstance(contract_rows, list) or len(contract_rows) != 3 or
            contract_rows[0] != row or
            contract_rows[1:] != expected_readiness["references"]):
        die("matched baseline contract rows 未由深验结果/冻结引用闭合")
    expected_p50 = cfg.num("compare", "matched_linf8_p50")
    if abs(row["linf8_p50"] - expected_p50) > 1e-10:
        die("matched baseline p50 不闭合")
    # End-of-validation stable raw re-read closes replacement races.
    if (metrics_path.read_bytes() != metrics_raw or
            provenance_path.read_bytes() != provenance_raw or
            (run / "train_meta.json").read_bytes() != train_meta_raw or
            (run / "run_args.json").read_bytes() != run_args_raw):
        die("matched baseline artifact 在深验期间发生变化")
    row.update(metrics=str(metrics_path), metrics_sha256=metrics_sha,
               metrics_bytes=len(metrics_raw), provenance=str(provenance_path),
               provenance_sha256=provenance_sha)
    return row


def readiness_preflight(cfg: T0.Cfg, smoke: bool) -> dict:
    """在任何 out_dir/shards/train 前深验冻结的 XL_FINALIZED 消费链。"""
    if smoke:
        return dict(required=False, passed=False, bypassed=True,
                    reason="--smoke 显式豁免 XL_FINALIZED readiness",
                    finalized_path=None, finalized_sha256=None)

    finalized_path = Path(cfg.str_("align", "xl_finalized"))
    final, final_raw, final_sha = read_stable_json(
        finalized_path, "XL_FINALIZED")
    final_keys = {"schema", "epr", "state", "terminal_reason", "generated",
                  "rule", "interval", "decision", "checkpoint", "full_metrics",
                  "evaluator", "runtime_dependencies", "controller_manifest",
                  "status", "subset", "producer_metrics", "original_payload"}
    expect_keys(final, final_keys, "XL_FINALIZED")
    if (final.get("schema") != XL_FINALIZED_SCHEMA or
            final.get("epr") != XL_EPR or final.get("state") != "evaluated" or
            final.get("terminal_reason") != "max_steps"):
        die("formal readiness XL_FINALIZED schema/epr/state/terminal 不闭合")
    if final.get("decision", {}).get("selected_step") != 8000:
        die("formal readiness XL_FINALIZED decision.selected_step != 8000")
    rule = final.get("rule", {})
    if (rule.get("metric") != "overall.model.p50" or
            rule.get("direction") != "min" or rule.get("min_delta") != 0.05 or
            rule.get("patience") != 5):
        die("formal readiness XL_FINALIZED rule != p50/min/.05/patience5")

    evaluator = final.get("evaluator", {})
    expect_keys(evaluator, {"path", "sha256", "config", "run_args",
                            "xl_provenance"}, "evaluator")
    for key in ("config", "run_args", "xl_provenance"):
        expect_keys(evaluator.get(key), {"path", "sha256"}, f"evaluator.{key}")
    config_raw, config_digest = read_stable_bytes(
        Path(evaluator["config"]["path"]), "evaluator.config")
    if config_digest != evaluator["config"]["sha256"]:
        die("formal readiness evaluator.config bytes sha 不闭合")
    for label, record in (("evaluator.run_args", evaluator["run_args"]),
                          ("evaluator.xl_provenance", evaluator["xl_provenance"])):
        path = Path(record["path"])
        _, raw, digest = read_stable_json(path, label)
        if digest != record["sha256"]:
            die(f"formal readiness {label} bytes sha 不闭合")
        if label.endswith("run_args"):
            run_args = json.loads(raw)
            run_args_raw = raw
        elif label.endswith("xl_provenance"):
            xl_prov = json.loads(raw)
            xl_raw = raw
    config_path = Path(evaluator["config"]["path"])
    config_sha = evaluator["config"]["sha256"]
    producer_run = Path(evaluator["run_args"]["path"]).parent
    if Path(evaluator["run_args"]["path"]).name != "run_args.json":
        die("formal readiness evaluator.run_args 不是 producer run_args.json")
    xl_cfg = T0.Cfg(config_path)
    if (run_args.get("config_path") != str(config_path.resolve()) or
            run_args.get("frozen_sha256", {}).get("config") != config_sha or
            run_args.get("config") != xl_cfg.d or
            run_args.get("arm") != "arm_clut_xl" or
            run_args.get("eval", {}).get("columns_registered") != XL_COLUMNS or
            run_args.get("eval", {}).get("strata") != XL_STRATA or
            run_args.get("edit", {}).get("condition") != "inv_lut" or
            run_args.get("edit", {}).get("contract") != "oracle_lut"):
        die("formal readiness evaluator run_args 不是冻结 XL/26列/oracle_lut")
    schedule = run_args.get("schedule", {})
    ev = run_args.get("eval", {})
    if (schedule.get("config_max_steps") != 8000 or
            schedule.get("effective_max_steps") != 8000 or
            schedule.get("eval_every") != 400 or ev.get("interval_n") != 128 or
            ev.get("final_n") != 4560):
        die("formal readiness evaluator run_args schedule/eval 不闭合")
    if (xl_prov.get("config") != str(config_path.resolve()) or
            xl_prov.get("config_sha256") != config_sha or
            xl_prov.get("subset") != xl_cfg.d.get("subset")):
        die("formal readiness xl_provenance config 不闭合")

    runtime = final.get("runtime_dependencies", {})
    if set(runtime) != XL_RUNTIME_DEPENDENCIES:
        die("formal readiness runtime_dependencies key 集不闭合")
    for name, record in runtime.items():
        expect_keys(record, {"path", "sha256", "attested_by"},
                    f"runtime_dependencies.{name}")
        path = Path(record["path"])
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            die(f"formal readiness runtime dependency bytes sha 不闭合: {name}")
        frozen_digest = run_args.get("frozen_sha256", {}).get(name)
        if frozen_digest is not None and frozen_digest != record["sha256"]:
            die(f"formal readiness runtime/run_args frozen sha 不闭合: {name}")
        xl_digest = xl_prov.get("sha256", {}).get(f"{name}.py")
        if xl_digest is not None and xl_digest != record["sha256"]:
            die(f"formal readiness runtime/xl_provenance sha 不闭合: {name}")
    if (evaluator["path"] != runtime["eval_sprf_checkpoint"]["path"] or
            evaluator["sha256"] != runtime["eval_sprf_checkpoint"]["sha256"]):
        die("formal readiness evaluator 与 runtime dependency 不闭合")

    ctl, ctl_raw, ctl_sha = verified_json_record(
        final.get("controller_manifest", {}), "controller_manifest")
    ctl_keys = {"schema", "epr", "mode", "generated", "state",
                "terminal_reason", "rule", "interval", "run_schema",
                "decision", "checkpoint", "provenance"}
    expect_keys(ctl, ctl_keys, "controller_manifest")
    if (ctl.get("schema") != XL_CONTROLLER_SCHEMA or ctl.get("epr") != XL_EPR or
            ctl.get("mode") != "freeze" or ctl.get("state") != "checkpoint_frozen" or
            ctl.get("terminal_reason") != "max_steps" or ctl.get("rule") != rule or
            ctl.get("interval") != final.get("interval") or
            ctl.get("decision") != final.get("decision")):
        die("formal readiness controller natural-max 身份/证据不闭合")
    decision = ctl["decision"]
    if (decision.get("triggered") is not False or
            decision.get("trigger_step") is not None or
            decision.get("selected_step") != 8000 or
            decision.get("complete_interval_count") != 20 or
            decision.get("intervals_through_trigger") != 20 or
            len(decision.get("evidence", [])) != 20):
        die("formal readiness controller decision 不是自然终态20点")
    interval = ctl.get("interval", {})
    expect_keys(interval, {"path", "sha256", "sequence"}, "controller.interval")
    interval_path = Path(interval["path"])
    try:
        interval_raw = interval_path.read_bytes()
        interval_raw2 = interval_path.read_bytes()
        interval_rows = [json.loads(line) for line in interval_raw.splitlines()
                         if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        die(f"formal readiness interval evidence 不可读: {exc}")
    sequence = [{"step": x.get("step"),
                 "p50": x.get("overall", {}).get("model", {}).get("p50")}
                for x in interval_rows]
    if (interval_raw2 != interval_raw or not interval_raw.endswith(b"\n") or
            hashlib.sha256(interval_raw).hexdigest() != interval["sha256"] or
            [x.get("step") for x in interval_rows] != list(range(400, 8001, 400)) or
            any(x.get("n") != 128 or x.get("eval_kind") != "interval" or
                set(x.get("overall", {})) != set(XL_COLUMNS) for x in interval_rows) or
            interval["sequence"] != sequence):
        die("formal readiness interval 不是稳定完整20x400/n128/26列")
    run_schema = ctl.get("run_schema", {})
    if (run_schema.get("path") != evaluator["run_args"]["path"] or
            run_schema.get("sha256") != evaluator["run_args"]["sha256"] or
            run_schema.get("config_sha256") != config_sha or
            run_schema.get("max_steps") != 8000 or
            run_schema.get("eval_every") != 400 or
            run_schema.get("interval_n") != 128 or
            run_schema.get("final_n") != 4560 or
            run_schema.get("condition") != "inv_lut" or
            run_schema.get("contract") != "oracle_lut" or
            run_schema.get("columns_registered") != XL_COLUMNS):
        die("formal readiness controller run_schema 不闭合")
    expected_ctl_prov = {
        "run_args": dict(path=evaluator["run_args"]["path"],
                         sha256=evaluator["run_args"]["sha256"]),
        "xl_provenance": dict(path=evaluator["xl_provenance"]["path"],
                              sha256=evaluator["xl_provenance"]["sha256"]),
    }
    if ctl.get("provenance") != expected_ctl_prov:
        die("formal readiness controller provenance 不闭合")

    status_record = final.get("status", {})
    expect_keys(status_record, {"path", "sha256", "name", "phase", "rc",
                                "pid", "ended_at"}, "status")
    status, status_raw, status_sha = read_stable_json(
        Path(status_record["path"]), "status")
    if (status_sha != status_record["sha256"] or
            status_record.get("name") != "SPRF_CLUTXL_S1" or
            status_record.get("phase") != "done" or status_record.get("rc") != 0 or
            status_record.get("ended_at") is None or
            status.get("name") != "SPRF_CLUTXL_S1" or status.get("phase") != "done" or
            status.get("rc") != 0 or status.get("ended_at") is None or
            status.get("pid") != status_record.get("pid") or
            status.get("ended_at") != status_record.get("ended_at")):
        die("formal readiness status bytes/done/0/ended_at 不闭合")
    original = final.get("original_payload", {})
    expect_keys(original, {"status", "rc", "intentional_stop"}, "original_payload")
    if original != {"status": status, "rc": 0, "intentional_stop": None}:
        die("formal readiness original_payload 与自然终态 status 不闭合")

    subset = final.get("subset", {})
    subset_keys = {"keys_path", "keys_sha256", "train_n", "heldout_ids_path",
                   "heldout_ids_sha256", "heldout_id_n", "heldout_sample_n"}
    expect_keys(subset, subset_keys, "subset")
    keys, keys_sha = semantic_json_list(Path(subset["keys_path"]), "subset keys")
    held_ids, held_sha = semantic_json_list(
        Path(subset["heldout_ids_path"]), "heldout ids")
    frozen_subset = xl_cfg.d.get("subset", {})
    expected_subset = dict(
        keys_path=frozen_subset.get("keys_file"),
        keys_sha256=frozen_subset.get("sha256"), train_n=300000,
        heldout_ids_path=frozen_subset.get("heldout_ids_file"),
        heldout_ids_sha256=frozen_subset.get("heldout_ids_sha256"),
        heldout_id_n=1558, heldout_sample_n=4560)
    if (subset != expected_subset or xl_prov.get("subset") != frozen_subset or
            len(keys) != 300000 or subset.get("train_n") != 300000 or
            keys_sha != subset.get("keys_sha256") or len(held_ids) != 1558 or
            subset.get("heldout_id_n") != 1558 or
            held_sha != subset.get("heldout_ids_sha256") or
            subset.get("heldout_sample_n") != 4560):
        die("formal readiness subset 300k/1558/4560 semantic sha 不闭合")

    canonical_record = final.get("full_metrics", {})
    expect_keys(canonical_record,
                {"path", "sha256", "step", "contract", "n",
                 "columns_registered", "ledger", "source"}, "full_metrics")
    canonical, canonical_raw, canonical_sha = read_stable_json(
        Path(canonical_record["path"]), "canonical full_metrics")
    if (Path(canonical_record["path"]).resolve() !=
            finalized_path.parent.resolve() / "metrics_full.json" or
            canonical_sha != canonical_record["sha256"]):
        die("formal readiness canonical full_metrics bytes sha 不闭合")
    canonical_keys = {"schema", "epr", "terminal_reason", "generated", "smoke",
                      "checkpoint", "contract", "n", "columns_registered",
                      "ledger", "source", "provenance", "wall_seconds", "heldout"}
    expect_keys(canonical, canonical_keys, "canonical full_metrics")
    if (canonical.get("schema") != XL_CANONICAL_SCHEMA or
            canonical.get("epr") != XL_EPR or canonical.get("terminal_reason") != "max_steps" or
            canonical.get("smoke") is not False or canonical.get("contract") != "oracle_lut" or
            canonical.get("n") != 4560 or canonical.get("columns_registered") != XL_COLUMNS or
            canonical.get("checkpoint") != final.get("checkpoint") or
            canonical.get("ledger") != canonical_record["ledger"] or
            canonical.get("source") != canonical_record["source"]):
        die("formal readiness canonical header/checkpoint/source 不闭合")
    canonical_prov = canonical.get("provenance", {})
    expected_canonical_prov = dict(
        config=dict(path=str(config_path.resolve()), sha256=config_sha),
        run_args=dict(path=evaluator["run_args"]["path"],
                      sha256=evaluator["run_args"]["sha256"],
                      bytes=len(run_args_raw)),
        xl_provenance=dict(path=evaluator["xl_provenance"]["path"],
                           sha256=evaluator["xl_provenance"]["sha256"],
                           bytes=len(xl_raw)),
        controller=dict(path=final["controller_manifest"]["path"],
                        sha256=ctl_sha, bytes=len(ctl_raw)),
        status=dict(path=status_record["path"], sha256=status_sha,
                    bytes=len(status_raw), phase="done", rc=0,
                    ended_at=status_record["ended_at"]),
        subset=subset, runtime_dependencies=runtime)
    if canonical_prov != expected_canonical_prov:
        die("formal readiness canonical provenance 全链不闭合")
    if (canonical_record.get("step") != 8000 or
            canonical_record.get("contract") != "oracle_lut" or
            canonical_record.get("n") != 4560 or
            canonical_record.get("columns_registered") != XL_COLUMNS):
        die("formal readiness full_metrics record 8000/oracle/4560/26列不闭合")
    ledger = canonical_record.get("ledger", {})
    expect_keys(ledger, {"expected_calls", "column_calls", "column_missing",
                         "column_available", "column_sources", "strata"},
                "full_metrics.ledger")
    computed_ledger = validate_xl_heldout(canonical.get("heldout", {}),
                                          require_contract=False)
    if (computed_ledger != ledger or canonical.get("ledger") != ledger or
            ledger.get("expected_calls") != 4560 or ledger.get("strata") != XL_STRATA or
            list(ledger.get("column_calls", {})) != XL_COLUMNS or
            any(ledger["column_calls"].get(c) != 4560 for c in XL_COLUMNS)):
        die("formal readiness heldout-derived/canonical/full_metrics ledger 不闭合")
    source = canonical_record.get("source", {})
    expect_keys(source, {"kind", "producer_metrics"}, "full_metrics.source")
    if (source.get("kind") != "producer_full_final_validated" or
            source.get("producer_metrics") != final.get("producer_metrics")):
        die("formal readiness canonical source 不是 validated producer full-final")

    producer_record = final.get("producer_metrics", {})
    expect_keys(producer_record, {"path", "sha256", "bytes", "smoke", "steps",
                                  "effective_max_steps"}, "producer_metrics")
    producer_path = Path(producer_record["path"])
    if producer_path.resolve() != producer_run.resolve() / "metrics.json":
        die("formal readiness producer metrics path 不是 producer_run/metrics.json")
    producer, producer_raw, producer_sha = read_stable_json(
        producer_path, "producer_metrics")
    assertions = producer.get("assertions", {})
    a1 = assertions.get("A1_detail", {})
    a5 = assertions.get("A5_nfe", {})
    a7 = assertions.get("A7_heldout_guard", {})
    a10 = assertions.get("A10_anchor_contract", {})
    a12 = assertions.get("A12_edit_wiring", {})
    assertions_ok = (
        assertions.get("A1_step0_identity") is True and a1.get("bit_exact") is True and
        a5.get("nfe_per_stage") == 1 and a5.get("expected") == a5.get("n_stages") and
        assertions.get("A6_no_lut_restore") is True and
        a7.get("n_heldout_ids") == 1558 and
        isinstance(a7.get("checked_batches"), int) and
        a10.get("rollout_anchor") == "asset_y" and
        a10.get("teacher_anchor") == "chain_r" and
        a10.get("teacher_is_chain_state") is True and
        a10.get("rollout_is_asset_y") is True and
        a12.get("a12a_film_edit_slice", {}).get("grad_abs_sum", 0) > 0 and
        a12.get("a12b_edit_encoder", {}).get("grad_l2", 0) > 0 and
        a12.get("a12b_edit_encoder", {}).get("params_without_grad") == 0 and
        bool(assertions.get("A2_teacher_spotchecks")))
    if (producer_sha != producer_record["sha256"] or
            len(producer_raw) != producer_record["bytes"] or
            producer_record.get("smoke") is not False or
            producer_record.get("steps") != 8000 or
            producer_record.get("effective_max_steps") != 8000 or
            producer.get("smoke") is not False or producer.get("steps") != 8000 or
            producer.get("effective_max_steps") != 8000 or
            producer.get("epr") != "EPR-051/stage0/sprf" or
            producer.get("arm") != "arm_clut_xl" or
            producer.get("n_train") != 300000 or producer.get("n_heldout") != 4560 or
            producer.get("columns_registered") != XL_COLUMNS or
            producer.get("frozen_sha256") != run_args.get("frozen_sha256") or
            producer.get("run_args") != "run_args.json" or
            producer.get("edit", {}).get("contract") != "oracle_lut" or
            producer.get("heldout") != canonical.get("heldout") or not assertions_ok):
        die("formal readiness producer metrics bytes/formal/heldout 不闭合")
    producer_ledger = validate_xl_heldout(producer.get("heldout", {}),
                                          require_contract=False)
    if producer_ledger != ledger:
        die("formal readiness producer heldout-derived ledger 不闭合")
    heldout = canonical.get("heldout", {})
    model_values = heldout.get("overall", {}).get("model", {})

    cp_record = final.get("checkpoint", {})
    ctl_cp = ctl.get("checkpoint", {})
    expect_keys(ctl_cp, {"source_path", "source_sha256", "selected_path",
                         "sha256", "content_addressed_path", "step",
                         "config_path", "config_sha256"},
                "controller.checkpoint")
    if (ctl_cp.get("selected_path") != cp_record.get("path") or
            ctl_cp.get("sha256") != cp_record.get("sha256") or
            ctl_cp.get("source_path") != cp_record.get("source_path") or
            ctl_cp.get("source_sha256") != cp_record.get("source_sha256") or
            ctl_cp.get("content_addressed_path") != cp_record.get("content_addressed_path") or
            ctl_cp.get("step") != cp_record.get("step") or
            Path(ctl_cp.get("config_path", "")).resolve() != config_path.resolve() or
            ctl_cp.get("config_sha256") != cp_record.get("config_sha256")):
        die("formal readiness finalized/controller checkpoint 不闭合")
    cp, cp_path, cp_sha = snapshot_formal_checkpoint(
        cp_record, finalized_path, producer_run, config_sha,
        run_args.get("frozen_sha256", {}))

    xl = dict(name="C-LUT-XL", contract="oracle_lut", n=4560, step=8000,
              linf8_p50=float(model_values["p50"]),
              linf8_p95=float(model_values["p95"]),
              linf8_p99=float(model_values["p99"]),
              metrics=canonical_record["path"], metrics_sha256=canonical_sha)
    none = reference_row(Path(cfg.str_("compare", "none_run")),
                         "CHAINEND", "none")
    baseline_readiness = dict(
        required=True, passed=True, bypassed=False,
        finalized_path=str(finalized_path), finalized_sha256=final_sha,
        terminal_reason="max_steps", selected_step=8000,
        checkpoint=dict(path=str(cp_path), sha256=cp_sha, bytes=cp_record["bytes"],
                        content_addressed_path=cp_record["content_addressed_path"]),
        canonical_metrics=dict(path=canonical_record["path"], sha256=canonical_sha),
        controller_manifest=dict(path=final["controller_manifest"]["path"],
                                 sha256=ctl_sha),
        status=status_record, subset=subset, references=[xl, none])
    matched = matched_baseline_row(
        Path(cfg.str_("compare", "matched_run")), cfg, baseline_readiness)
    for label, path, expected_raw in (
            ("XL_FINALIZED", finalized_path, final_raw),
            ("controller_manifest", Path(final["controller_manifest"]["path"]), ctl_raw),
            ("status", Path(status_record["path"]), status_raw),
            ("canonical full_metrics", Path(canonical_record["path"]), canonical_raw),
            ("producer_metrics", producer_path, producer_raw)):
        try:
            current_raw = path.read_bytes()
        except OSError as exc:
            die(f"formal readiness 末尾复验 {label} 不可读: {exc}")
        if current_raw != expected_raw:
            die(f"formal readiness {label} 在 readiness 期间发生变化")
    return dict(
        **{k: v for k, v in baseline_readiness.items() if k != "references"},
        references=[matched, xl, none],
        _checkpoint_payload=cp, _run_args=run_args,
        _run_args_path=evaluator["run_args"]["path"],
        _run_args_sha256=evaluator["run_args"]["sha256"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--limit-samples", type=int, default=0)
    ap.add_argument("--stop-after", type=int, default=0)
    ap.add_argument("--device", default="",
                    help="仅 --smoke 可覆盖设备")
    ap.add_argument("--out-dir", default="",
                    help="仅 --smoke 可覆盖输出目录")
    a = ap.parse_args()

    if not a.smoke and (a.limit_samples or a.stop_after):
        die("--limit-samples/--stop-after 仅供 --smoke；正式结果禁止截断")

    cfg = T0.Cfg(Path(a.config))
    if (a.device or a.out_dir) and not a.smoke:
        die("--device/--out-dir 仅供 --smoke，正式运行不可覆盖冻结 config")
    formal_out = Path(cfg.str_("run", "out_dir")).resolve()
    if a.smoke:
        if not a.out_dir or a.limit_samples <= 0 or a.stop_after <= 0:
            die("--smoke 必须显式给独立 --out-dir、正 --limit-samples/--stop-after")
        smoke_out = Path(a.out_dir).resolve()
        if smoke_out == formal_out or formal_out in smoke_out.parents:
            die("--smoke out_dir 不得等于或位于 formal out_dir 内")
    out_dir = Path(a.out_dir or cfg.str_("run", "out_dir"))
    dev = a.device or cfg.str_("run", "device")
    seed = cfg.int_("run", "seed")
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False

    mode = cfg.str_("align", "mode", ("frozen", "joint"))
    if mode != "frozen":
        die("本入口只实现 A-frozen；A-joint 是预注册消融行，本期不跑")
    contract = cfg.str_("align", "contract", ("predicted_lut",))
    grid = cfg.int_("align", "hist_grid")
    active_only = cfg.bool_("align", "loss_active_stages_only")
    readiness = readiness_preflight(cfg, bool(a.smoke))
    formal_cp = readiness.pop("_checkpoint_payload", None)
    formal_ra = readiness.pop("_run_args", None)
    formal_ra_path = readiness.pop("_run_args_path", None)
    formal_ra_sha = readiness.pop("_run_args_sha256", None)
    if a.smoke:
        bk_run = Path(cfg.str_("align", "smoke_backbone_run"))
        ra_path = bk_run / cfg.str_("align", "smoke_backbone_run_args")
        if not ra_path.is_file():
            die(f"smoke backbone 缺 {ra_path}")
        ra = json.loads(ra_path.read_text())
    else:
        if formal_cp is None or formal_ra is None or formal_ra_path is None:
            die("formal readiness 未返回 finalized checkpoint/run_args snapshot")
        ra_path = Path(formal_ra_path)
        bk_run = ra_path.parent
        ra = formal_ra
    cd = ra["config"]
    if cd.get("edit", {}).get("condition") != "inv_lut":
        die(f"backbone {bk_run} 不是 inv_lut 臂")
    sub = cd.get("subset", {})
    if not sub.get("enabled") or int(sub.get("n", -1)) != 300000:
        die(f"backbone {bk_run} 不是 clutxl300k 冻结口径")
    if int(sub.get("expect_heldout_samples", -1)) != 4560:
        die("backbone subset 没钉死 expect_heldout_samples=4560")
    mc = MiniCfg(cd)

    ST.install_compact_row(T0)
    AA.install(T0)
    if not getattr(T0, "_SPRF_ARCHIVE_INSTALLED", False):
        die("X5 FAILED: archive_assets 补丁没装上")
    subset_report = install_subset(
        T0, sub["keys_file"], sub["sha256"], int(sub["n"]),
        sub["heldout_ids_file"], sub["heldout_ids_sha256"],
        int(sub["expect_heldout_samples"]))
    T0.ASSET_MODE = cd["data"]["asset_source"]
    bcfg = T0.Cfg(Path(ra["config_path"]))
    samples, blobs, _ = T0.load_shards(bcfg)
    law = T0.bind_build_config(samples, cd["guard"]["build_config_sha256_allowed"])
    n_steps = law["n_steps"]
    roll_helper_assertion = assert_roll_helper(n_steps, dev)
    train_all = sorted([s for s in samples if not s["heldout"]],
                       key=lambda s: (s["id"], s["depth"] or 0))
    held_all = sorted([s for s in samples if s["heldout"]],
                      key=lambda s: (s["id"], s["depth"] or 0))
    if len(train_all) != 300000 or len(held_all) != 4560:
        die(f"X3/X4c FAILED: {len(train_all)} train / {len(held_all)} held-out")
    heldout_ids = {s["id"] for s in held_all}
    train, held = train_all, held_all
    if a.limit_samples:
        train = train[:a.limit_samples]
        # Δ_shuffle 的硬守卫要求跨 source_id；冒烟截断也不能把它退化掉。
        held, seen_sources = [], set()
        for s in held_all:
            src = T0.source_id_of(s["id"])
            if src in seen_sources:
                continue
            held.append(s); seen_sources.add(src)
            if len(held) >= a.limit_samples:
                break
    print(f"samples: {len(train)} train / {len(held)} held-out; chain {n_steps}",
          flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    owned = ["steps.jsonl", "predictor.pt", "target_latents.pt", "train_meta.json",
             "metrics.json", "run_args.json", "provenance.json",
             "subset_report.json"]
    exists = [x for x in owned if (out_dir / x).exists()]
    if exists:
        die(f"输出目录已有 {exists}（D-20）；先挪走，禁止覆盖")
    (out_dir / "subset_report.json").write_text(
        json.dumps(subset_report, ensure_ascii=False, indent=1))

    # ---- 冻结主干 -------------------------------------------------------- #
    in_dim = int(ra["encoder"]["dim"]) * len(cd["encoder"]["inputs"])
    model = SF.SprfModel(in_dim, mc, n_steps, cd["flow"]["alpha_mode"],
                         cd["data"]["depth_values"]).to(dev)
    inv = EC.InvLutSource(cd["edit"]["inv_cache_dir"], int(cd["edit"]["grid"]))
    model.load_inv_table(inv.load_table())
    backbone_config = Path(ra["config_path"])
    frozen_cfg_sha = ra.get("frozen_sha256", {}).get("config")
    if not frozen_cfg_sha or sha256_file(backbone_config) != frozen_cfg_sha:
        die("A8 FAILED: backbone 当前 config 与 run_args 冻结 sha 不同")
    if a.smoke:
        cp_path = bk_run / cfg.str_("align", "smoke_backbone_ckpt")
        cp, cp_sha = load_torch_hashed(cp_path)
    else:
        cp = formal_cp
        cp_path = Path(readiness["checkpoint"]["path"])
        cp_sha = readiness["checkpoint"]["sha256"]
    if cp.get("config_sha256") != frozen_cfg_sha:
        die("A8 FAILED: backbone ckpt 的 config_sha256 与 run_args 不同")
    if not a.smoke and int(cp.get("step", -1)) != 8000:
        die("formal readiness: finalized checkpoint step != 8000")
    model.load_state_dict(cp["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if any(p.requires_grad for p in model.parameters()):
        die("B1 FAILED: 主干仍有可训参数")
    bk_fingerprint = module_sha256(model)

    # ---- 对齐目标库：edit_enc_frozen(inv_table)，只是 lut_id 的函数 -------- #
    t0 = time.time()
    targets = AP.build_target_latents(model.cond.edit_enc,
                                      model.inv_table).detach()
    null_row = int(model.edit_null_row)
    target_sha = tensor_sha256(targets)
    T0._atomic_save(dict(
        target_latents=targets.cpu(), indexed_by="inv_table row / lut_id",
        null_row=null_row, backbone_ckpt=str(cp_path),
        backbone_ckpt_sha256=cp_sha, tensor_sha256=target_sha),
        out_dir / "target_latents.pt")
    print(f"[align] 目标 latent 库 {tuple(targets.shape)} "
          f"({time.time()-t0:.1f}s); null 行 = {null_row}; "
          f"|t| 均值 {float(targets.norm(dim=1).mean()):.4f}", flush=True)

    # ---- 注入：只在**实例**上覆盖，共享源码一个字节不动 -------------------- #
    frozen_edit_enc = model.cond.edit_enc
    model.edit_descriptor = lambda src: src
    model.cond.edit_enc = nn.Identity()
    model.edit_condition = "predicted_lut"
    model.edit_contract = "predicted_lut"

    feats_cache = {}
    want = {T0.feature_key(s) for s in train} | {T0.feature_key(s) for s in held}
    want_shards = {s["shard"] for s in train} | {s["shard"] for s in held}
    frozen_cache_files = [Path(x["file"]) for x in ra.get("encoder", {}).get("files", [])
                          if x.get("shard") in want_shards]
    cache_files = (sorted(frozen_cache_files) if frozen_cache_files else
                   sorted(Path(cd["encoder"]["cache_dir"]).glob("siglip_*.pt")))
    for f in cache_files:
        for k, v in torch.load(f, map_location="cpu")["features"].items():
            if k in want:
                feats_cache[k] = v
    missing_feats = want - set(feats_cache)
    if missing_feats:
        die(f"pooled c cache 缺 {len(missing_feats)} 个 key，例如 "
            f"{sorted(missing_feats)[:3]}")
    print(f"[align] pooled c cached: {len(feats_cache)}", flush=True)

    feat_dim = AP.feature_dim(grid)
    latent_dim = int(cd["edit"]["latent_dim"])
    hidden = cfg.int_("align", "hidden")
    layers = cfg.int_("align", "layers")
    # T0: construction-order proof in an isolated RNG fork.  The dummy
    # embedding in AP consumes baseline RNG before the shared trunk/head.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        paired_base = AP_BASE.LatentPredictor(
            in_dim, feat_dim, n_steps, latent_dim, hidden, layers)
        torch.manual_seed(seed)
        paired_time = AP.LatentPredictor(
            in_dim, feat_dim, n_steps, latent_dim, hidden, layers)
    base_state = paired_base.state_dict()
    time_state = paired_time.state_dict()
    nonstage_keys = sorted(k for k in base_state if not k.startswith("emb_stage."))
    if (set(nonstage_keys) != {k for k in time_state
                              if not (k.startswith("time_affine.") or
                                      k == "time_code")} or
            any(not torch.equal(base_state[k], time_state[k])
                for k in nonstage_keys)):
        die("T0 FAILED: paired-init non-stage state_dict 不逐位相同")
    base_nonstage_sha = named_tensors_sha256(
        (k, base_state[k]) for k in nonstage_keys)
    time_nonstage_sha = named_tensors_sha256(
        (k, time_state[k]) for k in nonstage_keys)
    t0_paired = dict(bitexact=True, tensor_keys=len(nonstage_keys),
                     baseline_nonstage_sha256=base_nonstage_sha,
                     timefilm_nonstage_sha256=time_nonstage_sha)
    if base_nonstage_sha != time_nonstage_sha:
        die(f"T0 FAILED: paired-init fingerprint {t0_paired}")
    del paired_base, paired_time, base_state, time_state

    pred = AP.LatentPredictor(in_dim, feat_dim, n_steps,
                              latent_dim, hidden, layers).to(dev)
    print(f"[align] predictor params = {pred.n_params():,}; feat_dim {feat_dim}",
          flush=True)

    # ---- T0--T4：stage-conditioning block replacement 守卫 ---------------- #
    q = pred.time_code
    pred_state_keys = list(pred.state_dict())
    q_ref = pred.proj_c.weight
    t1 = dict(shape=list(q.shape), finite=bool(torch.isfinite(q).all()),
              unique=int(torch.unique(q, dim=0).shape[0]),
              rank=int(torch.linalg.matrix_rank(q).item()),
              requires_grad=bool(q.requires_grad),
              device_match=bool(q.device == q_ref.device),
              dtype_match=bool(q.dtype == q_ref.dtype),
              in_state_dict=bool("time_code" in pred_state_keys))
    if (n_steps != 6 or t1["shape"] != [6, 3] or not t1["finite"] or
            t1["unique"] != 6 or
            t1["rank"] != 3 or t1["requires_grad"] or
            not t1["device_match"] or not t1["dtype_match"] or
            t1["in_state_dict"]):
        die(f"T1 FAILED: fixed time code {t1}")
    registered_embeddings = [name for name, module in pred.named_modules()
                             if isinstance(module, nn.Embedding)]
    time_affine_params = sum(p.numel() for p in pred.time_affine.parameters())
    t2 = dict(registered_embedding_count=len(registered_embeddings),
              time_affine_params=time_affine_params,
              predictor_params=pred.n_params(),
              time_code_in_state_dict=bool("time_code" in pred_state_keys),
              time_code_in_checkpoint_payload=bool(
                  "time_code" in pred.state_dict()))
    if (registered_embeddings or time_affine_params != 3072 or
            pred.n_params() != 1713280 or t2["time_code_in_state_dict"] or
            t2["time_code_in_checkpoint_payload"]):
        die(f"T2 FAILED: parameter replacement {t2}")
    print(f"[assert] T0 paired-init OK {t0_paired}; T1 fixed q OK {t1}; "
          f"T2 params OK {t2}", flush=True)

    lr = cfg.num("optim", "lr")
    batch = cfg.int_("optim", "batch")
    opt = torch.optim.AdamW(pred.parameters(), lr=lr,
                            weight_decay=cfg.num("optim", "weight_decay"),
                            betas=tuple(cfg.list_("optim", "betas", float)))
    clip = cfg.num("optim", "grad_clip")
    warmup = cfg.int_("optim", "warmup_steps")
    sched = cfg.str_("optim", "schedule", ("cosine", "constant"))
    w_lat = cfg.num("loss", "w_latent")
    sl1b = cfg.num("loss", "smooth_l1_beta")
    w_pix = cfg.num("loss", "w_pixel")
    if w_pix != 0.0:
        die("[loss] w_pixel != 0：端到端像素项本期预注册为 0，未实现")

    ds = AlignPixels(train, blobs, n_steps, grid, inv)
    one_pass = math.ceil(len(train) / batch)
    one_pass_cap = cfg.bool_("run", "one_pass_cap")
    max_steps = (min(cfg.int_("run", "max_steps"), one_pass)
                 if one_pass_cap else cfg.int_("run", "max_steps"))
    if not one_pass_cap or max_steps != one_pass:
        die(f"A-frozen 必须恰好一 pass：one_pass_cap={one_pass_cap}, "
            f"max_steps={max_steps}, one_pass={one_pass}")
    log_every = cfg.int_("run", "log_every")
    nw = 0 if a.smoke else int(cd["data"]["num_workers"])
    dl = DataLoader(ds, batch_size=batch, shuffle=True,
                    num_workers=nw,
                    collate_fn=collate, drop_last=False,
                    worker_init_fn=T0.worker_init, persistent_workers=nw > 0,
                    pin_memory=dev.startswith("cuda"))
    steps_p = out_dir / "steps.jsonl"

    # ---- B4：把**目标** latent 注入，必须与 oracle 查表逐位相同 ------------ #
    b0 = collate([ds[i] for i in range(min(2, len(train)))])
    with torch.no_grad():
        nb = b0["rows"].shape[0]
        gamma0, beta0 = pred.modulation()
        gamma_batch0 = gamma0.unsqueeze(0).expand(nb, -1, -1)
        beta_batch0 = beta0.unsqueeze(0).expand(nb, -1, -1)
        t4 = dict(batch=int(nb),
                  gamma_torch_equal_zero=bool(torch.equal(
                      gamma_batch0, torch.zeros_like(gamma_batch0))),
                  beta_torch_equal_zero=bool(torch.equal(
                      beta_batch0, torch.zeros_like(beta_batch0))),
                  gamma_max_abs=float(gamma_batch0.abs().max()),
                  beta_max_abs=float(beta_batch0.abs().max()),
                  weight_nonzero=int(torch.count_nonzero(
                      pred.time_affine.weight)))
        if (not t4["gamma_torch_equal_zero"] or
                not t4["beta_torch_equal_zero"] or
                t4["gamma_max_abs"] != 0.0 or t4["beta_max_abs"] != 0.0 or
                t4["weight_nonzero"] != 0):
            die(f"T4 FAILED: actual-batch zero-init modulation {t4}")
        rows = b0["rows"].to(dev)
        c0 = T0.gather_feats(feats_cache, b0["key"], 0, dev)
        base0 = model.cond.base(c0)
        d0 = b0["depth"].to(dev)
        # oracle 路径（原样查表 + 原 edit_enc）
        m_or = SF.SprfModel(in_dim, mc, n_steps, cd["flow"]["alpha_mode"],
                            cd["data"]["depth_values"]).to(dev)
        m_or.load_inv_table(inv.load_table())
        m_or.load_state_dict(cp["model"]); m_or.eval()
        # 每阶段以与 oracle rollout 完全相同的 B×descriptor 批形生成 runtime
        # latent；注入侧直接递这份 latent，oracle 侧仍走 descriptor->edit_enc。
        # 这才逐位验证注入路径，而不是把两边都弱化成查同一 latent cache。
        runtime_targets = torch.stack([
            m_or.cond.edit_enc(m_or.inv_table[rows[:, k]])
            for k in range(n_steps)
        ], dim=1)
        cache_max = float((runtime_targets - targets[rows]).abs().max())
        if not torch.allclose(runtime_targets, targets[rows], rtol=1e-5, atol=5e-7):
            die(f"B4 target cache 与 runtime edit_enc 不一致 (max {cache_max:.3e})")
        P = 256 if a.smoke else 4096
        y0 = torch.rand(nb, P, 3, device=dev)
        al0 = (torch.rand(nb, n_steps, P, device=dev) * 0.6
               * b0["mask"].to(dev).unsqueeze(-1))
        un0 = ST.union_mask_of(al0); ah0 = ST.compose_alpha_hat(al0)
        s0 = torch.ones(nb, device=dev)
        n_st = int(d0.max())
        o_or, _ = SS.rollout_with_metrics(m_or, m_or.cond.base(c0), y0, ah0, al0,
                                          un0, d0, s0, cd["flow"]["path_mode"],
                                          n_st, 1, edits=rows)
        o_in, _ = SS.rollout_with_metrics(model, base0, y0, ah0, al0, un0, d0, s0,
                                          cd["flow"]["path_mode"], n_st, 1,
                                          edits=runtime_targets)
        b4 = bool(torch.equal(o_or, o_in))
        b4_max = float((o_or - o_in).abs().max())
        del m_or
    if not b4:
        die(f"B4 FAILED: 注入目标 latent 与 oracle 查表不逐位相同 (max {b4_max:.3e})")
    print(f"[assert] T4 actual-batch identity OK {t4}", flush=True)
    print(f"[assert] B4 注入路径等价 OK (torch.equal, max|diff|={b4_max:g}); "
          f"target cache max|runtime-cache|={cache_max:.3e}", flush=True)

    # ---- 生产契约：只读 c/y/beta，且 oracle LUT 传入即拒绝 ----------------- #
    s_probe = train[0]
    row_probe = json.loads(blobs[s_probe["id"]])
    x_probe, y_probe = T0.load_pair(s_probe["shard"], row_probe,
                                    s_probe["after_asset"])
    mask_probe = T0.depth_mask(s_probe["depth"], n_steps)
    al_probe = (T0.alpha_fields(row_probe, x_probe).reshape(n_steps, -1)
                * mask_probe.unsqueeze(-1))[:, :256].unsqueeze(0).to(dev)
    y_probe = y_probe.reshape(-1, 3)[:256].unsqueeze(0).to(dev)
    d_probe = n_steps if s_probe["depth"] is None else int(s_probe["depth"])
    prod_batch = SS.ProductionBatch(dict(
        cond=T0.gather_feats(feats_cache, [T0.feature_key(s_probe)], 0, dev),
        y=y_probe, alphas=al_probe, union=ST.union_mask_of(al_probe),
        depth=torch.tensor([d_probe], device=dev),
        s=torch.tensor([float(row_probe["calib"]["s"])], device=dev)))
    pred.eval()
    prod_out = restore_predicted_lut(
        model, pred, prod_batch, cd["flow"]["path_mode"],
        int(cd["solver"]["nfe_per_stage"]), d_probe, grid, dev, contract)
    try:
        restore_predicted_lut(
            model, pred, prod_batch, cd["flow"]["path_mode"],
            int(cd["solver"]["nfe_per_stage"]), d_probe, grid, dev, contract,
            oracle_lut=model.inv_table[0])
    except SystemExit:
        production_rejects_oracle = True
    else:
        die("生产 predicted_lut 入口没有拒绝 oracle LUT")
    if prod_out.shape != y_probe.shape:
        die(f"生产 predicted_lut 输出形状 {tuple(prod_out.shape)} != y")
    pred.train()
    production_guard = dict(ok=True, rejects_oracle_lut=production_rejects_oracle,
                            production_keys=list(SS.PRODUCTION_KEYS),
                            output_shape=list(prod_out.shape))
    print("[assert] predicted_lut production guard OK; oracle LUT rejected", flush=True)

    source_files = {
        "train_align_time.py": Path(__file__).resolve(),
        "align_predictor_time.py": _P.src("align_predictor_time.py"),
        "align_predictor.py": _P.src("align_predictor.py"),
        "archive_assets.py": _P.src("archive_assets.py"),
        "train_sprf_xl.py": _P.src("train_sprf_xl.py"),
        "stage_flow.py": _P.src("stage_flow.py"),
        "stage_solver.py": _P.src("stage_solver.py"),
        "stage_targets.py": _P.src("stage_targets.py"),
        "train_sprf.py": _P.src("train_sprf.py"),
        "train_stage0.py": _P.src("train_stage0.py"),
    }
    a8 = dict(
        source_sha256={k: sha256_file(v) for k, v in source_files.items()},
        align_config=dict(path=str(Path(a.config).resolve()), sha256=cfg.sha256),
        backbone_run_args=dict(
            path=str(ra_path),
            sha256=(sha256_file(ra_path) if a.smoke else formal_ra_sha)),
        backbone_config=dict(path=str(backbone_config), sha256=frozen_cfg_sha),
        backbone_ckpt=dict(path=str(cp_path), sha256=cp_sha, step=cp.get("step")),
        target_latents=dict(path=str(out_dir / "target_latents.pt"),
                            tensor_sha256=target_sha,
                            file_sha256=sha256_file(out_dir / "target_latents.pt"),
                            shape=list(targets.shape), indexed_by="lut_id/row"),
        subset=dict(n=int(sub["n"]), sha256=sub["sha256"],
                    heldout_n=int(sub["expect_heldout_samples"]),
                    heldout_ids_sha256=sub["heldout_ids_sha256"]),
        readiness=readiness)
    (out_dir / "run_args.json").write_text(json.dumps(dict(
        epr="EPR-051/stage0/sprf/T-ALIGN-TIMEFILM", mode=mode, contract=contract,
        config=cfg.d, backbone_config=cd, data_law=law,
        n_train=len(train), n_heldout=len(held), a8=a8,
        assertions=dict(B1_backbone_frozen=True,
                        B3_heldout_isolated=True,
                        B4_injection_bitexact=dict(
                            ok=b4, max_abs=b4_max,
                            target_cache_runtime_max_abs=cache_max),
                        production_guard=production_guard,
                        delta_edit_roll_helper=roll_helper_assertion,
                        T0_paired_init=t0_paired,
                        T1_time_code=t1, T2_parameter_replacement=t2,
                        T4_zero_init_identity=t4),
        invocation=dict(smoke=bool(a.smoke), limit_samples=int(a.limit_samples),
                        stop_after=int(a.stop_after), device=dev,
                        out_dir=str(out_dir))), ensure_ascii=False, indent=1))

    hist, step, b2, t3, seen_train = [], 0, {}, {}, set()
    t_start = time.time()
    stop_at = step + a.stop_after if a.stop_after else None
    while step < max_steps:
        for b in dl:
            if step >= max_steps:
                break
            bad = set(b["sid"]) & heldout_ids
            if bad:
                die(f"B3 FAILED: held-out 进入 predictor 训练 {sorted(bad)[:3]}")
            batch_tokens = {(sid, int(dep)) for sid, dep in
                            zip(b["sid"], b["depth"].tolist())}
            dup = batch_tokens & seen_train
            if dup:
                die(f"one-pass FAILED: 重复训练样本 {sorted(dup)[:3]}")
            seen_train.update(batch_tokens)
            for g in opt.param_groups:
                g["lr"] = T0.lr_at(step, lr, warmup, max_steps, sched)
            c = T0.gather_feats(feats_cache, b["key"], 0, dev)
            f = b["feat"].to(dev, non_blocking=True)
            rows = b["rows"].to(dev, non_blocking=True)
            msk = b["mask"].to(dev, non_blocking=True)
            tgt = targets[rows]                       # (B,K,128)
            out = pred(c, f)
            w = (msk if active_only else torch.ones_like(msk)).unsqueeze(-1)
            l_raw = nn.functional.smooth_l1_loss(out, tgt, reduction="none",
                                                 beta=sl1b)
            loss = w_lat * (l_raw * w).sum() / (w.sum().clamp_min(1.0) * out.shape[-1])
            if not torch.isfinite(loss):
                die(f"step {step}: loss = {float(loss)}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if not b2:
                gl2 = math.sqrt(sum(float((q.grad ** 2).sum())
                                    for q in pred.parameters() if q.grad is not None))
                nnone = sum(1 for q in pred.parameters() if q.grad is None)
                b2 = dict(grad_l2=gl2, params_without_grad=nnone)
                if nnone or not gl2 > 0:
                    die(f"B2 FAILED: predictor 没拿到梯度 {b2}")
                print(f"[assert] B2 predictor 接线 OK grad_l2={gl2:.3e}", flush=True)
                tg = pred.time_affine.weight.grad
                t3 = dict(grad_l2=(0.0 if tg is None else float(tg.norm())),
                          finite=bool(tg is not None and torch.isfinite(tg).all()),
                          params_without_grad=int(tg is None))
                if (t3["params_without_grad"] or not t3["finite"] or
                        not t3["grad_l2"] > 0):
                    die(f"T3 FAILED: time_affine step1 gradient {t3}")
                print(f"[assert] T3 time_affine 接线 OK grad_l2="
                      f"{t3['grad_l2']:.3e}", flush=True)
            gn = float(nn.utils.clip_grad_norm_(pred.parameters(), clip))
            opt.step()
            step += 1
            if step % log_every == 0 or step == 1 or step == max_steps:
                with torch.no_grad():
                    cos = nn.functional.cosine_similarity(out, tgt, dim=-1)
                    cosm = float((cos * msk).sum() / msk.sum().clamp_min(1))
                    l2 = float(((out - tgt).norm(dim=-1) * msk).sum()
                               / msk.sum().clamp_min(1))
                rec = dict(step=step, loss=float(loss), latent_cos=cosm,
                           latent_l2=l2, lr=opt.param_groups[0]["lr"],
                           grad_norm=gn, seconds=round(time.time() - t_start, 1))
                hist.append(rec)
                T0.append_jsonl(steps_p, rec)
                print(f"  step {step}/{max_steps} loss={loss:.6f} "
                      f"cos={cosm:.4f} l2={l2:.4f} lr={rec['lr']:.2e} "
                      f"gn={gn:.3f}", flush=True)
            if stop_at and step >= stop_at:
                break
        if stop_at and step >= stop_at:
            break

    # 正式 one-pass 必须每条恰好一次；--stop-after smoke 是显式的机械截断。
    seen_assertion = dict(seen=len(seen_train), expected=len(train),
                          exact=(len(seen_train) == len(train)),
                          drop_last=False, one_pass_steps=one_pass)
    if not a.stop_after and not seen_assertion["exact"]:
        die(f"one-pass FAILED: seen {len(seen_train)} != train {len(train)}")

    model.cond.edit_enc = frozen_edit_enc
    after = module_sha256(model)
    if bk_fingerprint != after:
        die("B1 FAILED: 主干权重在训练中被改动")
    print("[assert] B1 主干逐位未变 OK", flush=True)
    model.cond.edit_enc = nn.Identity()
    model.edit_condition = "predicted_lut"
    model.edit_contract = "predicted_lut"
    predictor_payload = dict(step=step, predictor=pred.state_dict(),
                             config_sha256=cfg.sha256,
                             backbone_ckpt_sha256=cp_sha,
                             target_latents_sha256=target_sha,
                             contract=contract)
    if "time_code" in predictor_payload["predictor"]:
        die("T2 FAILED: nonpersistent time_code 进入checkpoint payload")
    predictor_path = out_dir / "predictor.pt"
    T0._atomic_save(predictor_payload, predictor_path)
    saved_predictor, _ = load_torch_hashed(predictor_path)
    t2["time_code_in_checkpoint"] = bool(
        "time_code" in saved_predictor.get("predictor", {}))
    if t2["time_code_in_checkpoint"]:
        die("T2 FAILED: time_code 进入落盘checkpoint")
    del saved_predictor
    heldout_result = final_eval(model, pred, held, blobs, feats_cache, cfg, cd,
                                dev, n_steps, grid, targets, inv, bool(a.smoke))
    hv = heldout_result["overall"]["model"]
    predicted_row = dict(
        name="T-ALIGN-TIMEFILM", contract="predicted_lut",
        n=int(heldout_result["n"]), step=step,
        linf8_p50=float(hv["p50"]), linf8_p95=float(hv["p95"]),
        linf8_p99=float(hv["p99"]))
    contract_rows = [predicted_row]
    if not a.smoke:
        expect = cfg.int_("compare", "expect_heldout")
        if expect != 4560:
            die(f"[compare] expect_heldout {expect} != 4560")
        contract_rows += readiness["references"]
    preregistered = {}
    if not a.smoke:
        matched_p50 = cfg.num("compare", "matched_linf8_p50")
        gate_p50 = cfg.num("compare", "gate_linf8_p50")
        delta_p50 = float(hv["p50"]) - matched_p50
        preregistered = dict(
            gate_linf8_p50=gate_p50,
            headline_linf8_p50=float(hv["p50"]),
            gate_pass=bool(float(hv["p50"]) <= gate_p50),
            matched_baseline_linf8_p50=matched_p50,
            delta_linf8_p50=delta_p50,
            direction_check_delta_lt_0=bool(delta_p50 < 0))
    metrics = dict(
        epr="EPR-051/stage0/sprf/T-ALIGN-TIMEFILM", generated=time.strftime(
            "%Y-%m-%dT%H:%M:%S%z"), smoke=bool(a.smoke), mode=mode,
        contract=contract, n_train=len(train), n_heldout=len(held), steps=step,
        predictor_params=pred.n_params(), feat_dim=feat_dim, hist_grid=grid,
        assertions=dict(B1_backbone_frozen=True, B1_state_sha256=after,
                        B2_predictor_grad=b2, B3_heldout_isolated=True,
                        B4_injection_bitexact=dict(
                            ok=b4, max_abs=b4_max,
                            target_cache_runtime_max_abs=cache_max),
                        X1_X5_subset=subset_report,
                        production_guard=production_guard,
                        delta_edit_roll_helper=roll_helper_assertion,
                        one_pass_seen=seen_assertion,
                        readiness=readiness,
                        T0_paired_init=t0_paired,
                        T1_time_code=t1, T2_parameter_replacement=t2,
                        T3_time_affine_grad=t3, T4_zero_init_identity=t4),
        a8=a8, preregistered=preregistered, contract_rows=contract_rows,
        heldout=heldout_result, train_history=hist,
        wall_seconds=round(time.time() - t_start, 1))
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=1))
    (out_dir / "train_meta.json").write_text(json.dumps(dict(
        epr="EPR-051/stage0/sprf/T-ALIGN-TIMEFILM", mode=mode, contract=contract,
        backbone=str(bk_run), backbone_step=cp.get("step"),
        n_train=len(train), steps=step, one_pass=one_pass,
        predictor_params=pred.n_params(), feat_dim=feat_dim, hist_grid=grid,
        assertions=dict(B1_backbone_frozen=True, B2_predictor_grad=b2,
                        B3_heldout_isolated=True,
                        B4_injection_bitexact=dict(
                            ok=b4, max_abs=b4_max,
                            target_cache_runtime_max_abs=cache_max),
                        production_guard=production_guard,
                        delta_edit_roll_helper=roll_helper_assertion,
                        one_pass_seen=seen_assertion,
                        readiness=readiness,
                        T0_paired_init=t0_paired,
                        T1_time_code=t1, T2_parameter_replacement=t2,
                        T3_time_affine_grad=t3, T4_zero_init_identity=t4),
        config=cfg.d, history=hist,
        wall_seconds=round(time.time() - t_start, 1)),
        ensure_ascii=False, indent=1))
    provenance = dict(
        epr="EPR-051/stage0/sprf/T-ALIGN-TIMEFILM", contract=contract, a8=a8,
        outputs={name: sha256_file(out_dir / name) for name in
                 ("predictor.pt", "target_latents.pt", "run_args.json",
                  "train_meta.json", "metrics.json", "subset_report.json")})
    (out_dir / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=1))
    print(f"[align] predictor -> {out_dir/'predictor.pt'}", flush=True)
    print(f"[align] metrics -> {out_dir/'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
