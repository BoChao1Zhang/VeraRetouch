"""32x256 / 64x128 bit-identity check for the five EPR-025..029 arms.

Runs each arm's CPU smoke path twice at the same seed:

  A = the code as it is now (the shared EPR-030 caliber);
  B = the same code with the pre-2026-08-16 expressions restored by
      monkeypatch -- the ungated ``cfg.lambda_hc`` / ``cfg.lambda_sparse`` the
      loss used to be called with, and the literal ``text.split("x")`` batch
      parse.

Then compares ``steps.jsonl`` row by row.  Wall-clock columns
(``wall_ms_per_step`` / ``wall_s`` / ``wall_time_s``) are excluded from the
comparison and reported separately: they are timers, not results.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import shutil
import sys
from pathlib import Path

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/bitid")
TIMERS = {"wall_ms_per_step", "wall_s", "wall_time_s", "t_step_s"}


def rows_of(p: Path) -> list[dict]:
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]


def numeric_sha(p: Path) -> str:
    payload = [
        {k: v for k, v in sorted(r.items()) if k not in TIMERS} for r in rows_of(p)]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest()


def old_parse(text: str) -> tuple[int, int]:
    b, q = text.lower().split("x")          # the pre-change literal parse
    return int(b), int(q)


@contextlib.contextmanager
def pre_change(mod, arm_mod=None):
    """Restore the pre-2026-08-16 expressions on one runner / arm module."""
    from q3vl.whatb import caliber as K

    saved = []
    for m in (x for x in (mod, arm_mod) if x is not None):
        if hasattr(m, "K"):
            saved.append((m.K, "effective_lambda_hc", m.K.effective_lambda_hc))
            saved.append((m.K, "effective_lambda_sparse", m.K.effective_lambda_sparse))
            saved.append((m.K, "parse_batch_split", m.K.parse_batch_split))
            m.K.effective_lambda_hc = lambda v, level: v
            m.K.effective_lambda_sparse = lambda v, level: v
            m.K.parse_batch_split = old_parse
        for name in ("effective_lambda_hc", "effective_lambda_sparse"):
            if hasattr(m, name):
                saved.append((m, name, getattr(m, name)))
                setattr(m, name, lambda v, level: v)
    try:
        yield
    finally:
        for obj, name, val in reversed(saved):
            setattr(obj, name, val)
        K.effective_lambda_hc = K.effective_lambda_hc
        K.parse_batch_split = K.parse_batch_split


def _call(fn, *a, **kw):
    try:
        fn(*a, **kw)
    except SystemExit:
        pass                      # the degeneracy gate on a 3-step run
    except Exception as exc:      # the publication gate (field_pred needs the
        print(f"  [gate] {type(exc).__name__}", flush=True)   # where product)


# --------------------------------------------------------------------------- #
def run_qdual(d: Path, split: str) -> Path:
    from q3vl.whatb.scripts import run_qdual_arm as R
    _call(R.main, ["--smoke", "--smoke-steps", "4", "--z-source", "synthetic",
                   "--skip-colorspan-assert", "--max-train-samples", "64",
                   "--eval-max-samples", "2", "--lib-sample", "4",
                   "--baseline-repeats", "2", "--quick-eval-n", "2",
                   "--batch-split", split, "--seed", "20260810",
                   "--out-root", str(d), "--run-name", "r"])
    return d / "r" / "steps.jsonl"


def run_interpc(d: Path, split: str) -> Path:
    from q3vl.whatb.scripts import run_interpc_arm as R
    _call(R.main, ["--stage", "self-test", "--out-root", str(d),
                   "--interp-weight", "1", "--run-name", "r",
                   "--batch-split", split, "--seed", "20260810"])
    return d / "runs" / "r" / "steps.jsonl"


def run_affonly(d: Path, split: str) -> Path:
    from q3vl.whatb.scripts import run_affonly_arm as R
    _call(R.main, ["--run-dir", str(d), "--smoke", "--z-source", "synthetic",
                   "--skip-colorspan-assert", "--device", "cpu",
                   "--batch-split", split, "--total-steps", "3", "--eval-n", "2",
                   "--quick-n", "2", "--lib-rows", "4", "--repeats", "2",
                   "--eval-every", "0", "--no-bucket-pools", "--train-n", "128",
                   "--seed", "20260810"])
    return d / "steps.jsonl"


def run_idgate(d: Path, split: str) -> Path:
    from q3vl.whatb.scripts import run_idgate_arm as R
    _call(R.main, ["--run-dir", str(d), "--smoke", "--batch-split", split,
                   "--max-steps", "3", "--eval-every", "100000",
                   "--quick-eval-n", "4", "--library-size", "4",
                   "--n-repeats", "2", "--interp-pairs", "2", "--strength-n", "2",
                   "--seed", "20260810"])
    return d / "steps.jsonl"


def run_g4d(d: Path, split: str) -> Path:
    from q3vl.whatb.scripts import run_g4d_arm as R
    _call(R.main, ["--out", str(d), "--smoke", "--device", "cpu",
                   "--batch-split", split, "--steps", "3", "--eval-limit", "2",
                   "--quick-eval-n", "2", "--lib-sample", "4",
                   "--quick-eval-every", "0", "--seed", "20260810"])
    return d / "steps.jsonl"


ARMS = {
    "affonly": (run_affonly, "q3vl.whatb.scripts.run_affonly_arm",
                "q3vl.whatb.arms.affonly"),
    "interpc": (run_interpc, "q3vl.whatb.scripts.run_interpc_arm",
                "q3vl.whatb.arms.interpc"),
    "idgate": (run_idgate, "q3vl.whatb.scripts.run_idgate_arm",
               "q3vl.whatb.arms.idgate"),
    "g4d": (run_g4d, "q3vl.whatb.scripts.run_g4d_arm", "q3vl.whatb.arms.g4d"),
    "qdual": (run_qdual, "q3vl.whatb.scripts.run_qdual_arm",
              "q3vl.whatb.arms.qdual"),
}


def main() -> int:
    import importlib

    report: dict[str, dict] = {}
    only = sys.argv[2].split(",") if len(sys.argv) > 2 else list(ARMS)
    for name in only:
        fn, runner_mod, arm_mod = ARMS[name]
        R = importlib.import_module(runner_mod)
        A = importlib.import_module(arm_mod)
        for split in ("32x256", "64x128"):
            out: dict = {}
            for tag in ("new", "old"):
                d = OUT / f"{name}_{split}_{tag}"
                shutil.rmtree(d, ignore_errors=True)
                d.mkdir(parents=True)
                if tag == "old":
                    with pre_change(R, A):
                        path = fn(d, split)
                else:
                    path = fn(d, split)
                out[tag] = numeric_sha(path)
                out[f"{tag}_n_rows"] = len(rows_of(path))
                out[f"{tag}_row0"] = {
                    k: v for k, v in rows_of(path)[0].items() if k not in TIMERS}
            out["identical_excluding_timers"] = out["new"] == out["old"]
            report[f"{name}/{split}"] = out
            print(f"{name}/{split}: identical={out['identical_excluding_timers']} "
                  f"sha={out['new']}", flush=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: {"identical_excluding_timers":
                          v["identical_excluding_timers"],
                          "sha256_numeric": v["new"], "n_rows": v["new_n_rows"]}
                      for k, v in report.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
