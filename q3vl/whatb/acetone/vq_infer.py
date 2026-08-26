"""Stage 2, row C (AceTone env): the LUT tokenizer round trip + ``A_axis``.

    /home/bc/data/external/acetone/venv/bin/python \\
        q3vl/whatb/acetone/vq_infer.py --inputs <inputs dir> --out <artifacts dir>

Reads ``inputs/luts32.npz`` (GT bank LUTs already put through AceTone's own
``resize_lut``), runs ``model/vq.py:VQVAE3DLUT`` with the in-repo weights
``model/acetone-vqvae-d64.pt``, and writes

    recon32.npz    lut_id -> (32,32,32,3) float32, the decoder's output
    tokens.json    lut_id -> the 64 codebook indices (4x4x4, row-major)
    vq_facts.json  checkpoint facts, codebook usage, A_axis

No metric is computed here.  This script is deliberately unable to import the
campaign package: it loads ``q3vl/whatb/lutdata.py`` by file for the ``A_axis``
measurement and nothing else.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


def _load_bridge():
    path = Path(__file__).resolve().parent / "bridge.py"
    spec = importlib.util.spec_from_file_location("_acetone_bridge", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", default="/home/bc/data/runs/what_b/acetone_inputs")
    ap.add_argument("--out", default="/home/bc/data/runs/what_b/whatb_ACETONE_tokenizer/artifacts")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--skip-axis", action="store_true")
    args = ap.parse_args(argv)

    bridge = _load_bridge()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    facts = {"repo": bridge.repo_facts()}
    if not args.skip_axis:
        facts["A_axis"] = bridge.axis_parity()
        print(json.dumps({"A_axis": facts["A_axis"]}, indent=2), flush=True)

    model, vq_facts = bridge.load_vq(device=args.device)
    facts["vq"] = vq_facts

    data = np.load(Path(args.inputs) / "luts32.npz")
    lut_ids = sorted(data.files)
    recon: dict[str, np.ndarray] = {}
    tokens: dict[str, list[int]] = {}
    usage: Counter = Counter()
    for start in range(0, len(lut_ids), args.batch):
        chunk = lut_ids[start:start + args.batch]
        grids = np.stack([np.asarray(data[k], dtype=np.float32) for k in chunk])
        idx, rec = bridge.vq_roundtrip(model, grids, device=args.device)
        for j, k in enumerate(chunk):
            recon[k] = rec[j]
            flat = idx[j].reshape(-1).tolist()
            tokens[k] = [int(v) for v in flat]
            usage.update(flat)
        print(f"[vq] {start + len(chunk)}/{len(lut_ids)}", flush=True)

    np.savez_compressed(out / "recon32.npz", **recon)
    (out / "tokens.json").write_text(json.dumps(tokens), encoding="utf-8")

    facts["tokenizer_run"] = {
        "n_luts": len(lut_ids), "tokens_per_lut": 64,
        "n_codes_used": len(usage), "codebook_size": vq_facts["codebook_size"],
        "code_usage_top10": usage.most_common(10),
        "recon_min": float(min(float(v.min()) for v in recon.values())),
        "recon_max": float(max(float(v.max()) for v in recon.values())),
        "n_nonfinite": int(sum(int((~np.isfinite(v)).sum()) for v in recon.values())),
    }
    (out / "vq_facts.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")
    print(json.dumps(facts["tokenizer_run"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
