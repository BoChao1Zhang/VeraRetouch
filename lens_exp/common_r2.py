# lens-exp round 2 common: R2 result paths, full-800 manifest, split, target matrix.
import os, sys, json, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from common import LR_ROOT, BOX_RE, TOKENS, TARGET_NAMES, compute_targets  # noqa: F401

RESULTS_R2 = os.path.expanduser("~/VeraRetouch/lens_exp_results_r2")
DUMPS_R2 = os.path.join(RESULTS_R2, "dumps")
C0_DIR = os.path.join(DUMPS_R2, "c0")

# round-1 known failures: no retouch token within 1200 new tokens
KNOWN_FAILS = {"CN_54", "CN_85"}

# 26-dim photometric-delta vector order (light 6 + colortemp 4 + colormixer 16)
DIM_NAMES = TARGET_NAMES["light"] + TARGET_NAMES["colortemp"] + TARGET_NAMES["colormixer"]


def build_manifest_r2():
    rows = []
    for lang in ["CN", "EN"]:
        for sid in sorted(os.listdir(os.path.join(LR_ROOT, lang)), key=int):
            d = os.path.join(LR_ROOT, lang, sid)
            try:
                txt = open(os.path.join(d, "user_want.txt"), encoding="utf-8").read().strip()
            except Exception:
                continue
            if not (os.path.exists(os.path.join(d, "input.jpg")) and os.path.exists(os.path.join(d, "gt.jpg"))):
                continue
            key = f"{lang}_{sid}"
            if key in KNOWN_FAILS:
                continue
            rows.append(dict(
                key=key, lang=lang, sid=sid,
                input_path=os.path.join(d, "input.jpg"),
                gt_path=os.path.join(d, "gt.jpg"),
                prompt=txt, has_box=int(bool(BOX_RE.search(txt))),
            ))
    rows.sort(key=lambda r: r["key"])
    return rows


def load_manifest_r2():
    path = os.path.join(RESULTS_R2, "manifest_r2.json")
    if not os.path.exists(path):
        os.makedirs(RESULTS_R2, exist_ok=True)
        rows = build_manifest_r2()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=1)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_split_r2(seed=0, test_frac=0.2):
    """8:2 train/test stratified by language over captured C0 samples."""
    path = os.path.join(RESULTS_R2, "split_r2.json")
    if os.path.exists(path):
        return json.load(open(path))
    rows = [r for r in load_manifest_r2()
            if os.path.exists(os.path.join(C0_DIR, r["key"] + ".npz"))]
    rng = random.Random(seed)
    test = []
    for lang in ("CN", "EN"):
        ks = sorted(r["key"] for r in rows if r["lang"] == lang)
        rng.shuffle(ks)
        test += ks[: int(round(len(ks) * test_frac))]
    split = dict(test=sorted(test),
                 train=sorted(r["key"] for r in rows if r["key"] not in set(test)))
    json.dump(split, open(path, "w"), indent=1)
    return split


def load_targets_r2():
    """26-dim photometric delta per sample, cached; returns dict key -> np.float32[26]."""
    path = os.path.join(RESULTS_R2, "targets_r2.json")
    if os.path.exists(path):
        return {k: np.array(v, dtype=np.float32) for k, v in json.load(open(path)).items()}
    out = {}
    for r in load_manifest_r2():
        t = compute_targets(r["input_path"], r["gt_path"])
        out[r["key"]] = np.concatenate([t["light"], t["colortemp"], t["colormixer"]]).astype(np.float32)
    json.dump({k: v.tolist() for k, v in out.items()}, open(path, "w"))
    return out
