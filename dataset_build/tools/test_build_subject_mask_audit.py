from __future__ import annotations

from collections import Counter

from PIL import Image

from dataset_build.tools import build_subject_mask_audit as audit


def _row(subject: str, *, scene: str = "street", area: float = 0.2) -> dict:
    return {"main_subject": subject, "scene": scene, "mask_area": area}


def test_stratum_contract_covers_the_review_risk_buckets():
    assert audit._stratum(_row("bride and groom")) == "multi_instance"
    assert audit._stratum(_row("ocean waves")) == "landscape_or_no_subject"
    assert audit._stratum(_row("woman", area=0.04)) == "small_subject"
    assert audit._stratum(_row("woman", area=0.6)) == "large_subject"
    assert audit._stratum(_row("woman")) == "portrait"
    assert audit._stratum(_row("tabby cat")) == "animal"
    assert audit._stratum(_row("plate of food")) == "product_or_food"
    assert audit._stratum(_row("stone archway")) == "general"
    assert audit._stratum(_row("skyscrapers")) == "general"


def test_sampling_is_balanced_and_deterministic(tmp_path):
    rows = []
    examples = [
        _row("couple"),
        _row("mountain"),
        _row("woman", area=0.04),
        _row("woman", area=0.6),
        _row("woman"),
        _row("dog"),
        _row("cake"),
        _row("stone archway"),
    ]
    for group, example in enumerate(examples):
        for index in range(3):
            source = tmp_path / f"source_{group}_{index}.jpg"
            mask = tmp_path / f"mask_{group}_{index}.png"
            source.touch()
            mask.touch()
            rows.append({
                **example,
                "asset_id": f"asset-{group}-{index}",
                "path": str(source),
                "mask_path": str(mask),
            })

    first = audit._sample(rows, 16, seed=7)
    second = audit._sample(rows, 16, seed=7)

    assert [row["asset_id"] for row in first] == [
        row["asset_id"] for row in second]
    assert Counter(row["stratum"] for row in first) == {
        "multi_instance": 2,
        "landscape_or_no_subject": 2,
        "small_subject": 2,
        "large_subject": 2,
        "portrait": 2,
        "animal": 2,
        "product_or_food": 2,
        "general": 2,
    }


def test_tile_renders_source_mask_overlay(tmp_path):
    source = tmp_path / "source.png"
    mask = tmp_path / "mask.png"
    Image.new("RGB", (32, 24), (80, 100, 120)).save(source)
    Image.new("L", (32, 24), 0).save(mask)
    row = {
        "asset_id": "asset",
        "path": str(source),
        "mask_path": str(mask),
        "main_subject": "subject",
        "scene": "portrait",
        "mask_area": 0.2,
        "megapixels": 0.001,
        "stratum": "portrait",
        "audit_index": 1,
    }

    tile = audit._tile(row, 180, 140)

    assert tile.size == (180, 140)
