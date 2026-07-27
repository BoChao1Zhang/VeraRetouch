from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dataset_build.tools.dataset_plan import (
    MAX_KEY_CHARS,
    PlanError,
    PlanGroup,
    PlanRow,
    META_SUFFIX,
    normalise_extension,
    plan_presets,
    sanitize_key,
    write_group,
)
from dataset_build.tools.indexed_tar import (
    IndexedTarDataset,
    build_indexed_tar,
    verify_dataset,
)


class KeyPolicyTests(unittest.TestCase):
    def test_keys_stay_legal_stable_and_distinct(self) -> None:
        self.assertEqual(sanitize_key("ppr10k_000001"), "ppr10k_000001")
        # A dot would start the extension, so it can never survive in a key.
        self.assertNotIn(".", sanitize_key("007R8.JewpwU"))
        self.assertEqual(sanitize_key("007R8.JewpwU"), sanitize_key("007R8.JewpwU"))
        self.assertNotEqual(sanitize_key("a.b"), sanitize_key("a_b"))
        self.assertNotEqual(sanitize_key("中文"), sanitize_key("日本語"))
        self.assertTrue(sanitize_key("中文").isascii())
        long_key = sanitize_key("x" * 400)
        self.assertLessEqual(len(long_key), MAX_KEY_CHARS)
        self.assertNotEqual(long_key, sanitize_key("x" * 401))

    def test_extensions_are_lowercased_and_bounded(self) -> None:
        self.assertEqual(normalise_extension("a.JPG"), ".jpg")
        self.assertEqual(normalise_extension("a.raw_mask.PNG"), ".raw_mask.png")
        self.assertEqual(normalise_extension("a.extension-way-too-long.png"), ".extension-way-too-long.png")
        with self.assertRaisesRegex(PlanError, "needs an extension"):
            normalise_extension("no-extension")


class GroupWritingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_member_collision_is_a_planning_error(self) -> None:
        first, second = self.root / "a.png", self.root / "b.png"
        first.write_bytes(b"1")
        second.write_bytes(b"2")
        group = PlanGroup(group="img/portrait/ppr10k")
        group.add(PlanRow(path=first, logical_path="img/portrait/ppr10k/x.png"))
        group.add(PlanRow(path=second, logical_path="img/portrait/ppr10k/x.png"))
        with self.assertRaisesRegex(PlanError, "member collision"):
            write_group(group, self.root / "out", meta_staging=self.root / "staging")

    def test_plan_round_trips_through_the_packer(self) -> None:
        image = self.root / "src" / "ppr10k_000001.png"
        image.parent.mkdir()
        image.write_bytes(b"pixels")
        group = PlanGroup(group="img/portrait/ppr10k")
        group.add(
            PlanRow(
                path=image,
                logical_path="img/portrait/ppr10k/ppr10k_000001.source.png",
                meta={"role": "source"},
            )
        )
        summary = write_group(
            group,
            self.root / "plan",
            meta_staging=self.root / "staging",
            sample_meta={"ppr10k_000001": {"scene": "portrait", "corpus": "ppr10k"}},
        )
        self.assertEqual(summary["members"], 2)
        self.assertEqual(summary["samples"], 1)

        output = self.root / "packed"
        manifest = build_indexed_tar(
            None, output, plan=Path(summary["plan"]), shard_size_bytes=1024**2
        )
        self.assertEqual(manifest["sample_count"], 1)
        self.assertEqual(verify_dataset(output)["members"], 2)
        with IndexedTarDataset(output) as dataset:
            sample = dataset.read_sample("ppr10k_000001")
            self.assertEqual(sample[".source.png"], b"pixels")
            self.assertEqual(json.loads(sample[".vrmeta.json"])["scene"], "portrait")


class PresetPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.bank = self.root / "bank"
        self.bank.mkdir()
        self.presets = self.root / "recipes"
        self.presets.mkdir()
        self.baked = self.root / "baked"
        self.baked.mkdir()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _write_bank(self) -> None:
        rows = [
            {"preset_id": "rcp_aaa", "fmt": "xmp", "kind": "param", "pack_id": "E18",
             "path": str(self.presets / "quandian_000001.xmp")},
            {"preset_id": "rcp_bbb", "fmt": "cube", "kind": "lut", "pack_id": "H022",
             "path": str(self.presets / "quandian_000002.cube")},
            {"preset_id": "rcp_ccc", "fmt": "cube", "kind": "lut", "pack_id": "H022",
             "path": str(self.presets / "missing.cube")},
        ]
        for row in rows:
            if "missing" not in row["path"]:
                Path(row["path"]).write_bytes(row["preset_id"].encode())
        (self.bank / "features.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        # rcp_bbb is deliberately absent: the weak-effect filter drops it from the
        # taxonomy, so it must still be archived under _unclassified.
        (self.bank / "taxonomy.jsonl").write_text(
            json.dumps({"preset_id": "rcp_aaa", "major": "复古胶片", "minor": "黄绿胶片_03"}) + "\n",
            encoding="utf-8",
        )
        (self.baked / "rcp_aaa.cube").write_bytes(b"baked-lut")

    def test_groups_split_by_format_and_style_class(self) -> None:
        self._write_bank()
        summary = plan_presets(
            self.bank,
            self.root / "plans",
            meta_staging=self.root / "staging",
            baked_dir=self.baked,
        )
        groups = {str(item["group"]): item for item in summary}
        self.assertEqual(
            sorted(groups),
            ["preset/cube/_unclassified", "preset/xmp/复古胶片/黄绿胶片_03"],
        )
        # xmp sample: the preset itself, its baked LUT, and the metadata member.
        self.assertEqual(groups["preset/xmp/复古胶片/黄绿胶片_03"]["members"], 3)
        self.assertEqual(groups["preset/xmp/复古胶片/黄绿胶片_03"]["samples"], 1)
        # The missing file is skipped rather than faking an archived member.
        self.assertEqual(groups["preset/cube/_unclassified"]["members"], 2)

        plan = Path(groups["preset/xmp/复古胶片/黄绿胶片_03"]["plan"])
        logical = [json.loads(line)["logical_path"] for line in plan.read_text().splitlines()]
        self.assertEqual(
            logical,
            [
                "preset/xmp/复古胶片/黄绿胶片_03/rcp_aaa.baked.cube",
                "preset/xmp/复古胶片/黄绿胶片_03/rcp_aaa.vrmeta.json",
                "preset/xmp/复古胶片/黄绿胶片_03/rcp_aaa.xmp",
            ],
        )

        output = self.root / "packed"
        manifest = build_indexed_tar(None, output, plan=plan, shard_size_bytes=1024**2)
        self.assertEqual(manifest["sample_count"], 1)
        self.assertEqual(verify_dataset(output)["members"], 3)
        with IndexedTarDataset(output) as dataset:
            sample = dataset.read_sample("rcp_aaa")
            self.assertEqual(sample[".xmp"], b"rcp_aaa")
            self.assertEqual(sample[".baked.cube"], b"baked-lut")
            meta = json.loads(sample[".vrmeta.json"])
            self.assertEqual(meta["major"], "复古胶片")
            self.assertEqual(meta["pack_id"], "E18")


if __name__ == "__main__":
    unittest.main()
