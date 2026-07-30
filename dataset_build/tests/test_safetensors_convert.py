"""`tools/convert_onealign_safetensors.py` 的合成权重回归测试。

不碰真实 OneAlign 权重（15 GiB）：造两个小 shard + index，覆盖三个真实坑——
共享 storage 的一对张量、非连续张量、多种 dtype 的逐位保真。
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from dataset_build.tools.convert_onealign_safetensors import (
    ConvertError,
    bitwise_equal,
    convert,
    main,
    safetensors_shard_name,
)


def _make_fixture(root: Path) -> dict[str, dict[str, torch.Tensor]]:
    """写出两个分片 .bin + index，返回逐 shard 的期望张量（值层面）。"""
    torch.manual_seed(0)
    base = torch.randn(4, 6, dtype=torch.float32)
    shared = torch.randn(3, 5, dtype=torch.float16)
    pool = torch.randn(4, 3, dtype=torch.float32)
    shard_one = {
        # 同一张量两个 key（tie 权重的典型形态）：safetensors 直接拒收，必须克隆。
        "model.embed_tokens.weight": shared,
        "lm_head.weight": shared,
        # 非连续视图（转置），必须 .contiguous()。
        "model.layers.0.attn.q_proj.weight": base.t(),
        # 与上一条同 storage，但上一条已被拷贝解开，这里无须再克隆。
        "model.layers.0.attn.k_proj.weight": base,
    }
    shard_two = {
        "model.norm.weight": torch.randn(7, dtype=torch.bfloat16),
        "model.visual.vit_eos": torch.randn(1, 1, 3, dtype=torch.float32),
        "model.visual.counter": torch.arange(5, dtype=torch.int64),
        "model.visual.flag": torch.tensor([True, False, True]),
        "model.visual.empty": torch.zeros(0, dtype=torch.float32),
        # 同 storage 不同 offset 的两个连续切片：safetensors 按字节区间判重叠会
        # 放行，本工具按 storage 指纹保守克隆（多一份字节，值逐位不变）。
        "model.visual.slice_a": pool[0:2],
        "model.visual.slice_b": pool[2:4],
    }
    torch.save(shard_one, root / "pytorch_model-00001-of-00002.bin")
    torch.save(shard_two, root / "pytorch_model-00002-of-00002.bin")
    weight_map = {key: "pytorch_model-00001-of-00002.bin" for key in shard_one}
    weight_map.update({key: "pytorch_model-00002-of-00002.bin" for key in shard_two})
    total = sum(
        tensor.numel() * tensor.element_size()
        for shard in (shard_one, shard_two)
        for tensor in shard.values()
    )
    (root / "pytorch_model.bin.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}),
        encoding="utf-8",
    )
    return {
        "pytorch_model-00001-of-00002.bin": shard_one,
        "pytorch_model-00002-of-00002.bin": shard_two,
    }


class ConvertTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.expected = _make_fixture(self.root)
        self.addCleanup(self._tmp.cleanup)

    def _flat_expected(self) -> dict[str, torch.Tensor]:
        return {
            key: tensor
            for shard in self.expected.values()
            for key, tensor in shard.items()
        }

    def test_fixture_would_break_naive_save(self) -> None:
        """夹具确实踩到 safetensors 的红线，否则这组测试没有意义。"""
        shard = torch.load(
            self.root / "pytorch_model-00001-of-00002.bin",
            map_location="cpu",
            weights_only=True,
        )
        pair = {
            key: shard[key]
            for key in ("model.embed_tokens.weight", "lm_head.weight")
        }
        with self.assertRaisesRegex(RuntimeError, "share memory"):
            save_file(pair, str(self.root / "naive_alias.safetensors"))
        with self.assertRaises((RuntimeError, ValueError)):
            save_file(shard, str(self.root / "naive.safetensors"))

    def test_convert_is_bitwise_identical(self) -> None:
        summary = convert(str(self.root))
        self.assertTrue(summary["verified"])
        self.assertEqual(summary["keys"], len(self._flat_expected()))

        index = json.loads((self.root / "model.safetensors.index.json").read_text())
        self.assertEqual(set(index["weight_map"]), set(self._flat_expected()))
        self.assertEqual(
            index["metadata"]["total_size"],
            json.loads((self.root / "pytorch_model.bin.index.json").read_text())[
                "metadata"
            ]["total_size"],
        )
        # 分片编号 1:1 保留，key→shard 的归属不变。
        self.assertEqual(
            sorted(set(index["weight_map"].values())),
            ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"],
        )
        for bin_name, shard in self.expected.items():
            safe_name = safetensors_shard_name(bin_name, 1, 2)
            for key in shard:
                self.assertEqual(index["weight_map"][key], safe_name)

        for bin_name, shard in self.expected.items():
            path = self.root / safetensors_shard_name(bin_name, 1, 2)
            with safe_open(path, framework="pt", device="cpu") as handle:
                self.assertEqual(handle.metadata().get("format"), "pt")
                self.assertEqual(set(handle.keys()), set(shard))
                for key, want in shard.items():
                    got = handle.get_tensor(key)
                    self.assertEqual(got.dtype, want.dtype, key)
                    self.assertEqual(tuple(got.shape), tuple(want.shape), key)
                    self.assertTrue(bitwise_equal(want, got), key)

    def test_aliases_are_cloned_and_reported(self) -> None:
        summary = convert(str(self.root))
        # 每组共享 storage 只保留首个原张量，其余克隆成独立副本；
        # k_proj 与 q_proj 同 storage，但 q_proj 因非连续已被拷贝解开，不再计入。
        self.assertEqual(
            sorted(summary["aliased_cloned"]),
            ["lm_head.weight", "model.visual.slice_b"],
        )
        self.assertEqual(
            summary["made_contiguous"], ["model.layers.0.attn.q_proj.weight"]
        )
        shard_entry = summary["shards"][0]
        self.assertEqual(shard_entry["source"], "pytorch_model-00001-of-00002.bin")
        self.assertEqual(shard_entry["output"], "model-00001-of-00002.safetensors")
        self.assertEqual(shard_entry["aliased_cloned"], ["lm_head.weight"])

        # 读回后两个 key 各自独立：改一个不影响另一个。
        path = self.root / "model-00001-of-00002.safetensors"
        with safe_open(path, framework="pt", device="cpu") as handle:
            embed = handle.get_tensor("model.embed_tokens.weight").clone()
            head = handle.get_tensor("lm_head.weight")
        self.assertTrue(bitwise_equal(embed, head))
        self.assertNotEqual(embed.data_ptr(), head.data_ptr())

    def test_originals_are_untouched(self) -> None:
        before = {
            path.name: path.stat().st_mtime_ns
            for path in self.root.glob("pytorch_model*")
        }
        convert(str(self.root))
        after = {
            path.name: path.stat().st_mtime_ns
            for path in self.root.glob("pytorch_model*")
        }
        self.assertEqual(before, after)
        self.assertEqual(len(before), 3)
        # 没有 .tmp 残留。
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_dry_run_writes_nothing(self) -> None:
        summary = convert(str(self.root), dry_run=True)
        self.assertTrue(summary["dry_run"])
        self.assertFalse(summary["verified"])
        self.assertEqual(
            [entry["output"] for entry in summary["shards"]],
            ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"],
        )
        self.assertEqual(summary["shards"][0]["tensors"], 4)
        self.assertEqual(list(self.root.glob("*.safetensors*")), [])

    def test_missing_index_fails_loudly(self) -> None:
        (self.root / "pytorch_model.bin.index.json").unlink()
        with self.assertRaises(ConvertError):
            convert(str(self.root))

    def test_index_shard_mismatch_fails_loudly(self) -> None:
        path = self.root / "pytorch_model.bin.index.json"
        index = json.loads(path.read_text())
        index["weight_map"]["model.absent"] = "pytorch_model-00002-of-00002.bin"
        path.write_text(json.dumps(index), encoding="utf-8")
        with self.assertRaises(ConvertError):
            convert(str(self.root))

    def test_cli_dry_run_returns_zero(self) -> None:
        self.assertEqual(main(["--model-dir", str(self.root), "--dry-run"]), 0)
        self.assertEqual(main(["--model-dir", str(self.root)]), 0)
        self.assertTrue((self.root / "model.safetensors.index.json").is_file())

    def test_bitwise_equal_catches_nan_and_dtype(self) -> None:
        nan = torch.tensor([float("nan"), 1.0])
        self.assertTrue(bitwise_equal(nan, nan.clone()))
        self.assertFalse(torch.equal(nan, nan.clone()))
        self.assertFalse(
            bitwise_equal(torch.zeros(2, dtype=torch.float32), torch.zeros(2, dtype=torch.float16))
        )
        self.assertFalse(bitwise_equal(torch.zeros(2), torch.zeros(3)))


if __name__ == "__main__":
    unittest.main()
