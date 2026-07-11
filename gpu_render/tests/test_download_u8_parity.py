"""对拍 ``render_batch._download_u8``：GPU 端 u8 量化 + permute 的新路径必须与
旧 float 路径（量化后逐图 CPU transpose）逐位一致。CPU 张量即可验证——两条路
用同一套 torch 标量算子，量化语义 trunc(clamp(x,0,1)*255+0.5) 与设备无关。"""
from __future__ import annotations

import numpy as np
import torch

from gpu_render.gpu.render_batch import _download_u8


def _legacy_float_path(out_bchw: torch.Tensor) -> list:
    # 改造前实现（原样保留作 golden）：量化后 .cpu()，CPU 侧逐图 transpose+copy。
    u8 = out_bchw.clamp(0, 1).mul(255.0).add_(0.5).to(torch.uint8).cpu().numpy()
    return [np.ascontiguousarray(u8[b].transpose(1, 2, 0))
            for b in range(u8.shape[0])]


def test_download_u8_matches_legacy_path_bitwise():
    torch.manual_seed(0)
    x = torch.rand(3, 3, 17, 23, dtype=torch.float32)
    # 越界值走 clamp、精确 .5 边界走取整语义（add_(0.5) 截断 ≠ round 半偶）
    x[0, 0, 0, :6] = torch.tensor(
        [-0.25, 0.0, 1.0, 1.25, 126.5 / 255.0, 127.5 / 255.0])

    got = _download_u8(x)
    want = _legacy_float_path(x)

    assert len(got) == len(want) == 3
    for g, w in zip(got, want):
        assert g.dtype == np.uint8
        np.testing.assert_array_equal(g, w)


def test_download_u8_returns_contiguous_hwc_uint8():
    x = torch.linspace(0, 1, steps=2 * 3 * 4 * 5,
                       dtype=torch.float32).reshape(2, 3, 4, 5)
    outs = _download_u8(x)
    assert [o.shape for o in outs] == [(4, 5, 3), (4, 5, 3)]
    assert all(o.flags["C_CONTIGUOUS"] for o in outs)
    assert all(o.dtype == np.uint8 for o in outs)
