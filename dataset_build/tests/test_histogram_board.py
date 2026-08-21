"""E4: the four-panel histogram board that the diagnosis stage receives as image 2."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image

from dataset_build.agent_loop.histogram_board import (
    BOARD_BIN_GEOMETRY, BOARD_REVISION, BOARD_SIZE, C_BINS, HUE_SECTORS, L_BINS,
    RGB_BINS, board_png, panel_stats, render_board,
)


def _write_image(path: Path, seed: int = 3, size: tuple[int, int] = (301, 199)) -> Path:
    rng = np.random.default_rng(seed)
    pixels = (rng.random((size[1], size[0], 3)) * 255).astype(np.uint8)
    Image.fromarray(pixels, "RGB").save(path)
    return path


def test_board_png_is_byte_identical_across_two_real_renders(tmp_path: Path) -> None:
    """Determinism is the whole contract: same file in, same PNG bytes out."""
    source = _write_image(tmp_path / "source.png")
    first = board_png(source)
    second = board_png(source)
    assert first == second
    with Image.open(io.BytesIO(first)) as board:
        assert board.size == BOARD_SIZE
        assert board.format == "PNG"


def test_two_different_images_render_different_boards(tmp_path: Path) -> None:
    one = board_png(_write_image(tmp_path / "one.png", seed=1))
    two = board_png(_write_image(tmp_path / "two.png", seed=2))
    assert one != two


def test_panel_stats_reads_the_original_resolution_and_normalised_shares(
    tmp_path: Path,
) -> None:
    source = _write_image(tmp_path / "source.png", size=(640, 480))
    stats = panel_stats(source)
    assert stats["size"] == (640, 480)
    assert stats["pixels"] == 640 * 480
    assert len(stats["l_hist"]) == L_BINS
    assert len(stats["c_hist"]) == C_BINS
    assert len(stats["hue_shares"]) == HUE_SECTORS
    assert [len(channel) for channel in stats["rgb_hist"]] == [RGB_BINS] * 3
    assert sum(stats["l_hist"]) == 1.0
    assert sum(stats["c_hist"]) == 1.0
    assert all(sum(channel) == 1.0 for channel in stats["rgb_hist"])
    assert 0.0 <= stats["chromatic_share"] <= 1.0
    assert stats["l_p1"] <= stats["l_p50"] <= stats["l_p99"]
    assert stats["c_p50"] <= stats["c_p95"]


def test_panel_stats_are_the_only_input_of_the_pixels(tmp_path: Path) -> None:
    """`render_board` is a pure function of the statistics dict."""
    stats = panel_stats(_write_image(tmp_path / "source.png"))
    first = render_board(stats)
    second = render_board(dict(stats))
    assert first.size == BOARD_SIZE
    assert np.array_equal(np.asarray(first), np.asarray(second))


def test_grey_image_has_no_chroma_and_a_single_lightness_bin(tmp_path: Path) -> None:
    target = tmp_path / "grey.png"
    Image.new("RGB", (64, 48), (128, 128, 128)).save(target)
    stats = panel_stats(target)
    assert stats["chromatic_share"] == 0.0
    assert stats["hue_shares"] == [0.0] * HUE_SECTORS
    assert stats["c_p50"] == 0.0
    assert max(stats["l_hist"]) == 1.0
    assert stats["clip_low"] == 0.0 and stats["clip_high"] == 0.0
    # The board still renders (empty hue panel is drawn, not skipped).
    assert render_board(stats).size == BOARD_SIZE


def test_board_revision_and_bin_geometry_are_declared() -> None:
    assert BOARD_REVISION
    assert BOARD_SIZE == (640, 604)
    assert BOARD_BIN_GEOMETRY == {
        "l_bins": L_BINS, "rgb_bins": RGB_BINS, "c_bins": C_BINS, "c_max": 120.0,
        "hue_sectors": HUE_SECTORS, "hue_chroma_min": 10.0,
        "clip_low_l": 2.0, "clip_high_l": 98.0,
    }
