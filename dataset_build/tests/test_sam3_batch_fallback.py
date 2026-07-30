"""SAM3 batch forward: a failed batch must give its VRAM back before retrying.

The per-image fallback re-runs the same images one at a time, on the same card
that now also holds two OneAlign copies, so anything the failed batch still
pins is subtracted from the retry's budget — which is how an OOM turns into a
run of OOMs.
"""
from __future__ import annotations

import contextlib
import unittest
import weakref
from unittest import mock

from dataset_build.source_qa import sam3_subject_instances as sam3
from dataset_build.tools import eval_subject_instance_selector as selector


class FakeTensor:
    """Stands in for one GPU-resident tensor of the batch."""


class FakeInputs(dict):
    """Processor output: a mapping that is moved to the device and unpacked."""

    def to(self, device):
        self.device = device
        return self


class FakeDet:
    device = "cuda:0"

    def __call__(self, **kwargs):
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")


class FakeCuda:
    def __init__(self) -> None:
        self.empty_cache_calls = 0

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1


class FakeTorch:
    def __init__(self) -> None:
        self.cuda = FakeCuda()

    @staticmethod
    def inference_mode():
        return contextlib.nullcontext()


class FakeMasker:
    mask_threshold = 0.5

    def __init__(self, inputs: FakeInputs) -> None:
        self._torch = FakeTorch()
        self._det = FakeDet()
        self._inputs = inputs

    def _as_pil(self, path):
        return object(), (32, 48)

    def _proc(self, **kwargs):
        inputs, self._inputs = self._inputs, None
        return inputs


class Sam3BatchFallbackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            {"asset_id": f"a{index}", "source_path": f"/tmp/wp3fix-{index}.jpg",
             "main_subject": "person"}
            for index in range(2)
        ]
        self.works = ["/tmp/wp3fix-work-0", "/tmp/wp3fix-work-1"]

    def test_a_failed_batch_releases_its_tensors_before_the_per_image_retry(self) -> None:
        tensor = FakeTensor()
        inputs = FakeInputs(pixel_values=tensor)
        alive = (weakref.ref(inputs), weakref.ref(tensor))
        masker = FakeMasker(inputs)
        del inputs, tensor

        observed: list[tuple[int, bool, bool]] = []

        def retry(_masker, row, _work, _min_score, _dedupe_iou):
            observed.append((
                masker._torch.cuda.empty_cache_calls,
                alive[0]() is None,
                alive[1]() is None,
            ))
            return {"asset_id": row["asset_id"], "status": "ready", "proposals": []}

        with mock.patch.object(selector, "_sam_proposals", retry):
            out = sam3._sam_forward_batch(masker, self.rows, self.works, 0.4, 0.8)

        self.assertEqual([row["asset_id"] for row in out], ["a0", "a1"])
        # The cache is emptied once, and both the processor output and the
        # tensors it carried are unreachable, before the first retry runs.
        self.assertEqual(observed, [(1, True, True)] * 2)
        self.assertEqual(masker._torch.cuda.empty_cache_calls, 1)

    def test_a_per_image_failure_is_still_reported_per_row(self) -> None:
        masker = FakeMasker(FakeInputs(pixel_values=FakeTensor()))

        def retry(*_args, **_kwargs):
            raise RuntimeError("still out of memory")

        with mock.patch.object(selector, "_sam_proposals", retry):
            out = sam3._sam_forward_batch(masker, self.rows, self.works, 0.4, 0.8)

        self.assertEqual([row["status"] for row in out], ["sam_miss", "sam_miss"])
        self.assertEqual([row["asset_id"] for row in out], ["a0", "a1"])


if __name__ == "__main__":
    unittest.main()
