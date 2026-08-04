"""Dataset over the sft2seg indexed shards.

The model side consumes an *already converted* two-segment record: the seven
-> two segment rewrite (spec 4.2) is owned by the S0-DATA task, and this loader
refuses to invent it. Local assembly from the seven canonical fields exists but
is opt-in (``allow_local_assembly``) and is meant only for mock/unit tests --
otherwise a schema mismatch on the data side would be silently papered over
here and show up as a quality regression much later.

Image decode + spec-5 resize happen in ``__getitem__`` so they run in the
dataloader worker processes; the collator only does tensorisation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from torch.utils.data import Dataset

from .constants import CANONICAL_COLOR_FIELDS, CANONICAL_WHERE_FIELD, LEGACY_SEGMENT_TAGS
from .imageproc import ImageGeometry, ImageRejected, prepare_image
from .shards import MemberRef, ShardIndex, ShardStore

INSTRUCTION_KEYS = ("instruction", "user_instruction", "prompt", "request", "user_text")
WHERE_KEYS = ("where", "where_text", "where_body", "segment_where")
COLOR_KEYS = ("color", "color_text", "color_body", "segment_color")
IMAGE_PATH_KEYS = ("image_path", "input_image_path", "i_in_path", "image")
RECORD_ROLES = ("record", "json", "meta", "label")
IMAGE_ROLES = ("image", "jpg", "png", "i_in", "input_image")


class RecordSchemaError(RuntimeError):
    pass


def _first(d: dict[str, Any], keys) -> Any:
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None


def assemble_color_segment(record: dict[str, Any], sep: str = "\n") -> str:
    """Spec 4.2 ordering. Only used behind ``allow_local_assembly``."""
    parts = []
    for f in CANONICAL_COLOR_FIELDS:
        v = record.get(f)
        if v is None:
            raise RecordSchemaError(f"canonical field {f!r} missing; cannot assemble color segment")
        parts.append(str(v).strip())
    closing = record.get("closing") or record.get("closure") or record.get("conclusion")
    if isinstance(closing, str) and closing.strip():
        parts.append(closing.strip())
    return sep.join(parts)


@dataclass
class Sft2SegSample:
    sample_id: str
    image: Any  # PIL.Image.Image, already spec-5 sized
    geometry: ImageGeometry
    instruction: str
    where_text: str
    color_text: str
    meta: dict[str, Any]


class Sft2SegDataset(Dataset):
    def __init__(
        self,
        index: ShardIndex,
        store: ShardStore,
        image_root: str | None = None,
        allow_local_assembly: bool = False,
        assembly_separator: str = "\n",
        strict_legacy_tag_check: bool = True,
    ):
        self.index = index
        self.store = store
        self.image_root = image_root
        self.allow_local_assembly = allow_local_assembly
        self.assembly_separator = assembly_separator
        self.strict_legacy_tag_check = strict_legacy_tag_check

    def __len__(self) -> int:
        return len(self.index)

    # -- member resolution --------------------------------------------------
    @staticmethod
    def _pick(members: dict[str, MemberRef], roles) -> MemberRef | None:
        for r in roles:
            if r in members:
                return members[r]
        return None

    def load_record(self, ref) -> dict[str, Any]:
        member = self._pick(ref.members, RECORD_ROLES)
        if member is not None:
            return json.loads(self.store.read(member).decode("utf-8"))
        # index row itself may carry the fields inline
        if any(k in ref.meta for k in WHERE_KEYS + COLOR_KEYS + INSTRUCTION_KEYS):
            return dict(ref.meta)
        raise RecordSchemaError(
            f"sample {ref.sample_id}: no record member (tried {RECORD_ROLES}) and no inline "
            f"fields; member roles present={sorted(ref.members)}"
        )

    def load_image_bytes(self, ref, record: dict[str, Any]):
        member = self._pick(ref.members, IMAGE_ROLES)
        if member is not None:
            return self.store.read(member)
        path = _first(record, IMAGE_PATH_KEYS) or _first(ref.meta, IMAGE_PATH_KEYS)
        if path is None:
            raise ImageRejected(
                "image_missing",
                f"sample {ref.sample_id}: no image member (tried {IMAGE_ROLES}) and no path field",
            )
        if self.image_root and not str(path).startswith("/"):
            path = f"{self.image_root.rstrip('/')}/{path}"
        return path

    # -- segments -----------------------------------------------------------
    def extract_segments(self, sample_id: str, record: dict[str, Any]) -> tuple[str, str]:
        where = _first(record, WHERE_KEYS)
        color = _first(record, COLOR_KEYS)
        if where is None and self.allow_local_assembly:
            where = record.get(CANONICAL_WHERE_FIELD)
        if color is None and self.allow_local_assembly:
            color = assemble_color_segment(record, self.assembly_separator)
        if not isinstance(where, str) or not where.strip():
            raise RecordSchemaError(
                f"sample {sample_id}: no non-empty where segment (tried {WHERE_KEYS}); "
                f"record keys={sorted(record)}"
            )
        if not isinstance(color, str) or not color.strip():
            raise RecordSchemaError(
                f"sample {sample_id}: no non-empty color segment (tried {COLOR_KEYS}); "
                f"record keys={sorted(record)}"
            )
        where, color = where.strip(), color.strip()
        if self.strict_legacy_tag_check:
            blob = where + color
            leaked = [t for t in LEGACY_SEGMENT_TAGS if t in blob]
            if leaked:
                raise RecordSchemaError(
                    f"sample {sample_id}: legacy seven-segment tags leaked into the target: {leaked}"
                )
        return where, color

    def __getitem__(self, i: int) -> Sft2SegSample:
        ref = self.index[i]
        record = self.load_record(ref)
        instruction = _first(record, INSTRUCTION_KEYS) or _first(ref.meta, INSTRUCTION_KEYS)
        if instruction is None:
            raise RecordSchemaError(
                f"sample {ref.sample_id}: no instruction (tried {INSTRUCTION_KEYS}); "
                f"record keys={sorted(record)}"
            )
        where_text, color_text = self.extract_segments(ref.sample_id, record)
        img_src = self.load_image_bytes(ref, record)
        image, geometry = prepare_image(img_src)
        meta = {
            k: v for k, v in record.items()
            if k in ("split", "build", "source_image_id", "lut_id", "winner_confidence")
        }
        return Sft2SegSample(
            sample_id=ref.sample_id,
            image=image,
            geometry=geometry,
            instruction=instruction,
            where_text=where_text,
            color_text=color_text,
            meta=meta,
        )
