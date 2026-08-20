"""Single-chain 3B hard-defect validation contract."""
from __future__ import annotations

import hashlib
import threading
from typing import Any, Mapping

from .api_cache import canonical_json
from .persistence import AuditStore
from .prompts import validation_request
from .responses import CachedResponsesClient


class ChainValidator:
    def __init__(
        self, client: CachedResponsesClient, audit: AuditStore, *, concurrency: int = 1,
        campaign_id: str = "",
    ) -> None:
        self.client = client
        self.audit = audit
        self._semaphore = threading.BoundedSemaphore(max(1, int(concurrency)))
        self.campaign_id = campaign_id

    def validate(
        self, *, source_sha256: str, branch_id: str,
        source: Mapping[str, Any], global_after: Mapping[str, Any],
        final_after: Mapping[str, Any], assignment: Mapping[str, Any],
    ) -> dict[str, Any]:
        spec = validation_request(
            self.client.adapter.endpoint, source, global_after, final_after, assignment
        )
        with self._semaphore:
            result = self.client.request(spec)
        parsed = dict(result.response["parsed"])
        self.audit.record_request_context(
            result.request_hash, self.campaign_id, source_sha256, "chain_verify",
            bool(result.cache_hit),
        )
        validation_id = "val_" + hashlib.sha256(canonical_json({
            "source": source_sha256, "branch": branch_id,
            "global": global_after["sha256"], "final": final_after["sha256"],
            "request_hash": result.request_hash,
        }).encode()).hexdigest()[:24]
        record = {
            "validation_id": validation_id, "source_sha256": source_sha256,
            "branch_id": branch_id, "request_hash": result.request_hash,
            "passed": bool(parsed["passed"]), "defects": list(parsed["defects"]),
            "raw": parsed,
        }
        self.audit.record_validation(record)
        return {
            **record, "cache_hit": result.cache_hit, "usage": result.usage,
        }


__all__ = ["ChainValidator"]
