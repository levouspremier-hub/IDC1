"""Shared contract base: deterministic serialization and hashing.

Every contract inherits ContractBase so it gets:
  - strict schema (extra top-level keys are rejected),
  - JSON round-trip helpers (to_json / from_json),
  - a deterministic content hash (sha256 over canonical JSON),
    excluding any field literally named ``hash`` (which stores the frozen hash).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Self

from pydantic import BaseModel, ConfigDict


def canonical_json(obj: Any) -> str:
    """Canonical JSON with sorted keys, so dict order never affects the hash."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class ContractBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def to_json(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_json(cls, raw: str) -> Self:
        return cls.model_validate_json(raw)

    def content_hash(self) -> str:
        """Deterministic sha256 over all fields except a ``hash`` field, if any."""
        exclude = {"hash"} if "hash" in type(self).model_fields else set()
        data = self.model_dump(mode="json", exclude=exclude)
        return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()
