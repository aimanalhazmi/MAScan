"""Stable SHA-256 fingerprints for experiment reproducibility."""

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel


class ArtifactFingerprint(BaseModel):
    artifact: str
    sha256: str
    method: str


def file_sha256(path: str | Path) -> str:
    """Hash file bytes exactly as stored on disk."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_sha256(model: BaseModel) -> str:
    """Hash a Pydantic model as canonical JSON, independent of formatting."""
    canonical = json.dumps(
        model.model_dump(mode="json", by_alias=True, exclude_none=False),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
