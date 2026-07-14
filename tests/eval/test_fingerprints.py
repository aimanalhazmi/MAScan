from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel

from mascan.eval.fingerprints import file_sha256, model_sha256


class _FingerprintModel(BaseModel):
    name: str
    values: list[int]


def test_file_sha256_changes_with_file_contents():
    path = Path("tmp") / "test_fingerprints" / uuid4().hex / "artifact.txt"
    path.parent.mkdir(parents=True, exist_ok=False)
    path.write_text("one", encoding="utf-8")
    first = file_sha256(path)
    path.write_text("two", encoding="utf-8")

    assert file_sha256(path) != first


def test_model_sha256_is_stable_for_equivalent_models():
    first = _FingerprintModel(name="case", values=[1, 2])
    second = _FingerprintModel.model_validate({"values": [1, 2], "name": "case"})

    assert model_sha256(first) == model_sha256(second)
