import json
import sys
from pathlib import Path
from uuid import uuid4

from mascan.eval.readiness import GoldExperimentManifest


def test_init_gold_pricing_writes_manifest_models(mocker):
    test_dir = Path("tmp") / "test_pricing_cli" / uuid4().hex
    test_dir.mkdir(parents=True, exist_ok=False)
    manifest = GoldExperimentManifest(
        gold_standard_file="gold.json",
        expected_case_count=1,
        systems=[
            {"system_id": "a", "model": "m1", "response_file": "a.json"},
            {"system_id": "b", "model": "m1", "response_file": "b.json"},
            {"system_id": "c", "model": "m2", "response_file": "c.json"},
        ],
        pricing_file=str(test_dir / "model_pricing.json"),
    )
    manifest_path = test_dir / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(), encoding="utf-8")
    mocker.patch.object(
        sys,
        "argv",
        [
            "init_gold_pricing.py",
            "--manifest",
            str(manifest_path),
            "--captured-at",
            "2026-07-12",
        ],
    )

    from scripts import init_gold_pricing

    assert init_gold_pricing.main() == 0
    payload = json.loads((test_dir / "model_pricing.json").read_text(encoding="utf-8"))

    assert payload["captured_at"] == "2026-07-12"
    assert [entry["model"] for entry in payload["prices"]] == ["m1", "m2"]
    assert payload["prices"][0]["prompt_usd_per_1m_tokens"] == 0.0
