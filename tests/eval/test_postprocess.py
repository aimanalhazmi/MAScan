from pathlib import Path
from uuid import uuid4

from mascan.eval.postprocess import run_gold_postprocess
from mascan.eval.readiness import load_experiment_manifest
from tests.eval._gold_fixtures import build_complete_gold_run_fixture


def test_run_gold_postprocess_regenerates_derived_artifacts():
    out_dir = Path("tmp") / "test_postprocess" / uuid4().hex
    manifest_path = build_complete_gold_run_fixture(out_dir=out_dir)
    manifest = load_experiment_manifest(manifest_path)

    for filename in [
        "judged_all_priced.json",
        "system_summary.json",
        "case_trace.json",
        "mascan_vs_zero_shot.json",
        "mascan_vs_frontier.json",
        "gold_experiment_report.md",
    ]:
        (out_dir / filename).unlink()

    report = run_gold_postprocess(manifest, base_dir=".")

    assert report.is_ready is True
    assert (out_dir / "judged_all_priced.json").exists()
    assert (out_dir / "system_summary.json").exists()
    assert (out_dir / "case_trace.json").exists()
    assert (out_dir / "gold_experiment_report.md").exists()
