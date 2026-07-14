from collections import Counter

import pytest

from mascan.eval.gold_experiment import ModelResponseRecord
from mascan.eval.gold_standard import load_gold_standard
from mascan.eval.human_calibration import (
    HumanAnswerKeyEntry,
    build_human_calibration_bundle,
    filter_packet_for_rater,
    partition_cases_among_raters,
    validate_human_answer_key,
)


def _records(case_ids: list[str]) -> list[ModelResponseRecord]:
    records: list[ModelResponseRecord] = []
    for case_id in case_ids:
        for system_id in ["mascan", "zero_shot_same_model", "frontier_model"]:
            records.append(
                ModelResponseRecord(
                    case_id=case_id,
                    system_id=system_id,
                    model=f"{system_id}-model",
                    response_text=f"{system_id} response for {case_id}",
                )
            )
    return records


def test_build_human_calibration_bundle_partitions_all_cases_across_five_raters():
    dataset = load_gold_standard()
    case_ids = [case.case_id for case in dataset.cases]
    records = _records(case_ids)
    raters = ["rater_1", "rater_2", "rater_3", "rater_4", "rater_5"]

    bundle = build_human_calibration_bundle(
        dataset,
        records,
        systems=["mascan", "zero_shot_same_model", "frontier_model"],
        rater_ids=raters,
        cases_per_rater=5,
        seed=7,
    )

    assert len(bundle.packet.items) == 25
    assert len(bundle.packet.selected_case_ids) == 25
    assert len(bundle.answer_key) == 75
    assert bundle.packet.cases_per_rater == 5
    assert len(bundle.packet.rater_assignments) == 5
    assigned_case_ids = [
        case_id
        for assignment in bundle.packet.rater_assignments
        for case_id in assignment.case_ids
    ]
    assert sorted(assigned_case_ids) == sorted(case_ids)
    assert {output.label for item in bundle.packet.items for output in item.outputs} == {
        "A",
        "B",
        "C",
    }
    label_counts = Counter((entry.system_id, entry.label) for entry in bundle.answer_key)
    for system_id in ["mascan", "zero_shot_same_model", "frontier_model"]:
        counts = [label_counts[(system_id, label)] for label in ["A", "B", "C"]]
        assert max(counts) - min(counts) <= 1
        assert sum(counts) == 25


def test_build_human_calibration_bundle_is_reproducible():
    dataset = load_gold_standard()
    case_ids = [case.case_id for case in dataset.cases]
    records = _records(case_ids)
    raters = ["rater_1", "rater_2", "rater_3", "rater_4", "rater_5"]

    first = build_human_calibration_bundle(
        dataset,
        records,
        systems=["mascan", "zero_shot_same_model", "frontier_model"],
        rater_ids=raters,
        seed=11,
    )
    second = build_human_calibration_bundle(
        dataset,
        records,
        systems=["mascan", "zero_shot_same_model", "frontier_model"],
        rater_ids=raters,
        seed=11,
    )

    assert first.packet.selected_case_ids == second.packet.selected_case_ids
    assert first.packet.rater_assignments == second.packet.rater_assignments
    assert first.answer_key == second.answer_key


def test_filter_packet_for_rater_returns_only_assigned_cases():
    dataset = load_gold_standard()
    records = _records([case.case_id for case in dataset.cases])
    bundle = build_human_calibration_bundle(
        dataset,
        records,
        systems=["mascan", "zero_shot_same_model", "frontier_model"],
        rater_ids=["rater_1", "rater_2", "rater_3", "rater_4", "rater_5"],
        seed=11,
    )

    rater_packet = filter_packet_for_rater(bundle.packet, rater_id="rater_1")

    assert len(rater_packet.items) == 5
    assigned = next(
        assignment.case_ids
        for assignment in bundle.packet.rater_assignments
        if assignment.rater_id == "rater_1"
    )
    assert [item.case_id for item in rater_packet.items] == sorted(assigned)


def test_validate_human_answer_key_accepts_bundle_output():
    dataset = load_gold_standard()
    records = _records([case.case_id for case in dataset.cases[:25]])
    systems = ["mascan", "zero_shot_same_model", "frontier_model"]
    bundle = build_human_calibration_bundle(
        dataset,
        records,
        systems=systems,
        rater_ids=["rater_1", "rater_2", "rater_3", "rater_4", "rater_5"],
        seed=11,
    )

    report = validate_human_answer_key(
        bundle.packet,
        bundle.answer_key,
        expected_systems=systems,
    )

    assert report.is_valid
    assert report.expected_entry_count == 75
    assert report.observed_entry_count == 75
    assert report.assignment_mismatches == []


def test_validate_human_answer_key_rejects_label_imbalance():
    dataset = load_gold_standard()
    records = _records([case.case_id for case in dataset.cases[:25]])
    systems = ["mascan", "zero_shot_same_model", "frontier_model"]
    bundle = build_human_calibration_bundle(
        dataset,
        records,
        systems=systems,
        rater_ids=["rater_1", "rater_2", "rater_3", "rater_4", "rater_5"],
        seed=11,
    )
    fixed_label_systems = {
        "A": "mascan",
        "B": "zero_shot_same_model",
        "C": "frontier_model",
    }
    imbalanced_answer_key = [
        HumanAnswerKeyEntry(
            case_id=entry.case_id,
            label=entry.label,
            system_id=fixed_label_systems[entry.label],
            model=f"{fixed_label_systems[entry.label]}-model",
        )
        for entry in bundle.answer_key
    ]

    report = validate_human_answer_key(
        bundle.packet,
        imbalanced_answer_key,
        expected_systems=systems,
    )

    assert not report.is_valid
    assert report.system_mismatches == []
    assert report.label_imbalances


def test_validate_human_answer_key_reports_missing_duplicate_and_system_gap():
    dataset = load_gold_standard()
    records = _records([case.case_id for case in dataset.cases[:25]])
    systems = ["mascan", "zero_shot_same_model", "frontier_model"]
    bundle = build_human_calibration_bundle(
        dataset,
        records,
        systems=systems,
        rater_ids=["rater_1", "rater_2", "rater_3", "rater_4", "rater_5"],
        seed=11,
    )
    first = bundle.answer_key[0]
    bad_answer_key = [
        *bundle.answer_key[1:],
        HumanAnswerKeyEntry(
            case_id=first.case_id,
            label=bundle.answer_key[1].label,
            system_id=bundle.answer_key[1].system_id,
            model=bundle.answer_key[1].model,
        ),
    ]

    report = validate_human_answer_key(
        bundle.packet,
        bad_answer_key,
        expected_systems=systems,
    )

    assert not report.is_valid
    assert f"{first.case_id}|{first.label}" in report.missing_entries
    assert report.duplicate_entries
    assert report.system_mismatches


def test_partition_cases_among_raters_requires_exact_case_count():
    dataset = load_gold_standard()
    with pytest.raises(ValueError, match="Need exactly 25 cases"):
        partition_cases_among_raters(
            dataset.cases[:10],
            ["rater_1", "rater_2", "rater_3", "rater_4", "rater_5"],
            cases_per_rater=5,
        )


def test_build_human_calibration_bundle_requires_complete_outputs():
    dataset = load_gold_standard()
    records = _records([dataset.cases[0].case_id])

    with pytest.raises(ValueError, match="Need exactly 25 cases"):
        build_human_calibration_bundle(
            dataset,
            records,
            systems=["mascan", "zero_shot_same_model", "frontier_model"],
            rater_ids=["rater_1", "rater_2", "rater_3", "rater_4", "rater_5"],
        )
