"""Build human-rater packets for LLM-judge calibration."""

import random
from collections.abc import Sequence

from pydantic import BaseModel, Field

from mascan.eval.gold_experiment import ModelResponseRecord, response_lookup
from mascan.eval.gold_standard import GoldStandardCase, GoldStandardDataset

DEFAULT_HUMAN_CALIBRATION_SEED = 20260712
DEFAULT_CASES_PER_RATER = 5

HUMAN_CALIBRATION_INSTRUCTIONS = (
    "For each assigned case, read the prompt, expected output, categorization "
    "targets, and anonymized model responses. For Analytical Depth, enumerate the "
    "distinct causal claims in the response, score each claim with the 1-3 "
    "rubric, then enter one integer summary score: 1 if the average is below "
    "1.5, 2 if it is at least 1.5 and below 2.5, and 3 if it is 2.5 or above. "
    "For Categorization Accuracy, inspect each target factor row. Mark true "
    "only when the response discusses that factor and primarily places it in "
    "the expected PESTEL bucket; mark false when it is missing, placed in the "
    "wrong bucket, or treated only under a misleading category."
)


class HumanPacketOutput(BaseModel):
    label: str
    response_text: str


class HumanPacketItem(BaseModel):
    case_id: str
    case_title: str
    prompt: str
    expected_output: dict[str, list[str]]
    category_targets: list[dict[str, str]]
    outputs: list[HumanPacketOutput]


class HumanAnswerKeyEntry(BaseModel):
    case_id: str
    label: str
    system_id: str
    model: str


class HumanRaterAssignment(BaseModel):
    rater_id: str
    case_ids: list[str]


class HumanCalibrationPacket(BaseModel):
    seed: int
    selected_case_ids: list[str]
    cases_per_rater: int
    rater_assignments: list[HumanRaterAssignment] = Field(default_factory=list)
    instructions: str
    rating_scale: dict[str, str]
    items: list[HumanPacketItem]


class HumanCalibrationBundle(BaseModel):
    packet: HumanCalibrationPacket
    answer_key: list[HumanAnswerKeyEntry] = Field(default_factory=list)


class HumanAnswerKeyValidationReport(BaseModel):
    is_valid: bool
    expected_entry_count: int
    observed_entry_count: int
    missing_entries: list[str] = Field(default_factory=list)
    duplicate_entries: list[str] = Field(default_factory=list)
    unexpected_entries: list[str] = Field(default_factory=list)
    duplicate_packet_outputs: list[str] = Field(default_factory=list)
    empty_answer_key_fields: list[str] = Field(default_factory=list)
    system_mismatches: list[str] = Field(default_factory=list)
    label_imbalances: list[str] = Field(default_factory=list)
    assignment_mismatches: list[str] = Field(default_factory=list)


def eligible_cases(
    dataset: GoldStandardDataset,
    records: Sequence[ModelResponseRecord],
    *,
    systems: Sequence[str],
) -> list[GoldStandardCase]:
    lookup = response_lookup(records)
    return [
        case
        for case in dataset.cases
        if all((case.case_id, system_id) in lookup for system_id in systems)
    ]


def partition_cases_among_raters(
    cases: Sequence[GoldStandardCase],
    rater_ids: Sequence[str],
    *,
    cases_per_rater: int,
    seed: int = DEFAULT_HUMAN_CALIBRATION_SEED,
) -> list[HumanRaterAssignment]:
    """Assign each case to exactly one rater in a reproducible shuffle."""
    if not rater_ids:
        raise ValueError("rater_ids must not be empty")
    if cases_per_rater <= 0:
        raise ValueError("cases_per_rater must be positive")
    expected_total = len(rater_ids) * cases_per_rater
    if len(cases) != expected_total:
        raise ValueError(
            f"Need exactly {expected_total} cases for "
            f"{len(rater_ids)} raters x {cases_per_rater} cases each, "
            f"found {len(cases)}"
        )
    rng = random.Random(seed)
    shuffled = list(cases)
    rng.shuffle(shuffled)
    assignments: list[HumanRaterAssignment] = []
    for index, rater_id in enumerate(rater_ids):
        start = index * cases_per_rater
        stop = start + cases_per_rater
        assignments.append(
            HumanRaterAssignment(
                rater_id=rater_id,
                case_ids=[case.case_id for case in shuffled[start:stop]],
            )
        )
    return assignments


def assigned_rater_id(packet: HumanCalibrationPacket, case_id: str) -> str:
    """Return the single rater assigned to a packet case."""
    for assignment in packet.rater_assignments:
        if case_id in assignment.case_ids:
            return assignment.rater_id
    raise KeyError(f"No rater assignment found for case_id={case_id}")


def filter_packet_for_rater(
    packet: HumanCalibrationPacket,
    *,
    rater_id: str,
) -> HumanCalibrationPacket:
    """Return only the cases assigned to one human rater."""
    assigned_case_ids = {
        case_id
        for assignment in packet.rater_assignments
        if assignment.rater_id == rater_id
        for case_id in assignment.case_ids
    }
    if not assigned_case_ids:
        raise ValueError(f"No cases assigned to rater_id={rater_id!r}")
    items = [item for item in packet.items if item.case_id in assigned_case_ids]
    return packet.model_copy(
        update={
            "items": items,
            "selected_case_ids": [item.case_id for item in items],
        }
    )


def build_human_calibration_bundle(
    dataset: GoldStandardDataset,
    records: Sequence[ModelResponseRecord],
    *,
    systems: Sequence[str],
    rater_ids: Sequence[str],
    cases_per_rater: int = DEFAULT_CASES_PER_RATER,
    seed: int = DEFAULT_HUMAN_CALIBRATION_SEED,
) -> HumanCalibrationBundle:
    """Build a distributed blinded packet covering all assigned gold cases.

  Each rater receives a disjoint subset of cases. Every case still includes
  three anonymized system outputs (A/B/C), but only the assigned rater scores
  that case.
    """
    if not rater_ids:
        raise ValueError("rater_ids must not be empty")
    candidates = eligible_cases(dataset, records, systems=systems)
    assignments = partition_cases_among_raters(
        candidates,
        rater_ids,
        cases_per_rater=cases_per_rater,
        seed=seed,
    )
    selected_cases = [
        case
        for case in candidates
        if case.case_id in {case_id for assignment in assignments for case_id in assignment.case_ids}
    ]
    selected_cases.sort(key=lambda case: case.case_id)
    lookup = response_lookup(records)
    rng = random.Random(seed)

    items: list[HumanPacketItem] = []
    answer_key: list[HumanAnswerKeyEntry] = []
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    system_orders = _balanced_system_orders(
        systems,
        n=len(selected_cases),
        rng=rng,
    )

    for case, system_ids in zip(selected_cases, system_orders, strict=True):
        outputs: list[HumanPacketOutput] = []
        for idx, system_id in enumerate(system_ids):
            label = labels[idx]
            record = lookup[(case.case_id, system_id)]
            outputs.append(
                HumanPacketOutput(label=label, response_text=record.response_text)
            )
            answer_key.append(
                HumanAnswerKeyEntry(
                    case_id=case.case_id,
                    label=label,
                    system_id=system_id,
                    model=record.model,
                )
            )
        items.append(
            HumanPacketItem(
                case_id=case.case_id,
                case_title=case.case_title,
                prompt=case.prompt,
                expected_output=case.expected_output.model_dump(mode="json"),
                category_targets=[
                    target.model_dump(mode="json") for target in case.category_targets
                ],
                outputs=outputs,
            )
        )

    packet = HumanCalibrationPacket(
        seed=seed,
        selected_case_ids=[case.case_id for case in selected_cases],
        cases_per_rater=cases_per_rater,
        rater_assignments=assignments,
        instructions=HUMAN_CALIBRATION_INSTRUCTIONS,
        rating_scale={
            "1": "Surface: mentions external factors without connecting them to business impact.",
            "2": "Analytical: explains direct operational, financial, demand, compliance, or reputational impact.",
            "3": "Strategic: traces impact into a strategic risk, opportunity, recommendation, or shift.",
        },
        items=items,
    )
    return HumanCalibrationBundle(packet=packet, answer_key=answer_key)


def validate_human_answer_key(
    packet: HumanCalibrationPacket,
    answer_key: Sequence[HumanAnswerKeyEntry],
    *,
    expected_systems: Sequence[str] | None = None,
) -> HumanAnswerKeyValidationReport:
    """Validate that the coordinator answer key exactly matches packet labels."""
    expected_keys = [
        _packet_key(item.case_id, output.label)
        for item in packet.items
        for output in item.outputs
    ]
    observed_keys = [_packet_key(entry.case_id, entry.label) for entry in answer_key]
    expected_counts = _counts(expected_keys)
    observed_counts = _counts(observed_keys)
    expected_set = set(expected_counts)
    observed_set = set(observed_counts)

    duplicate_packet_outputs = sorted(
        key for key, count in expected_counts.items() if count > 1
    )
    duplicate_entries = sorted(
        key for key, count in observed_counts.items() if count > 1
    )
    missing_entries = sorted(expected_set - observed_set)
    unexpected_entries = sorted(observed_set - expected_set)
    empty_answer_key_fields = _empty_answer_key_fields(answer_key)
    system_mismatches = _answer_key_system_mismatches(
        packet,
        answer_key,
        expected_systems=expected_systems,
    )
    label_imbalances = _answer_key_label_imbalances(
        packet,
        answer_key,
        expected_systems=expected_systems,
    )
    assignment_mismatches = _assignment_mismatches(packet)

    problems = (
        missing_entries
        or duplicate_entries
        or unexpected_entries
        or duplicate_packet_outputs
        or empty_answer_key_fields
        or system_mismatches
        or label_imbalances
        or assignment_mismatches
    )
    return HumanAnswerKeyValidationReport(
        is_valid=not problems,
        expected_entry_count=len(expected_keys),
        observed_entry_count=len(answer_key),
        missing_entries=missing_entries,
        duplicate_entries=duplicate_entries,
        unexpected_entries=unexpected_entries,
        duplicate_packet_outputs=duplicate_packet_outputs,
        empty_answer_key_fields=empty_answer_key_fields,
        system_mismatches=system_mismatches,
        label_imbalances=label_imbalances,
        assignment_mismatches=assignment_mismatches,
    )


def _assignment_mismatches(packet: HumanCalibrationPacket) -> list[str]:
    mismatches: list[str] = []
    if not packet.rater_assignments:
        return ["packet is missing rater_assignments"]
    assigned_case_ids = [
        case_id for assignment in packet.rater_assignments for case_id in assignment.case_ids
    ]
    packet_case_ids = [item.case_id for item in packet.items]
    if sorted(assigned_case_ids) != sorted(packet_case_ids):
        mismatches.append(
            "rater_assignments case IDs do not match packet items "
            f"(assigned={sorted(assigned_case_ids)}, packet={sorted(packet_case_ids)})"
        )
    if len(assigned_case_ids) != len(set(assigned_case_ids)):
        mismatches.append("rater_assignments contain duplicate case IDs")
    expected_total = len(packet.rater_assignments) * packet.cases_per_rater
    if len(packet_case_ids) != expected_total:
        mismatches.append(
            f"packet should contain {expected_total} cases "
            f"({len(packet.rater_assignments)} raters x "
            f"{packet.cases_per_rater} cases each), found {len(packet_case_ids)}"
        )
    for assignment in packet.rater_assignments:
        if len(assignment.case_ids) != packet.cases_per_rater:
            mismatches.append(
                f"{assignment.rater_id}: expected {packet.cases_per_rater} cases, "
                f"found {len(assignment.case_ids)}"
            )
    return mismatches


def _balanced_system_orders(
    systems: Sequence[str],
    *,
    n: int,
    rng: random.Random,
) -> list[list[str]]:
    """Return per-case system orders with balanced label positions."""
    if not systems:
        raise ValueError("systems must not be empty")
    shuffled = list(systems)
    rng.shuffle(shuffled)
    rotations = [
        [shuffled[(label_index + offset) % len(shuffled)] for label_index in range(len(shuffled))]
        for offset in range(len(shuffled))
    ]
    orders = [rotations[index % len(rotations)] for index in range(n)]
    rng.shuffle(orders)
    return [list(order) for order in orders]


def _answer_key_system_mismatches(
    packet: HumanCalibrationPacket,
    answer_key: Sequence[HumanAnswerKeyEntry],
    *,
    expected_systems: Sequence[str] | None,
) -> list[str]:
    if not expected_systems:
        return []
    expected = set(expected_systems)
    mismatches: list[str] = []
    for item in packet.items:
        systems = {
            entry.system_id
            for entry in answer_key
            if entry.case_id == item.case_id
            and _packet_key(entry.case_id, entry.label)
            in {_packet_key(item.case_id, output.label) for output in item.outputs}
        }
        if systems != expected:
            mismatches.append(
                f"{item.case_id}: expected systems {sorted(expected)}, "
                f"found {sorted(systems)}"
            )
    return mismatches


def _answer_key_label_imbalances(
    packet: HumanCalibrationPacket,
    answer_key: Sequence[HumanAnswerKeyEntry],
    *,
    expected_systems: Sequence[str] | None,
) -> list[str]:
    labels = sorted({output.label for item in packet.items for output in item.outputs})
    if not labels:
        return []
    packet_keys = {
        _packet_key(item.case_id, output.label)
        for item in packet.items
        for output in item.outputs
    }
    systems = (
        sorted(set(expected_systems))
        if expected_systems
        else sorted({entry.system_id for entry in answer_key})
    )
    imbalances: list[str] = []
    for system_id in systems:
        counts = {label: 0 for label in labels}
        for entry in answer_key:
            if (
                entry.system_id == system_id
                and entry.label in counts
                and _packet_key(entry.case_id, entry.label) in packet_keys
            ):
                counts[entry.label] += 1
        values = list(counts.values())
        if values and max(values) - min(values) > 1:
            imbalances.append(
                f"{system_id}: label counts should be balanced; found {counts}"
            )
    return imbalances


def _empty_answer_key_fields(
    answer_key: Sequence[HumanAnswerKeyEntry],
) -> list[str]:
    empty: list[str] = []
    for index, entry in enumerate(answer_key):
        for field in ["case_id", "label", "system_id", "model"]:
            if not getattr(entry, field).strip():
                empty.append(f"{index}:{field}")
    return empty


def _packet_key(case_id: str, label: str) -> str:
    return f"{case_id}|{label}"


def _counts(values: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts
