"""Inter-rater reliability utilities for human and LLM calibration."""

from collections import Counter
from collections.abc import Hashable, Sequence
from typing import Literal


def cohen_kappa(
    rater_a: Sequence[Hashable],
    rater_b: Sequence[Hashable],
    *,
    labels: Sequence[Hashable] | None = None,
) -> float:
    """Compute Cohen's kappa for two raters over categorical labels."""
    if len(rater_a) != len(rater_b):
        raise ValueError("rater_a and rater_b must have the same number of ratings")
    if not rater_a:
        raise ValueError("at least one paired rating is required")

    categories = (
        list(labels)
        if labels is not None
        else sorted(set(rater_a) | set(rater_b))  # type: ignore[type-var]  # labels are sortable at runtime
    )
    if not categories:
        raise ValueError("at least one category label is required")

    n = len(rater_a)
    observed = sum(1 for a, b in zip(rater_a, rater_b, strict=True) if a == b) / n
    counts_a = Counter(rater_a)
    counts_b = Counter(rater_b)
    expected = sum((counts_a[label] / n) * (counts_b[label] / n) for label in categories)

    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return round((observed - expected) / (1.0 - expected), 6)


def weighted_cohen_kappa(
    rater_a: Sequence[Hashable],
    rater_b: Sequence[Hashable],
    *,
    labels: Sequence[Hashable],
    weighting: Literal["linear", "quadratic"] = "quadratic",
) -> float:
    """Compute weighted Cohen's kappa for ordered categorical labels."""
    if len(rater_a) != len(rater_b):
        raise ValueError("rater_a and rater_b must have the same number of ratings")
    if not rater_a:
        raise ValueError("at least one paired rating is required")
    if len(labels) < 2:
        raise ValueError("at least two ordered category labels are required")
    if weighting not in {"linear", "quadratic"}:
        raise ValueError("weighting must be 'linear' or 'quadratic'")

    categories = list(labels)
    label_index = {label: index for index, label in enumerate(categories)}
    unknown = (set(rater_a) | set(rater_b)) - set(label_index)
    if unknown:
        raise ValueError(
            f"ratings contain labels not present in labels: {sorted(unknown)}"  # type: ignore[type-var]
        )

    n = len(rater_a)
    max_distance = len(categories) - 1
    observed_disagreement = 0.0
    for a, b in zip(rater_a, rater_b, strict=True):
        observed_disagreement += _category_distance(
            label_index[a],
            label_index[b],
            max_distance=max_distance,
            weighting=weighting,
        )
    observed_disagreement /= n

    counts_a = Counter(rater_a)
    counts_b = Counter(rater_b)
    expected_disagreement = 0.0
    for label_a in categories:
        for label_b in categories:
            expected_disagreement += (
                (counts_a[label_a] / n)
                * (counts_b[label_b] / n)
                * _category_distance(
                    label_index[label_a],
                    label_index[label_b],
                    max_distance=max_distance,
                    weighting=weighting,
                )
            )

    if expected_disagreement == 0.0:
        return 1.0 if observed_disagreement == 0.0 else 0.0
    return round(1.0 - (observed_disagreement / expected_disagreement), 6)


def fleiss_kappa(
    item_ratings: Sequence[Sequence[Hashable]],
    *,
    labels: Sequence[Hashable] | None = None,
) -> float:
    """Compute Fleiss' kappa for multiple raters over categorical labels.

    Args:
        item_ratings: Matrix shaped as items x raters. Every item must have the
            same number of ratings.
        labels: Optional fixed category order. If omitted, labels are inferred.
    """
    if not item_ratings:
        raise ValueError("at least one rated item is required")

    n_raters = len(item_ratings[0])
    if n_raters < 2:
        raise ValueError("at least two raters are required")
    if any(len(row) != n_raters for row in item_ratings):
        raise ValueError("every item must have the same number of ratings")

    categories = (
        list(labels)
        if labels is not None
        else sorted({rating for row in item_ratings for rating in row})  # type: ignore[type-var]
    )
    if not categories:
        raise ValueError("at least one category label is required")

    n_items = len(item_ratings)
    category_totals: Counter[Hashable] = Counter()
    item_agreements: list[float] = []

    for row in item_ratings:
        counts = Counter(row)
        category_totals.update(counts)
        agreement = sum(counts[label] * (counts[label] - 1) for label in categories)
        item_agreements.append(agreement / (n_raters * (n_raters - 1)))

    observed = sum(item_agreements) / n_items
    expected = sum(
        (category_totals[label] / (n_items * n_raters)) ** 2 for label in categories
    )

    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return round((observed - expected) / (1.0 - expected), 6)


def _category_distance(
    left_index: int,
    right_index: int,
    *,
    max_distance: int,
    weighting: Literal["linear", "quadratic"],
) -> float:
    distance = abs(left_index - right_index) / max_distance
    if weighting == "quadratic":
        return distance**2
    return distance
