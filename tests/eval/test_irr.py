import pytest

from mascan.eval.irr import cohen_kappa, fleiss_kappa, weighted_cohen_kappa


def test_cohen_kappa_perfect_agreement():
    assert cohen_kappa([1, 2, 3, 1], [1, 2, 3, 1]) == 1.0


def test_cohen_kappa_known_partial_agreement():
    # Observed agreement = 2/4. Expected agreement = 3/8.
    # Kappa = (0.5 - 0.375) / (1 - 0.375) = 0.2
    assert cohen_kappa(["A", "A", "B", "B"], ["A", "B", "B", "A"]) == 0.0
    assert cohen_kappa(["A", "A", "B", "B"], ["A", "A", "A", "B"]) == 0.5


def test_cohen_kappa_validates_lengths():
    with pytest.raises(ValueError):
        cohen_kappa([1], [1, 2])


def test_weighted_cohen_kappa_handles_ordered_depth_labels():
    assert weighted_cohen_kappa([1, 2, 3], [1, 2, 3], labels=[1, 2, 3]) == 1.0

    nominal = cohen_kappa([1, 1, 3, 3], [1, 2, 2, 3], labels=[1, 2, 3])
    weighted = weighted_cohen_kappa(
        [1, 1, 3, 3],
        [1, 2, 2, 3],
        labels=[1, 2, 3],
        weighting="quadratic",
    )

    assert nominal < weighted < 1.0


def test_weighted_cohen_kappa_rejects_unknown_labels():
    with pytest.raises(ValueError):
        weighted_cohen_kappa([1, 4], [1, 3], labels=[1, 2, 3])


def test_fleiss_kappa_perfect_agreement():
    ratings = [
        [1, 1, 1, 1],
        [2, 2, 2, 2],
        [3, 3, 3, 3],
    ]

    assert fleiss_kappa(ratings) == 1.0


def test_fleiss_kappa_known_example():
    ratings = [
        ["A", "A", "A", "B"],
        ["A", "B", "B", "B"],
        ["B", "B", "B", "B"],
    ]

    assert fleiss_kappa(ratings) == pytest.approx(0.25, abs=0.00001)


def test_fleiss_kappa_validates_rectangular_matrix():
    with pytest.raises(ValueError):
        fleiss_kappa([[1, 1], [1]])
