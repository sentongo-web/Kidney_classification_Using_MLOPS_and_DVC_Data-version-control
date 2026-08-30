import numpy as np

from cnnClassifier.components.model_evaluation_mlflow import compute_expected_calibration_error


def test_ece_matches_hand_computed_value_for_two_bins():
    # Bin [0.9, 1.0]: confidences {0.95, 0.95}, both correct -> accuracy 1.0
    # Bin [0.5, 0.6]: confidences {0.55, 0.55}, both wrong   -> accuracy 0.0
    confidences = np.array([0.95, 0.95, 0.55, 0.55])
    predicted_labels = np.array([0, 1, 0, 1])
    true_labels = np.array([0, 1, 1, 0])

    ece = compute_expected_calibration_error(confidences, predicted_labels, true_labels, n_bins=10)

    expected = 0.5 * abs(1.0 - 0.95) + 0.5 * abs(0.0 - 0.55)
    assert np.isclose(ece, expected)


def test_perfectly_calibrated_predictions_yield_zero_ece():
    # Confidence equals empirical accuracy in every populated bin.
    confidences = np.array([1.0, 1.0, 1.0, 1.0])
    predicted_labels = np.array([1, 1, 1, 1])
    true_labels = np.array([1, 1, 1, 1])

    ece = compute_expected_calibration_error(confidences, predicted_labels, true_labels, n_bins=10)

    assert np.isclose(ece, 0.0)


def test_empty_cohort_returns_zero_without_raising():
    ece = compute_expected_calibration_error(np.array([]), np.array([]), np.array([]))
    assert ece == 0.0


def test_ece_is_bounded_between_zero_and_one():
    rng = np.random.default_rng(0)
    confidences = rng.uniform(0.5, 1.0, size=500)
    predicted_labels = rng.integers(0, 2, size=500)
    true_labels = rng.integers(0, 2, size=500)

    ece = compute_expected_calibration_error(confidences, predicted_labels, true_labels)

    assert 0.0 <= ece <= 1.0
