"""Unit tests for VisualCandidateClassifier module."""

# ruff: noqa: N806, N803


import numpy as np
from sklearn.linear_model import LogisticRegression

from dta.perception.visual_classifier import VisualCandidateClassifier


def test_visual_classifier_initialization() -> None:
    classifier = VisualCandidateClassifier(model_path="non_existent_model.pkl", threshold=0.60)
    assert classifier.threshold == 0.60
    assert classifier.classifier is None


def test_feature_extraction_single() -> None:
    classifier = VisualCandidateClassifier(model_path="non_existent_model.pkl")
    # Synthetic BGR crop 32x32
    crop_bgr = np.zeros((32, 32, 3), dtype=np.uint8)
    crop_bgr[:, :] = [0, 255, 0]

    features = classifier.extract_features(crop_bgr)
    assert isinstance(features, np.ndarray)
    assert features.shape == (576,)
    assert features.dtype == np.float32


def test_feature_extraction_batch() -> None:
    classifier = VisualCandidateClassifier(model_path="non_existent_model.pkl")
    crops = [
        np.zeros((32, 32, 3), dtype=np.uint8),
        np.ones((48, 24, 3), dtype=np.uint8) * 128,
    ]

    features = classifier.extract_batch_features(crops)
    assert isinstance(features, np.ndarray)
    assert features.shape == (2, 576)
    assert features.dtype == np.float32


def test_prediction_with_mock_model(tmp_path) -> None:
    # Train a dummy linear model on synthetic 576-d data
    X = np.random.randn(20, 576).astype(np.float32)
    y = np.array([1 if i % 2 == 0 else 0 for i in range(20)])
    model = LogisticRegression(C=0.1)
    model.fit(X, y)

    import pickle
    model_file = tmp_path / "test_model.pkl"
    with open(model_file, "wb") as f:
        pickle.dump(model, f)

    classifier = VisualCandidateClassifier(model_path=str(model_file), threshold=0.50)
    assert classifier.classifier is not None

    crop_bgr = np.zeros((32, 32, 3), dtype=np.uint8)
    proba = classifier.predict_proba(crop_bgr)
    assert 0.0 <= proba <= 1.0

    is_ent = classifier.is_entity(crop_bgr)
    assert isinstance(is_ent, bool)
