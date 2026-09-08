"""Visual Candidate Classifier module using MobileNetV3 embeddings and scikit-learn model inference."""

# ruff: noqa: N806, N803



import os
import pickle
from typing import Any

import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

from dta.core.logger import logger


class PipelineWrapper:
    """Wrapper encapsulating StandardScaler and classifier model for clean pickling."""

    def __init__(self, sc: Any, clf: Any) -> None:
        self.sc = sc
        self.clf = clf

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_sc = self.sc.transform(X)
        res = self.clf.predict_proba(X_sc)
        return np.asarray(res)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_sc = self.sc.transform(X)
        res = self.clf.predict(X_sc)
        return np.asarray(res)


class VisualCandidateClassifier:
    """Offline visual candidate classifier using MobileNetV3 Small deep embeddings."""

    def __init__(
        self,
        model_path: str = "models/entity_classifier.pkl",
        threshold: float = 0.50,
    ) -> None:
        self.model_path = model_path
        self.threshold = threshold
        self.classifier: Any = None

        # Preprocessing pipeline matching ImageNet standards
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # MobileNetV3 Small backbone on CPU in eval mode
        self.device = torch.device("cpu")
        try:
            backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        except Exception:
            # Fallback for older torchvision syntax
            backbone = models.mobilenet_v3_small(pretrained=True)

        backbone.classifier = torch.nn.Identity()
        backbone.eval()
        self.backbone = backbone.to(self.device)

        # Attempt to load trained classifier model if present
        if os.path.isfile(self.model_path):
            self.load_model(self.model_path)

    def load_model(self, model_path: str) -> None:
        """Load trained scikit-learn model pipeline from pickle file."""
        try:
            with open(model_path, "rb") as f:
                self.classifier = pickle.load(f)
            self.model_path = model_path
            logger.info(f"Loaded visual classifier model from '{model_path}'.")
        except Exception as exc:
            logger.warning(f"Failed to load visual classifier model from '{model_path}': {exc}")
            self.classifier = None

    def extract_features(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Extract 576-dimensional MobileNetV3 feature embedding for a BGR image crop.

        Args:
            crop_bgr: OpenCV BGR image crop (Height x Width x 3).

        Returns:
            1D numpy array of float32 features (shape: (576,)).
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return np.zeros(576, dtype=np.float32)

        # Convert BGR to RGB
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)

        tensor_img = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.backbone(tensor_img).squeeze(0).cpu().numpy()

        res = embedding.astype(np.float32)
        return np.asarray(res)

    def extract_batch_features(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        """Extract MobileNetV3 feature embeddings for a batch of BGR image crops.

        Args:
            crops_bgr: List of OpenCV BGR image crops.

        Returns:
            2D numpy array of float32 features (shape: (N, 576)).
        """
        if not crops_bgr:
            return np.zeros((0, 576), dtype=np.float32)

        tensors = []
        for crop in crops_bgr:
            if crop is None or crop.size == 0:
                # Dummy empty image fallback
                dummy = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
                tensors.append(self.transform(dummy))
            else:
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(crop_rgb)
                tensors.append(self.transform(pil_img))

        batch_tensor = torch.stack(tensors).to(self.device)

        with torch.no_grad():
            embeddings = self.backbone(batch_tensor).cpu().numpy()

        res = embeddings.astype(np.float32)
        return np.asarray(res)

    def predict_proba(self, crop_bgr: np.ndarray) -> float:
        """Predict probability that the input image crop is a True Positive entity.

        Args:
            crop_bgr: OpenCV BGR image crop.

        Returns:
            Float probability in [0.0, 1.0]. Returns 0.50 if model is not loaded.
        """
        if self.classifier is None:
            return 0.50

        embedding = self.extract_features(crop_bgr).reshape(1, -1)
        try:
            if hasattr(self.classifier, "predict_proba"):
                probas = self.classifier.predict_proba(embedding)
                return float(probas[0, 1])
            if hasattr(self.classifier, "decision_function"):
                decision = self.classifier.decision_function(embedding)
                # Sigmoid transform
                return float(1.0 / (1.0 + np.exp(-decision[0])))
            preds = self.classifier.predict(embedding)
            return float(preds[0])
        except Exception as exc:
            logger.warning(f"Error predicting candidate probability: {exc}")
            return 0.50

    def is_entity(self, crop_bgr: np.ndarray, threshold: float | None = None) -> bool:
        """Return True if candidate probability exceeds classification threshold.

        Args:
            crop_bgr: OpenCV BGR image crop.
            threshold: Optional custom decision threshold.

        Returns:
            Boolean prediction.
        """
        cutoff = threshold if threshold is not None else self.threshold
        proba = self.predict_proba(crop_bgr)
        return proba >= cutoff
