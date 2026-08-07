"""Turns one photo into an answer: which diagnosis, how sure, and whether to
treat it as malignant.

This is the piece the API will call. Nothing here trains or measures anything;
it loads a trained model once and then answers questions about single images.

Loading the weights takes a second or two, so the model is held in an object
rather than read from disk on every call. A web server builds one Predictor
at startup and reuses it for every request.
"""

import torch

from src.config import (
    BENIGN,
    CLASS_NAMES,
    CLASSES,
    CONFIDENCE_THRESHOLD,
    MALIGNANT,
    MALIGNANT_CLASSES,
    MALIGNANT_THRESHOLD,
    MODELS,
)
from src.data import build_transform
from src.train import build_model, pick_device

MALIGNANT_INDEXES = [CLASSES.index(dx) for dx in MALIGNANT_CLASSES]


class Predictor:
    """A trained model, loaded once, answering one photo at a time."""

    def __init__(self, name="resnet"):
        settings = MODELS[name]

        self.device = pick_device()
        self.model = build_model(name).to(self.device)
        self.model.load_state_dict(
            torch.load(settings["weights"], map_location=self.device)
        )
        # Without eval() the BatchNorm layers would read statistics off the
        # current batch. A batch of one photo would then depend on itself, and
        # the same picture would score differently every time.
        self.model.eval()

        # No augmentation, and the same normalisation the model was trained
        # with. Anything else quietly costs accuracy.
        self.transform = build_transform(
            augment=False, mean=settings["mean"], std=settings["std"]
        )

    def predict(self, image):
        """Answer for one PIL image. Returns a dict ready to be sent as JSON."""
        # unsqueeze(0) turns (3, 224, 224) into (1, 3, 224, 224). The network
        # always expects a batch, even when the batch holds a single photo.
        batch = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(batch)
            probabilities = torch.softmax(logits, dim=1)[0].cpu()

        top = int(probabilities.argmax())
        confidence = float(probabilities[top])
        malignant_probability = float(probabilities[MALIGNANT_INDEXES].sum())

        return {
            "diagnosis": CLASSES[top],
            "name": CLASS_NAMES[CLASSES[top]],
            "confidence": confidence,
            # False means the named diagnosis is a coin toss and should not be
            # shown as an answer. The category below is unaffected: the model
            # can be sure a lesion is malignant while unsure which of the three
            # malignant diagnoses it is.
            "confident": confidence >= CONFIDENCE_THRESHOLD,
            "malignant_probability": malignant_probability,
            "category": (
                MALIGNANT
                if malignant_probability >= MALIGNANT_THRESHOLD
                else BENIGN
            ),
            "probabilities": {
                name: float(value) for name, value in zip(CLASSES, probabilities)
            },
        }
