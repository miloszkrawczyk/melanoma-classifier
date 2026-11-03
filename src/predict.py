"""Turns one photo into an answer: which diagnosis, how sure, and whether to
treat it as malignant.

This is the piece the API will call. Nothing here trains or measures anything;
it loads a trained model once and then answers questions about single images.

Loading the weights takes a second or two, so the model is held in an object
rather than read from disk on every call. A web server builds one Predictor
at startup and reuses it for every request.
"""

import torch
from torchvision import transforms

from src.config import (
    BENIGN,
    CLASS_NAMES,
    CLASSES,
    CONFIDENCE_THRESHOLD,
    IMAGE_SIZE,
    MALIGNANT,
    MALIGNANT_CLASSES,
    MALIGNANT_THRESHOLD,
    MODELS,
    SKIN_MAX_ROUGHNESS,
    SKIN_MIN_RED,
    SKIN_MIN_ROUGHNESS,
    SKIN_MIN_SATURATION,
    SKIN_RED_TOLERANCE,
)
from src.device import pick_device
from src.gradcam import heatmap, last_conv_layer, overlay
from src.models.resnet import build_resnet18

MALIGNANT_INDEXES = [CLASSES.index(dx) for dx in MALIGNANT_CLASSES]


def roughness(tensor):
    """Average brightness change between neighbouring pixels.

    Noise jumps around from pixel to pixel and lands near 0.22. A flat colour
    never changes and lands at 0.0. Photographs of skin sit between the two.
    """
    grey = tensor.mean(dim=0)
    across = (grey[:, 1:] - grey[:, :-1]).abs().mean()
    down = (grey[1:, :] - grey[:-1, :]).abs().mean()
    return float((across + down) / 2)


def looks_like_skin(tensor):
    """A rough guard against photos that are not skin lesions at all.

    Three cheap checks: the picture has to have the texture of a photograph,
    it has to carry colour rather than being grey, and it has to be
    red-dominant the way skin close-ups are.

    This is a heuristic, not a classifier. It turns away noise, flat colours,
    screenshots and diagrams. It will happily let through a photograph of a
    beige wall, and it is not meant to be the last line of defence.
    """
    texture = SKIN_MIN_ROUGHNESS <= roughness(tensor) <= SKIN_MAX_ROUGHNESS

    # float() on every measurement, not just at the end. Comparing tensors
    # gives back tensors, and "and" hands back whichever operand it stopped
    # on, so leaving them as tensors makes this return a tensor whenever the
    # image passes. JSON has no way to write that down.
    saturation = float((tensor.max(dim=0).values - tensor.min(dim=0).values).mean())
    red, green, blue = (float(value) for value in tensor.mean(dim=[1, 2]))

    red_dominant = (
        red >= green - SKIN_RED_TOLERANCE and red >= blue - SKIN_RED_TOLERANCE
    )

    return (
        texture
        and saturation >= SKIN_MIN_SATURATION
        and red_dominant
        and red >= SKIN_MIN_RED
    )


class Predictor:
    """The trained ResNet, loaded once, answering one photo at a time.

    Deliberately hard-wired to the ResNet rather than taking a model name. The
    CNN exists as a baseline to compare against in the README, and importing it
    here would drag the training code and pandas into a web service that has no
    use for either.
    """

    def __init__(self):
        settings = MODELS["resnet"]

        self.device = pick_device()
        # pretrained=False: the fine-tuned weights land on top of this a
        # line later, so downloading ImageNet's first would be wasted.
        self.model = build_resnet18(pretrained=False).to(self.device)
        # weights_only=True because a .pth is a pickle and unpickling runs
        # whatever the file says to run. That is not hypothetical here: the
        # README tells people to download resnet.pth from the Hugging Face Hub,
        # so this reads a file that came off the internet. The keyword makes
        # torch accept tensors and refuse everything else.
        self.model.load_state_dict(
            torch.load(settings["weights"], map_location=self.device, weights_only=True)
        )
        # Without eval() the BatchNorm layers would read statistics off the
        # current batch. A batch of one photo would then depend on itself, and
        # the same picture would score differently every time.
        self.model.eval()

        # Two steps rather than one Compose, because looks_like_skin needs the
        # picture before normalisation, while it still holds plain 0-1 pixels.
        self.to_tensor = transforms.Compose(
            [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()]
        )
        # The same normalisation the model was trained with. Anything else
        # quietly costs accuracy.
        self.normalise = transforms.Normalize(
            mean=settings["mean"], std=settings["std"]
        )

        # Only the layer is kept. The hook that reads it goes on and comes off
        # inside each call, which is what keeps two requests from reading each
        # other's activations. Leaving a hook registered for the life of the
        # process is what made that possible before.
        self.layer = last_conv_layer(self.model, "resnet")

    def explain(self, image):
        """The photo with a heatmap of where the answer came from painted on.

        Two passes through the network, not one: the first to find out which
        diagnosis won, the second to ask what that particular diagnosis rested
        on. A heatmap is always about one class, so there is nothing to draw
        until the class is known.
        """
        image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        batch = self.normalise(self.to_tensor(image)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            top = int(self.model(batch).argmax())

        # No no_grad() around this one. Grad-CAM is built out of gradients, so
        # switching them off would leave it nothing to work with.
        cam = heatmap(self.model, self.layer, batch, top)

        return overlay(image, cam)

    def predict(self, image):
        """Answer for one PIL image. Returns a dict ready to be sent as JSON."""
        pixels = self.to_tensor(image.convert("RGB"))

        # unsqueeze(0) turns (3, 224, 224) into (1, 3, 224, 224). The network
        # always expects a batch, even when the batch holds a single photo.
        batch = self.normalise(pixels).unsqueeze(0).to(self.device)

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
            # False means the picture does not look like a photo of skin
            # at all, so nothing below it should be shown as an answer.
            "looks_like_skin": looks_like_skin(pixels),
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
