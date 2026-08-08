"""The page a person actually uses.

Takes a Predictor rather than building one, so the interface and the JSON API
share a single copy of the model instead of holding 43 MB of weights twice.

Everything here is presentation. The three flags predict() returns decide what
gets shown, and this file only chooses the wording and the colours.
"""

import gradio as gr

from src.config import CONFIDENCE_THRESHOLD, MALIGNANT, MALIGNANT_THRESHOLD

DISCLAIMER = """
### Educational project, not a diagnostic tool

Trained on HAM10000, a set of dermatoscope images. Phone photos look different
enough that it will do worse on them. It misses roughly 6% of malignant
lesions on data it has never seen, and flags plenty of harmless moles.

**Anything on your skin that is new, changing or bothering you is a question
for a dermatologist, not for this page.** Uploaded photos are not stored.
"""

NOT_SKIN = """
## Cannot read this photo

It does not look like a close-up of a skin lesion. Try a photo where the
lesion fills most of the frame, in even light.
"""


def verdict(result):
    """The headline, worded from the three flags predict() hands back."""
    malignant = result["category"] == MALIGNANT
    mark = "!️" if malignant else "OK"
    heading = "Suspicious" if malignant else "Probably harmless"

    lines = [f"## {mark} {heading}"]

    if result["confident"]:
        lines.append(f"**{result['name']}**, {result['confidence']:.0%} confidence")
    else:
        # Naming a diagnosis here would be a guess: below the threshold the
        # model is right about half the time. The line above still holds,
        # because it comes from three probabilities added together rather
        # than from one.
        lines.append(
            f"Not confident enough to name the diagnosis "
            f"(best guess {result['confidence']:.0%}, "
            f"needs {CONFIDENCE_THRESHOLD:.0%})."
        )

    lines.append(
        f"Probability the lesion is malignant: "
        f"**{result['malignant_probability']:.0%}** "
        f"(flagged above {MALIGNANT_THRESHOLD:.0%})"
    )

    if malignant:
        lines.append("\nWorth showing to a dermatologist.")

    return "\n\n".join(lines)


def build_ui(predictor):
    """Assemble the page around an already-loaded model."""

    def analyse(image):
        if image is None:
            return "Upload a photo to start.", {}, None

        result = predictor.predict(image)

        if not result["looks_like_skin"]:
            return NOT_SKIN, {}, None

        return verdict(result), result["probabilities"], predictor.explain(image)

    with gr.Blocks(title="Melanoma classifier", theme=gr.themes.Soft()) as page:
        gr.Markdown(DISCLAIMER)

        with gr.Row():
            with gr.Column():
                photo = gr.Image(type="pil", label="Photo of the lesion")
                button = gr.Button("Analyse", variant="primary")

            with gr.Column():
                answer = gr.Markdown("Upload a photo to start.")
                # gr.Label draws a bar per class, so a split answer is visible
                # rather than hidden behind a single winner.
                scores = gr.Label(label="All seven diagnoses", num_top_classes=7)
                heatmap = gr.Image(label="What the model looked at")

        button.click(analyse, inputs=photo, outputs=[answer, scores, heatmap])

    return page
