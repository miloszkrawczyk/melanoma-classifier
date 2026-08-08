"""The page a person actually uses.

Takes a Predictor rather than building one, so the interface and the JSON API
share a single copy of the model instead of holding 43 MB of weights twice.

Everything here is presentation. The three flags predict() returns decide what
gets shown, and this file only chooses the wording, the colours and the
animations. The layout is one long page: an animated intro, three cards on how
it works, then the analyser itself.

The seven-class breakdown is drawn by hand as HTML instead of gr.Label,
because the built-in component cannot do hover tooltips, and a bar named
"bkl" means nothing to anyone who has not read the dataset paper.
"""

from pathlib import Path

import gradio as gr

from src.config import CONFIDENCE_THRESHOLD, MALIGNANT, MALIGNANT_THRESHOLD

EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"

# What the tooltip says about each class. Deliberately plain language: the
# point of the bubble is that "vasc" stops being a four-letter mystery.
DISEASES = {
    "akiec": (
        "Actinic keratosis",
        True,
        (
            "A rough, scaly patch caused by years of sun exposure. Counted as "
            "malignant here because it can progress into squamous cell "
            "carcinoma if left alone."
        ),
    ),
    "bcc": (
        "Basal cell carcinoma",
        True,
        (
            "The most common skin cancer. It grows slowly and almost never "
            "spreads, but it does keep growing and needs treatment."
        ),
    ),
    "bkl": (
        "Benign keratosis",
        False,
        (
            "A harmless growth such as a seborrheic keratosis or a sun spot. "
            "Often looks alarming while being nothing to worry about."
        ),
    ),
    "df": (
        "Dermatofibroma",
        False,
        (
            "A small, firm, benign nodule, often appearing after a minor "
            "injury or an insect bite."
        ),
    ),
    "mel": (
        "Melanoma",
        True,
        (
            "The dangerous one: cancer of the pigment cells. Very treatable "
            "when caught early, life-threatening when caught late."
        ),
    ),
    "nv": (
        "Melanocytic nevus",
        False,
        "An ordinary mole. Two thirds of the training data are these.",
    ),
    "vasc": (
        "Vascular lesion",
        False,
        (
            "A growth made of small blood vessels, such as an angioma. "
            "Typically harmless, with a distinctive red or purple colour."
        ),
    ),
}

# The custom pieces are plain HTML, so the styling lives here rather than in a
# theme. Gradio's CSS variables are used wherever the page has to survive both
# the light and the dark mode; the hero keeps its own fixed palette on
# purpose: graphite grey fading into bottle green, with red held back for the
# one thing that has to alarm anybody.
CSS = """
html { scroll-behavior: smooth; }
.gradio-container { max-width: 1100px !important; margin: 0 auto !important; }
footer { display: none !important; }

@keyframes fade-up {
  from { opacity: 0; transform: translateY(18px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes bob {
  0%, 100% { transform: translateY(0); }
  50%      { transform: translateY(7px); }
}

/* --- intro --- */
.hero {
  min-height: 88vh;
  display: flex; flex-direction: column;
  justify-content: center; align-items: center;
  text-align: center;
  border-radius: 16px;
  padding: 48px 24px;
  background: linear-gradient(155deg, #424242 0%, #24463a 55%, #0c3527 100%);
  color: #eef1f0;
}
.hero > * { animation: fade-up 0.8s ease-out both; }
.hero h1 {
  animation-delay: 0.25s;
  font-size: 3.2rem; font-weight: 700; margin: 0.4em 0;
  color: #ffffff;
}
.hero .sub {
  animation-delay: 0.45s;
  max-width: 640px; font-size: 1.15rem; line-height: 1.7; color: #ccd4d0;
}
.hero .stack {
  animation-delay: 0.6s;
  margin-top: 18px; font-size: 0.82rem; letter-spacing: 0.08em;
  color: rgba(255, 255, 255, 0.45);
}
.hero .warning {
  animation-delay: 0.75s;
  margin-top: 30px; max-width: 620px;
  padding: 16px 22px; border-radius: 10px;
  border: 1px solid #c47b3f; border-left-width: 4px;
  background: rgba(0, 0, 0, 0.28);
  font-size: 0.92rem; line-height: 1.6; color: #f0d9bb; text-align: left;
}
.hero .warning b { color: #ffb066; display: block; margin-bottom: 4px;
  font-size: 1.02rem; letter-spacing: 0.02em; }
.hero .scroll {
  animation: fade-up 0.8s ease-out 0.9s both, bob 1.6s ease-in-out 1.8s infinite;
  margin-top: 44px; font-size: 1.6rem; color: #7fbf9f; text-decoration: none;
}

/* --- how it works --- */
.steps { display: flex; gap: 16px; margin: 28px 0; flex-wrap: wrap; }
.step {
  flex: 1; min-width: 240px;
  border: 1px solid var(--border-color-primary);
  background: var(--background-fill-secondary);
  border-radius: 12px; padding: 20px 22px;
  animation: fade-up 0.7s ease-out both;
}
.step:nth-child(2) { animation-delay: 0.15s; }
.step:nth-child(3) { animation-delay: 0.3s; }
.step .n {
  display: inline-block; width: 28px; height: 28px; line-height: 28px;
  text-align: center; border-radius: 50%;
  background: #0c3527; color: #7fbf9f; font-weight: 700; margin-bottom: 10px;
}
.step h3 { margin: 0 0 6px 0; font-size: 1.02rem; }
.step p { margin: 0; font-size: 0.9rem; line-height: 1.55; color: var(--body-text-color-subdued); }

/* --- verdict card --- */
.verdict {
  border-radius: 12px; padding: 20px 24px;
  border: 1px solid var(--border-color-primary);
  border-left-width: 5px;
  background: var(--background-fill-secondary);
  animation: fade-up 0.5s ease-out both;
}
.verdict h2 { margin: 0 0 10px 0; font-size: 1.35rem; }
.verdict p { margin: 6px 0; line-height: 1.55; }
.verdict .note { color: var(--body-text-color-subdued); font-size: 0.9rem; }
.verdict.bad  { border-left-color: #d64545; }
.verdict.bad h2 { color: #d64545; }
.verdict.good { border-left-color: #1a6b4f; }
.verdict.good h2 { color: #2f8f6c; }
.verdict.void { border-left-color: var(--border-color-primary); }

.meter {
  height: 8px; border-radius: 4px; margin-top: 6px;
  background: var(--border-color-primary); overflow: hidden;
}
.meter > div { height: 100%; border-radius: 4px; background: #d64545; }

/* --- the seven bars --- */
.diag { margin-top: 14px; }
.diag-row {
  position: relative;
  display: grid; grid-template-columns: 210px 1fr 52px;
  gap: 12px; align-items: center;
  padding: 7px 10px; border-radius: 8px;
  transition: background 0.15s ease;
}
.diag-row:hover { background: var(--background-fill-secondary); }
.diag-name { font-size: 0.88rem; display: flex; align-items: center; gap: 8px; }
.diag-name .dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.diag-name .code { color: var(--body-text-color-subdued); font-family: var(--font-mono); font-size: 0.78rem; }
.diag-track { height: 10px; border-radius: 5px; background: var(--border-color-primary); overflow: hidden; }
.diag-fill { height: 100%; border-radius: 5px; transition: width 0.6s ease; }
.diag-pct { font-size: 0.85rem; text-align: right; font-variant-numeric: tabular-nums; }

.diag-tip {
  position: absolute; bottom: calc(100% + 6px); left: 12px; z-index: 30;
  width: 340px; max-width: 80vw;
  padding: 12px 14px; border-radius: 10px;
  background: #424242; color: #e6ebe8;
  font-size: 0.85rem; line-height: 1.5;
  opacity: 0; visibility: hidden; transform: translateY(4px);
  transition: opacity 0.18s ease, transform 0.18s ease;
  pointer-events: none;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
}
.diag-row:hover .diag-tip { opacity: 1; visibility: visible; transform: translateY(0); }
.diag-tip b { color: #7fbf9f; }
.diag-tip .tag {
  float: right; font-size: 0.72rem; padding: 2px 8px; border-radius: 999px;
}
.diag-tip .tag.mal { background: #55222a; color: #ff9d9d; }
.diag-tip .tag.ben { background: #123d2f; color: #7fbf9f; }

.caption { font-size: 0.85rem; color: var(--body-text-color-subdued); line-height: 1.5; margin-top: 6px; }
.smallprint { text-align: center; font-size: 0.82rem; color: var(--body-text-color-subdued); margin: 30px 0 10px 0; }
.smallprint a { color: #2f8f6c; }
"""

HERO = """
<section class="hero">
  <h1>Skin Lesion Classifier</h1>
  <p class="sub">A ResNet18 fine-tuned on 10,015 dermatoscope images. Upload a
  photo of a skin lesion and it names one of seven diagnoses, says whether the
  lesion should be treated as malignant, and shows exactly which part of the
  photo the answer rested on.</p>
  <div class="stack">PYTORCH · FASTAPI · DOCKER · GRADIO</div>
  <div class="warning">
    <b>This is not a medical tool.</b>
    It is an educational machine-learning project, built to show how a model is
    trained, evaluated and served. It cannot diagnose anything, it is wrong
    often enough to matter, and nothing it says should decide whether you see a
    doctor. If something on your skin is new, changing or worrying you, that is
    a question for a dermatologist. Photos are analysed in memory and never
    stored.
  </div>
  <a class="scroll" href="#analyser" title="See it work">&#9660;</a>
</section>
"""

STEPS = """
<div class="steps">
  <div class="step">
    <div class="n">1</div>
    <h3>A pretrained network, retargeted</h3>
    <p>ResNet18 arrives knowing edges and textures from 1.28M ImageNet photos.
    Only its final layer is replaced and the whole thing fine-tuned on
    HAM10000, split by lesion so no lesion sits in both training and test.</p>
  </div>
  <div class="step">
    <div class="n">2</div>
    <h3>A threshold, not the top class</h3>
    <p>The malignant call adds up the three malignant probabilities and flags
    anything above 15%. A photo scoring 30% melanoma and 29% carcinoma is a
    "mole" by the top-class rule — and clearly worth a look by this one. On
    held-out data it catches 92% of malignant lesions.</p>
  </div>
  <div class="step">
    <div class="n">3</div>
    <h3>Split by lesion, not by photo</h3>
    <p>HAM10000 photographs some lesions more than once. Splitting the 10,015
    photos at random puts the same lesion on both sides of the fence, so the
    model gets tested on pictures it has effectively already seen. Splitting
    the 7,470 lesions instead costs a few points of accuracy and buys a number
    that actually means something.</p>
  </div>
</div>
"""

HEATMAP_CAPTION = """
<p class="caption"><b>Reading the heatmap:</b> red marks the regions that
pushed the answer, blue was ignored. It is blurry by nature — the map has the
resolution of the network's last layer (7×7), not of the photo. If the heat
lands beside the lesion rather than on it, the diagnosis is not to be
trusted.</p>
"""

PLACEHOLDER = """
<div class="verdict void">
  <h2>No photo yet</h2>
  <p class="note">Upload a close-up of a skin lesion, or pick one of the
  examples below the upload box, then press Analyse.</p>
</div>
"""

NOT_SKIN = """
<div class="verdict void">
  <h2>Cannot read this photo</h2>
  <p>It does not look like a close-up of a skin lesion. Try a photo where the
  lesion fills most of the frame, in even light.</p>
  <p class="note">This check is a heuristic on colour and texture. It exists
  because the classifier itself will happily diagnose random noise with 86%
  confidence — a classifier can only ever answer with one of the classes it
  was taught.</p>
</div>
"""


def verdict_html(result):
    """The headline card, worded from the three flags predict() hands back."""
    malignant = result["category"] == MALIGNANT
    style = "bad" if malignant else "good"
    heading = "&#9888; Suspicious" if malignant else "&#10003; Probably harmless"

    if result["confident"]:
        diagnosis_line = (
            f"<p><b>{DISEASES[result['diagnosis']][0]}</b> "
            f"({result['diagnosis']}), {result['confidence']:.0%} confidence</p>"
        )
    else:
        # Naming a diagnosis below the threshold would be a guess: the model
        # is right about half the time there. The malignant line still stands,
        # because it comes from three probabilities added together.
        diagnosis_line = (
            f"<p>Not confident enough to name the exact diagnosis "
            f"(best guess {result['confidence']:.0%}, "
            f"needs {CONFIDENCE_THRESHOLD:.0%}).</p>"
        )

    advice = (
        "<p><b>Worth showing to a dermatologist.</b></p>" if malignant else ""
    )

    return f"""
<div class="verdict {style}">
  <h2>{heading}</h2>
  {diagnosis_line}
  <p class="note">Probability the lesion is malignant:
  <b>{result["malignant_probability"]:.0%}</b>
  (flagged above {MALIGNANT_THRESHOLD:.0%})</p>
  <div class="meter"><div style="width:{result["malignant_probability"]:.1%}"></div></div>
  {advice}
</div>
"""


def bars_html(probabilities):
    """All seven probabilities as bars, each with a hover tooltip.

    Hand-rolled instead of gr.Label so hovering a row can highlight it and pop
    a bubble with the full disease name and a two-line description.
    """
    rows = []
    for code, value in sorted(probabilities.items(), key=lambda kv: -kv[1]):
        name, malignant, blurb = DISEASES[code]
        colour = "#d64545" if malignant else "#1a6b4f"
        tag = (
            '<span class="tag mal">malignant</span>'
            if malignant
            else '<span class="tag ben">benign</span>'
        )
        rows.append(f"""
<div class="diag-row">
  <div class="diag-name">
    <span class="dot" style="background:{colour}"></span>
    <span>{name}</span> <span class="code">{code}</span>
  </div>
  <div class="diag-track"><div class="diag-fill" style="width:{value:.1%}; background:{colour}"></div></div>
  <div class="diag-pct">{value:.1%}</div>
  <div class="diag-tip">{tag}<b>{name}</b> ({code})<br>{blurb}</div>
</div>""")

    legend = (
        '<p class="caption">Hover a row for what the diagnosis actually is. '
        "Red classes count as malignant, green as benign.</p>"
    )
    return f'<div class="diag">{"".join(rows)}</div>{legend}'


# Gradio 6 moved theme and css out of Blocks() and into mount_gradio_app()
# and launch(). Exported here so api/main.py can hand them over at mount time;
# passing them to Blocks() is silently ignored.
THEME = gr.themes.Soft(primary_hue="emerald", neutral_hue="gray").set(
    # Gradio's "gray" neutrals are tinted blue, which took over the whole page
    # and fought the bottle green. These are flat greys built around #424242.
    body_background_fill="#333333",
    body_background_fill_dark="#333333",
    background_fill_primary="#424242",
    background_fill_primary_dark="#424242",
    background_fill_secondary="#4d4d4d",
    background_fill_secondary_dark="#4d4d4d",
    block_background_fill="#424242",
    block_background_fill_dark="#424242",
    border_color_primary="#5c5c5c",
    border_color_primary_dark="#5c5c5c",
    block_border_color="#5c5c5c",
    block_border_color_dark="#5c5c5c",
    input_background_fill="#3a3a3a",
    input_background_fill_dark="#3a3a3a",
    body_text_color="#e8e8e8",
    body_text_color_dark="#e8e8e8",
    body_text_color_subdued="#b0b0b0",
    body_text_color_subdued_dark="#b0b0b0",
    block_label_text_color="#e8e8e8",
    block_label_text_color_dark="#e8e8e8",
)


def build_ui(predictor):
    """Assemble the page around an already-loaded model."""

    def analyse(image):
        if image is None:
            return PLACEHOLDER, "", gr.update(value=None, visible=False)

        result = predictor.predict(image)

        if not result["looks_like_skin"]:
            return NOT_SKIN, "", gr.update(value=None, visible=False)

        return (
            verdict_html(result),
            bars_html(result["probabilities"]),
            gr.update(value=predictor.explain(image), visible=True),
        )

    with gr.Blocks(title="Skin Lesion Classifier") as page:
        gr.HTML(HERO)
        gr.HTML(STEPS)

        with gr.Column(elem_id="analyser"), gr.Row():
            with gr.Column(scale=5):
                photo = gr.Image(type="pil", label="Photo of the lesion")
                button = gr.Button("Analyse", variant="primary", size="lg")
                gr.Examples(
                    examples=[
                        [str(path)] for path in sorted(EXAMPLES_DIR.glob("*.jpg"))
                    ],
                    inputs=photo,
                    label="No photo of your own? Pick one of these",
                )
                gr.HTML(
                    '<p class="caption">Photos are analysed in memory and '
                    "not stored. This is an educational demo trained on "
                    "dermatoscope images — phone photos will do worse. "
                    "Anything on your skin that worries you is a question "
                    "for a dermatologist.</p>"
                )

            with gr.Column(scale=6):
                answer = gr.HTML(PLACEHOLDER)
                scores = gr.HTML("")
                heatmap = gr.Image(
                    label="What the model looked at", visible=False
                )
                gr.HTML(HEATMAP_CAPTION)

        gr.HTML(
            '<p class="smallprint">'
            "Built by Miłosz Krawczyk · "
            '<a href="https://github.com/miloszkrawczyk/melanoma-classifier">source on GitHub</a>'
            " · "
            '<a href="https://huggingface.co/miloszkrawczyk/skin-lesion-classifier">model card</a>'
            " · data: Tschandl et al., <i>HAM10000</i>, Sci. Data 5, 180161 (2018), CC BY-NC"
            "</p>"
        )

        button.click(analyse, inputs=photo, outputs=[answer, scores, heatmap])

    return page
