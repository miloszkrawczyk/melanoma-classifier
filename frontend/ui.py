"""The page a person actually uses.

Takes a Predictor rather than building one, so the interface and the JSON API
share a single copy of the model instead of holding 43 MB of weights twice.

Everything here is presentation. The three flags predict() returns decide what
gets shown, and this file only chooses the wording, the colours and the
animations. The layout is one long page: an animated intro, three cards on how
it works, then the analyser itself.

Two things are done by hand rather than with a Gradio component. The
seven-class breakdown is HTML instead of gr.Label, because the built-in one
cannot do hover tooltips and a bar named "bkl" means nothing to anyone who has
not read the dataset paper. And both languages are rendered into every page,
with CSS hiding whichever is not selected, so the switch in the corner is
instant and does not need the model to run again.
"""

import time
from pathlib import Path

import gradio as gr

from src.config import CONFIDENCE_THRESHOLD, MALIGNANT, MALIGNANT_THRESHOLD

EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"

# How long the progress bar takes at minimum. The prediction itself runs in
# well under a second, and a bar that flashes past is worse than none: nobody
# reads it and the page just blinks. These stages are real, the pauses only
# hold each one long enough to be read. Drop the numbers to make it quicker.
STAGES = (
    (0.15, "Wczytuję zdjęcie · Reading the photo", 0.9),
    (0.35, "Sprawdzam, czy to zmiana skórna · Checking it is a skin lesion", 1.1),
    (0.65, "Analizuję siecią neuronową · Running the network", 1.4),
    (0.90, "Rysuję mapę uwagi · Drawing the attention map", 1.2),
)

# name, malignant, description. Plain language on purpose: the point of the
# tooltip is that "vasc" stops being a four-letter mystery.
DISEASES = {
    "akiec": {
        "malignant": True,
        "en": (
            "Actinic keratosis",
            (
                "A rough, scaly patch caused by years of sun exposure. Counted as "
                "malignant here because it can turn into squamous cell carcinoma "
                "if left alone."
            ),
        ),
        "pl": (
            "Rogowacenie słoneczne",
            (
                "Szorstka, łuszcząca się plama po latach słońca. Liczona tu jako "
                "złośliwa, bo nieleczona może przejść w raka kolczystokomórkowego."
            ),
        ),
    },
    "bcc": {
        "malignant": True,
        "en": (
            "Basal cell carcinoma",
            (
                "The most common skin cancer. It grows slowly and almost never "
                "spreads, but it does keep growing and needs treatment."
            ),
        ),
        "pl": (
            "Rak podstawnokomórkowy",
            (
                "Najczęstszy nowotwór skóry. Rośnie powoli i prawie nigdy nie daje "
                "przerzutów, ale rośnie dalej i wymaga leczenia."
            ),
        ),
    },
    "bkl": {
        "malignant": False,
        "en": (
            "Benign keratosis",
            (
                "A harmless growth such as a seborrheic keratosis or a sun spot. "
                "Often looks alarming while being nothing to worry about."
            ),
        ),
        "pl": (
            "Rogowacenie łojotokowe",
            (
                "Niegroźna zmiana, na przykład brodawka łojotokowa albo plama "
                "słoneczna. Często wygląda groźnie, będąc niczym."
            ),
        ),
    },
    "df": {
        "malignant": False,
        "en": (
            "Dermatofibroma",
            (
                "A small, firm, benign nodule, often appearing after a minor "
                "injury or an insect bite."
            ),
        ),
        "pl": (
            "Włókniak twardy",
            (
                "Mały, twardy, łagodny guzek, często pojawiający się po drobnym "
                "skaleczeniu albo ukąszeniu owada."
            ),
        ),
    },
    "mel": {
        "malignant": True,
        "en": (
            "Melanoma",
            (
                "The dangerous one: cancer of the pigment cells. Very treatable "
                "when caught early, life-threatening when caught late."
            ),
        ),
        "pl": (
            "Czerniak",
            (
                "Ten groźny: nowotwór komórek barwnikowych. Bardzo dobrze "
                "wyleczalny wcześnie wykryty, śmiertelny wykryty późno."
            ),
        ),
    },
    "nv": {
        "malignant": False,
        "en": (
            "Melanocytic nevus",
            "An ordinary mole. Two thirds of the training data are these.",
        ),
        "pl": (
            "Znamię barwnikowe",
            "Zwykły pieprzyk. Dwie trzecie danych treningowych to właśnie one.",
        ),
    },
    "vasc": {
        "malignant": False,
        "en": (
            "Vascular lesion",
            (
                "A growth made of small blood vessels, such as an angioma. "
                "Typically harmless, with a distinctive red or purple colour."
            ),
        ),
        "pl": (
            "Zmiana naczyniowa",
            (
                "Zmiana zbudowana z drobnych naczyń krwionośnych, na przykład "
                "naczyniak. Zwykle niegroźna, o charakterystycznym czerwonym "
                "lub fioletowym kolorze."
            ),
        ),
    },
}


def both(en, pl, tag="span"):
    """Render a phrase in both languages, letting CSS show one of them."""
    return (
        f'<{tag} data-lang="en">{en}</{tag}>'
        f'<{tag} data-lang="pl">{pl}</{tag}>'
    )


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

/* --- language switch --- */
.langbar {
  position: fixed; top: 14px; right: 18px; z-index: 100;
  display: flex; gap: 2px; padding: 3px;
  background: rgba(30, 30, 30, 0.82); border: 1px solid #5c5c5c;
  border-radius: 999px; backdrop-filter: blur(6px);
}
.langbar button {
  border: 0; background: transparent; cursor: pointer;
  color: #b0b0b0; font-size: 0.76rem; font-weight: 600; letter-spacing: 0.06em;
  padding: 5px 12px; border-radius: 999px;
  transition: background 0.15s ease, color 0.15s ease;
}
.langbar button:hover { color: #e8e8e8; }
html:not([lang="pl"]) .langbar button[data-set="en"],
html[lang="pl"] .langbar button[data-set="pl"] {
  background: #0c3527; color: #7fbf9f;
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

/* --- language, last on purpose ---
   Both languages sit in the page and this hides the one not in use. Only the
   inactive language is touched, so the active one keeps whatever display its
   own rules give it; hiding everything and restoring one with
   "display: revert" does not survive a rule like ".hero .warning b".
   !important is doing real work here. Gradio rewrites the stylesheet, adding
   a copy of every rule prefixed with ".gradio-container... .contain", which
   lifts ".hero .warning b" from three points of specificity to six. My copy
   gets the same prefix and is then unusable, since it asks for an <html>
   inside the container. Without !important the scoped copy wins and both
   languages show at once.
   With no lang attribute at all the English shows, so the page is never blank
   while the script is still loading. */
html:not([lang="pl"]) [data-lang="pl"] { display: none !important; }
html[lang="pl"] [data-lang="en"] { display: none !important; }
"""

# Injected into <head> rather than passed as Gradio's js= parameter, which
# silently did not run. A plain script tag has no lifecycle to get caught in.
# The click handler is delegated to the document because Gradio renders and
# re-renders the buttons itself, so binding them directly would not survive.
HEAD = """
<script>
(function () {
  function set(lang) {
    document.documentElement.lang = lang;
    try { localStorage.setItem("lang", lang); } catch (e) {}
  }
  var saved = null;
  try { saved = localStorage.getItem("lang"); } catch (e) {}
  // The brackets matter: "a || b ? x : y" groups as "(a || b) ? x : y",
  // which would turn a saved choice of "en" into "pl".
  set(saved || ((navigator.language || "en").indexOf("pl") === 0 ? "pl" : "en"));

  document.addEventListener("click", function (event) {
    var button = event.target.closest(".langbar button");
    if (button) set(button.dataset.set);
  });

  // Gradio sets <html lang> from the browser on its own, after this runs and
  // again on re-render, which would quietly undo the visitor's choice.
  new MutationObserver(function () {
    var wanted = null;
    try { wanted = localStorage.getItem("lang"); } catch (e) {}
    if (wanted && document.documentElement.lang !== wanted) {
      document.documentElement.lang = wanted;
    }
  }).observe(document.documentElement, { attributes: true, attributeFilter: ["lang"] });
})();
</script>
"""

LANGBAR = """
<div class="langbar">
  <button data-set="pl" type="button">PL</button>
  <button data-set="en" type="button">EN</button>
</div>
"""

HERO = f"""
<section class="hero">
  {both("Skin Lesion Classifier", "Klasyfikator zmian skórnych", tag="h1")}
  {both(
      "A ResNet18 fine-tuned on 10,015 dermatoscope images. Upload a photo of "
      "a skin lesion and it names one of seven diagnoses, says whether the "
      "lesion should be treated as malignant, and shows exactly which part of "
      "the photo the answer rested on.",
      "ResNet18 dostrojony na 10 015 zdjęciach z dermatoskopu. Wgraj zdjęcie "
      "zmiany skórnej, a model poda jedną z siedmiu diagnoz, powie, czy "
      "traktować ją jako złośliwą, i pokaże, na którym fragmencie zdjęcia "
      "oparł odpowiedź.",
      tag="p",
  ).replace('<p data-lang', '<p class="sub" data-lang')}
  <div class="stack">PYTORCH · FASTAPI · DOCKER · GRADIO</div>
  <div class="warning">
    {both("This is not a medical tool.", "To nie jest narzędzie medyczne.", tag="b")}
    {both(
        "It is an educational machine-learning project, built to show how a "
        "model is trained, evaluated and served. It cannot diagnose anything, "
        "it is wrong often enough to matter, and nothing it says should decide "
        "whether you see a doctor. If something on your skin is new, changing "
        "or worrying you, that is a question for a dermatologist. Photos are "
        "analysed in memory and never stored.",
        "To edukacyjny projekt uczenia maszynowego, pokazujący, jak trenuje "
        "się, ocenia i wdraża model. Nie stawia diagnozy, myli się na tyle "
        "często, że ma to znaczenie, i nic, co powie, nie powinno decydować o "
        "wizycie u lekarza. Jeśli coś na Twojej skórze jest nowe, zmienia się "
        "albo niepokoi, to pytanie do dermatologa. Zdjęcia są analizowane w "
        "pamięci i nigdy nie są zapisywane.",
    )}
  </div>
  <a class="scroll" href="#analyser" title="↓">&#9660;</a>
</section>
"""

STEPS = f"""
<div class="steps">
  <div class="step">
    <div class="n">1</div>
    {both("A pretrained network, retargeted", "Gotowa sieć, przestawiona na nowe zadanie", tag="h3")}
    {both(
        "ResNet18 arrives knowing edges and textures from 1.28M ImageNet "
        "photos. Only its final layer is replaced and the whole thing "
        "fine-tuned on HAM10000.",
        "ResNet18 przychodzi, znając krawędzie i tekstury z 1,28 mln zdjęć z "
        "ImageNetu. Podmieniamy tylko ostatnią warstwę i dostrajamy całość na "
        "zbiorze HAM10000.",
        tag="p",
    )}
  </div>
  <div class="step">
    <div class="n">2</div>
    {both("A threshold, not the top class", "Próg zamiast najwyższej klasy", tag="h3")}
    {both(
        "The malignant call adds up the three malignant probabilities and "
        "flags anything above 15%. A photo scoring 30% melanoma and 29% "
        "carcinoma is a &quot;mole&quot; by the top-class rule, and clearly worth a "
        "look by this one. On held-out data it catches 92% of malignant "
        "lesions.",
        "Ocena złośliwości sumuje trzy prawdopodobieństwa klas złośliwych i "
        "oznacza wszystko powyżej 15%. Zdjęcie z wynikiem 30% czerniaka i 29% "
        "raka jest „pieprzykiem” według reguły maksimum, a wyraźnie "
        "podejrzane według tej. Na danych testowych wyłapuje 92% zmian "
        "złośliwych.",
        tag="p",
    )}
  </div>
  <div class="step">
    <div class="n">3</div>
    {both("Split by lesion, not by photo", "Podział po zmianach, nie po zdjęciach", tag="h3")}
    {both(
        "HAM10000 photographs some lesions more than once. Splitting the "
        "10,015 photos at random puts the same lesion on both sides of the "
        "fence, so the model gets tested on pictures it has effectively "
        "already seen. Splitting the 7,470 lesions instead costs a few points "
        "of accuracy and buys a number that actually means something.",
        "HAM10000 fotografuje część zmian skórnych wielokrotnie. Losowy "
        "podział 10 015 zdjęć wrzuca tę samą zmianę po obu stronach, więc "
        "model jest testowany na tym, co w praktyce już widział. Podział 7 470 "
        "zmian kosztuje kilka punktów trafności i daje liczbę, która coś "
        "znaczy.",
        tag="p",
    )}
  </div>
</div>
"""

HEATMAP_CAPTION = f"""
<p class="caption">
  {both(
      "<b>Reading the heatmap:</b> red marks the regions that pushed the "
      "answer, blue was ignored. It is blurry by nature, since the map has the "
      "resolution of the network's last layer (7×7), not of the photo. If the "
      "heat lands beside the lesion rather than on it, the diagnosis is not to "
      "be trusted.",
      "<b>Jak czytać mapę:</b> czerwony oznacza fragmenty, które zaważyły na "
      "odpowiedzi, niebieski został zignorowany. Jest rozmyta z natury, bo ma "
      "rozdzielczość ostatniej warstwy sieci (7×7), a nie zdjęcia. Jeśli "
      "ciepło leży obok zmiany, a nie na niej, diagnozie nie należy ufać.",
  )}
</p>
"""

PLACEHOLDER = f"""
<div class="verdict void">
  {both("No photo yet", "Jeszcze bez zdjęcia", tag="h2")}
  <p class="note">
    {both(
        "Upload a close-up of a skin lesion, or pick one of the examples below "
        "the upload box, then press Analyse.",
        "Wgraj zbliżenie zmiany skórnej albo wybierz jeden z przykładów pod "
        "polem wgrywania, a potem kliknij Analyse.",
    )}
  </p>
</div>
"""

NOT_SKIN = f"""
<div class="verdict void">
  {both("Cannot read this photo", "Nie potrafię odczytać tego zdjęcia", tag="h2")}
  <p>
    {both(
        "It does not look like a close-up of a skin lesion. Try a photo where "
        "the lesion fills most of the frame, in even light.",
        "To nie wygląda na zbliżenie zmiany skórnej. Spróbuj zdjęcia, na "
        "którym zmiana wypełnia większość kadru, w równym świetle.",
    )}
  </p>
  <p class="note">
    {both(
        "This check is a heuristic on colour and texture. It exists because "
        "the classifier itself will happily diagnose random noise with 86% "
        "confidence: a classifier can only ever answer with one of the classes "
        "it was taught.",
        "To sprawdzenie jest heurystyką opartą na kolorze i teksturze. Istnieje, "
        "bo sam klasyfikator bez wahania diagnozuje losowy szum z 86% "
        "pewnością: sieć potrafi odpowiedzieć wyłącznie jedną z klas, których "
        "ją nauczono.",
    )}
  </p>
</div>
"""


def verdict_html(result):
    """The headline card, worded from the three flags predict() hands back."""
    malignant = result["category"] == MALIGNANT
    style = "bad" if malignant else "good"

    heading = both(
        "&#9888; Suspicious" if malignant else "&#10003; Probably harmless",
        "&#9888; Podejrzana" if malignant else "&#10003; Prawdopodobnie niegroźna",
        tag="h2",
    )

    if result["confident"]:
        code = result["diagnosis"]
        line = both(
            f"<b>{DISEASES[code]['en'][0]}</b> ({code}), "
            f"{result['confidence']:.0%} confidence",
            f"<b>{DISEASES[code]['pl'][0]}</b> ({code}), "
            f"pewność {result['confidence']:.0%}",
            tag="p",
        )
    else:
        # Naming a diagnosis below the threshold would be a guess: the model is
        # right about half the time there. The malignant line still stands,
        # because it comes from three probabilities added together.
        line = both(
            f"Not confident enough to name the exact diagnosis "
            f"(best guess {result['confidence']:.0%}, "
            f"needs {CONFIDENCE_THRESHOLD:.0%}).",
            f"Za mało pewności, żeby nazwać dokładną diagnozę "
            f"(najlepsze przypuszczenie {result['confidence']:.0%}, "
            f"potrzeba {CONFIDENCE_THRESHOLD:.0%}).",
            tag="p",
        )

    chance = both(
        f"Probability the lesion is malignant: "
        f"<b>{result['malignant_probability']:.0%}</b> "
        f"(flagged above {MALIGNANT_THRESHOLD:.0%})",
        f"Prawdopodobieństwo złośliwości: "
        f"<b>{result['malignant_probability']:.0%}</b> "
        f"(oznaczane powyżej {MALIGNANT_THRESHOLD:.0%})",
    )

    advice = ""
    if malignant:
        advice = both(
            "<b>Worth showing to a dermatologist.</b>",
            "<b>Warto pokazać dermatologowi.</b>",
            tag="p",
        )

    return f"""
<div class="verdict {style}">
  {heading}
  {line}
  <p class="note">{chance}</p>
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
        disease = DISEASES[code]
        malignant = disease["malignant"]
        colour = "#d64545" if malignant else "#1a6b4f"

        tag = (
            '<span class="tag mal">' + both("malignant", "złośliwa") + "</span>"
            if malignant
            else '<span class="tag ben">' + both("benign", "łagodna") + "</span>"
        )
        name = both(disease["en"][0], disease["pl"][0])
        blurb = both(disease["en"][1], disease["pl"][1])

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

    legend = '<p class="caption">' + both(
        "Hover a row for what the diagnosis actually is. Red classes count as "
        "malignant, green as benign.",
        "Najedź na wiersz, żeby zobaczyć, czym jest dana diagnoza. Czerwone "
        "klasy liczą się jako złośliwe, zielone jako łagodne.",
    ) + "</p>"

    return f'<div class="diag">{"".join(rows)}</div>{legend}'


def build_ui(predictor):
    """Assemble the page around an already-loaded model."""

    def analyse(image, progress=gr.Progress()):
        if image is None:
            return PLACEHOLDER, "", gr.update(value=None, visible=False)

        for fraction, label, pause in STAGES:
            progress(fraction, desc=label)
            time.sleep(pause)

        result = predictor.predict(image)

        if not result["looks_like_skin"]:
            return NOT_SKIN, "", gr.update(value=None, visible=False)

        return (
            verdict_html(result),
            bars_html(result["probabilities"]),
            gr.update(value=predictor.explain(image), visible=True),
        )

    with gr.Blocks(title="Skin Lesion Classifier") as page:
        gr.HTML(LANGBAR)
        gr.HTML(HERO)
        gr.HTML(STEPS)

        with gr.Column(elem_id="analyser"), gr.Row():
            with gr.Column(scale=5):
                gr.HTML(
                    '<p class="caption">'
                    + both("Photo of the lesion", "Zdjęcie zmiany skórnej")
                    + "</p>"
                )
                photo = gr.Image(type="pil", show_label=False)
                button = gr.Button("Analyse", variant="primary", size="lg")
                gr.Examples(
                    examples=[
                        [str(path)] for path in sorted(EXAMPLES_DIR.glob("*.jpg"))
                    ],
                    inputs=photo,
                    label=" ",
                )
                gr.HTML(
                    '<p class="caption">'
                    + both(
                        "No photo of your own? Pick one of the examples "
                        "above. Photos are analysed in memory and not "
                        "stored. This is an educational demo trained on "
                        "dermatoscope images, so phone photos will do "
                        "worse.",
                        "Nie masz własnego zdjęcia? Wybierz jeden z "
                        "przykładów powyżej. Zdjęcia są analizowane w "
                        "pamięci i nie są zapisywane. To demonstracja "
                        "edukacyjna, trenowana na zdjęciach z "
                        "dermatoskopu, więc na zdjęciach z telefonu "
                        "wypadnie gorzej.",
                    )
                    + "</p>"
                )

            with gr.Column(scale=6):
                answer = gr.HTML(PLACEHOLDER)
                scores = gr.HTML("")
                gr.HTML(
                    '<p class="caption">'
                    + both("What the model looked at", "Na co model patrzył")
                    + "</p>"
                )
                heatmap = gr.Image(show_label=False, visible=False)
                gr.HTML(HEATMAP_CAPTION)

        gr.HTML(
            '<p class="smallprint">Miłosz Krawczyk · '
            '<a href="https://github.com/miloszkrawczyk/melanoma-classifier">'
            + both("source on GitHub", "kod na GitHubie")
            + "</a> · "
            '<a href="https://huggingface.co/miloszkrawczyk/skin-lesion-classifier">'
            + both("model card", "karta modelu")
            + "</a> · "
            + both("data", "dane")
            + ": Tschandl et al., <i>HAM10000</i>, Sci. Data 5, 180161 (2018), CC BY-NC"
            "</p>"
        )

        button.click(analyse, inputs=photo, outputs=[answer, scores, heatmap])

    return page


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
