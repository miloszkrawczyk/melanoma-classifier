"""The web service: send it a photo, get a diagnosis back.

    uv run uvicorn api.main:app --port 8000

Then open http://127.0.0.1:8000/docs to try it from a browser.

Nothing here decides anything. It unpacks an upload, hands it to Predictor and
turns the answer into JSON. That separation is deliberate: predict.py knows
nothing about HTTP, so it can be called from a test, a script or a notebook
just as easily.
"""

from fastapi import FastAPI, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from src.predict import Predictor

app = FastAPI(
    title="Melanoma classifier",
    description=(
        "Educational project, not a diagnostic tool. Trained on HAM10000. "
        "See a dermatologist about anything on your skin."
    ),
)

# Built once, when the server starts, and reused for every request. Loading
# 43 MB of weights takes a second or two, which would be a second or two added
# to every single request if it happened inside the handler.
predictor = Predictor()


@app.get("/health")
def health():
    """Says the service is up.

    Looks pointless and is not. Hosting platforms poll an address like this to
    decide whether the app finished starting and whether it needs a restart.
    Without one they have to guess from whether the port is open, which stays
    true even when the app is broken.
    """
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile):
    """Take one photo of a skin lesion and answer three questions about it.

    Whether the picture is a skin lesion at all, which diagnosis it looks like,
    and whether to treat it as malignant. Those three are separate on purpose:
    the model can be sure a lesion is malignant while having no idea which of
    the three malignant diagnoses it is looking at.
    """
    try:
        image = Image.open(file.file)
    except UnidentifiedImageError:
        # 400 means "your request was wrong", as against 500 which would mean
        # "our code broke". Sending a text file here is the caller's mistake.
        raise HTTPException(status_code=400, detail="that file is not an image")

    return predictor.predict(image)
