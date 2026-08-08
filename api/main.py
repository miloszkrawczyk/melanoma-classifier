"""The web service: send it a photo, get a diagnosis back.

    uv run uvicorn api.main:app --port 8000

Then open http://127.0.0.1:8000/docs to try it from a browser.

Nothing here decides anything. It unpacks an upload, hands it to Predictor and
turns the answer into JSON. That separation is deliberate: predict.py knows
nothing about HTTP, so it can be called from a test, a script or a notebook
just as easily.
"""

from io import BytesIO

import gradio as gr
from fastapi import FastAPI, HTTPException, Response, UploadFile
from PIL import Image, UnidentifiedImageError

from frontend.ui import CSS, THEME, build_ui
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


def read_image(file):
    """Open an upload as a picture, or say plainly that it was not one."""
    try:
        return Image.open(file.file)
    except UnidentifiedImageError:
        # 400 means "your request was wrong", as against 500 which would mean
        # "our code broke". Sending a text file here is the caller's mistake.
        raise HTTPException(status_code=400, detail="that file is not an image")


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
    return predictor.predict(read_image(file))


@app.post(
    "/explain",
    responses={200: {"content": {"image/png": {}}}},
    response_class=Response,
)
async def explain(file: UploadFile):
    """Return the photo with a heatmap of what the answer rested on.

    An image rather than JSON, because the point is to be looked at. If the
    heat sits on the lesion the model was reading the right thing; if it sits
    on a ruler or a hair in the corner, the answer is worth nothing however
    confident it sounded.
    """
    picture = predictor.explain(read_image(file))

    # PNG rather than JPEG. The heatmap has large flat areas of one colour,
    # which JPEG smears into blotches.
    buffer = BytesIO()
    picture.save(buffer, format="PNG")

    return Response(buffer.getvalue(), media_type="image/png")


# The interface and the JSON API in one process, sharing one model. Mounted
# last, because it takes over the root path and everything above keeps its own.
#
#   /          the page
#   /predict   json
#   /explain   png
#   /docs      api documentation
app = gr.mount_gradio_app(app, build_ui(predictor), path="/", theme=THEME, css=CSS)
