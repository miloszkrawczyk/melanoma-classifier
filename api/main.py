"""The web service. Right now it only answers whether it is alive.

    uv run uvicorn api.main:app --reload

Then open http://127.0.0.1:8000/health in a browser.
"""

from fastapi import FastAPI

app = FastAPI(title="Melanoma classifier")


@app.get("/health")
def health():
    """Says the service is up.

    Looks pointless and is not. Hosting platforms poll an address like this
    to decide whether the app finished starting and whether it needs a
    restart. Without one they have to guess from whether the port is open,
    which stays true even when the app is broken.
    """
    return {"status": "ok"}
