"""Pushes the trained weights to the Hugging Face Hub as a model repository.

    uv run hf auth login
    uv run python -m scripts.upload_model miloszkrawczyk/skin-lesion-classifier

Solves a real problem: the weights are 43 MB and gitignored, so cloning the
GitHub repo leaves you unable to build the image or run anything. Putting them
on the Hub means the code lives in one place and the model in another, which
is how this is usually arranged, and the README can point at both.

Model repositories are free on the Hub. Spaces, which host a running
container, are not.
"""

import sys

from huggingface_hub import HfApi

from src.config import MODELS, PROJECT_ROOT

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("give me a repo id, e.g. miloszkrawczyk/skin-lesion-classifier")

    repo_id = sys.argv[1]
    weights = MODELS["resnet"]["weights"]

    if not weights.exists():
        # Two ways to be missing it, and retraining is the expensive one. A
        # checkout from before the format change has the .pth and needs only
        # the converter.
        raise SystemExit(
            f"no {weights}.\n"
            "  have the old .pth? uv run python -m scripts.convert_weights\n"
            "  have neither?     uv run python -m src.train resnet"
        )

    api = HfApi()

    # Private to start with. A model card quoting test-set numbers should be
    # read before it is published, and flipping it public later is one click.
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)

    api.upload_file(
        path_or_fileobj=PROJECT_ROOT / "deploy" / "model" / "README.md",
        path_in_repo="README.md",
        repo_id=repo_id,
    )

    # Renamed on the way up: the name says what it is without needing the
    # weights/ directory the project happens to keep it in.
    #
    # safetensors, because a published pickle asks everyone who downloads it to
    # run whatever is inside. The older resnet.pth already on the Hub is left
    # alone, so the download instruction in the README keeps working.
    api.upload_file(
        path_or_fileobj=weights,
        path_in_repo="resnet.safetensors",
        repo_id=repo_id,
        commit_message="upload fine-tuned resnet18",
    )

    print(f"done: https://huggingface.co/{repo_id}")
    print("It is private. Settings -> Change visibility to publish it.")
