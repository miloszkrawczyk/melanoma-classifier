"""Pushes the app to a Hugging Face Space, which builds and hosts the image.

    uv run hf auth login
    uv run python -m scripts.deploy_space miloszkrawczyk/melanoma-classifier

A Space is a git repository on Hugging Face's side. Handing it a Dockerfile
makes it build the image and serve it under a public address.

Only the files the container needs go up. The dataset, the training scripts,
the tests and the recorded results stay in the GitHub repo, which is where
they belong. The weights do go, all 43 MB of them, because they are gitignored
on GitHub and the image cannot be built without them.
"""

import sys

from huggingface_hub import HfApi

from src.config import MODELS, PROJECT_ROOT

# Same list as the Dockerfile copies, plus the Dockerfile itself.
UPLOAD = [
    "Dockerfile",
    "src/**",
    "api/**",
    "frontend/**",
    "weights/resnet.pth",
]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("give me a space id, e.g. miloszkrawczyk/melanoma-classifier")

    repo_id = sys.argv[1]

    if not MODELS["resnet"]["weights"].exists():
        raise SystemExit(
            "no weights/resnet.pth. Train them first: "
            "uv run python -m src.train resnet"
        )

    api = HfApi()

    # Private on creation, so a half-built Space is not public while the first
    # image is still compiling. Flipping it later is one click in the Space's
    # settings; exist_ok means this line leaves an existing Space alone,
    # including its visibility.
    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="docker",
        private=True,
        exist_ok=True,
    )
    print(f"space ready: https://huggingface.co/spaces/{repo_id}")

    # The Space reads its settings from the frontmatter at the top of its own
    # README, so that file has to be the one in space/, not the project README.
    api.upload_file(
        path_or_fileobj=PROJECT_ROOT / "space" / "README.md",
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="space",
    )

    api.upload_folder(
        folder_path=PROJECT_ROOT,
        repo_id=repo_id,
        repo_type="space",
        allow_patterns=UPLOAD,
        commit_message="deploy from the project repo",
    )

    print("pushed. The Space builds the image now, which takes a few minutes.")
    print(f"watch it at https://huggingface.co/spaces/{repo_id}?logs=build")
