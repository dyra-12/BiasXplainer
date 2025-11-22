#!/usr/bin/env python3
"""Upload repository files to a Hugging Face Space using huggingface_hub.

This script reads the token saved by `hf auth login` from the local cache
and uploads the repository contents to the given Space repo. It deliberately
skips large or runtime-specific folders such as `models/`, `results/`,
`.git/` and `__pycache__` to avoid pushing bulky artifacts.

Usage: python scripts/push_to_hf_space.py
"""

import os
from huggingface_hub import HfApi


def main():
    repo_id = "Dyra1204/BiasGuard-Pro"
    skip_dirs = {"models", "results", ".git", "__pycache__", ".venv", ".cache"}

    # Read token saved by `hf auth login` (if available)
    token_path = os.path.expanduser("~/.cache/huggingface/token")
    if not os.path.exists(token_path):
        token_path = os.path.expanduser("~/.cache/huggingface/stored_tokens")

    token = None
    if os.path.exists(token_path):
        with open(token_path, "r") as f:
            token = f.read().strip()

    if not token:
        raise SystemExit("No Hugging Face token found in cache. Run `hf auth login` first.")

    api = HfApi()

    print(f"Creating or using Space repo: {repo_id} ...")
    # For Spaces, specify the SDK (gradio / streamlit / docker / static)
    api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="gradio", exist_ok=True, token=token)

    # Walk repository and upload files (skip ignored directories)
    to_upload = []
    for root, dirs, files in os.walk("."):
        # mutate dirs in-place to skip unwanted directories
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for fn in files:
            if fn.endswith((".pyc", ".pyo", ".log")):
                continue
            # skip hidden cache files in repo root
            rel_dir = os.path.relpath(root, ".")
            if rel_dir.startswith("."):
                rel_dir = ""
            local_path = os.path.join(root, fn)
            # avoid uploading the uploader script itself twice when executed from repo
            if local_path.startswith("./.git"):
                continue
            repo_path = os.path.normpath(os.path.join(rel_dir, fn))
            to_upload.append((local_path, repo_path))

    print(f"Found {len(to_upload)} files to upload (excluding {', '.join(sorted(skip_dirs))}).")

    for local_path, repo_path in to_upload:
        # huggingface_hub expects path_in_repo without leading ./
        path_in_repo = repo_path.lstrip("./")
        try:
            print(f"Uploading {path_in_repo} ...")
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="space",
                token=token,
            )
        except Exception as e:
            print(f"Failed to upload {path_in_repo}: {e}")

    print("Upload finished. The Space should start building shortly.")


if __name__ == "__main__":
    main()
