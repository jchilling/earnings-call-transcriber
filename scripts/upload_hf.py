"""Upload all scraped data to HuggingFace.

Uses upload_large_folder from data/audio/ which auto-chunks into multiple commits.
Repo structure mirrors local: alphamemo/{ticker}/*, twse_webpro/{ticker}/*, twse_mops/{ticker}/*

Usage:
    python scripts/upload_hf.py
    python scripts/upload_hf.py --source alphamemo
    python scripts/upload_hf.py --source twse
    python scripts/upload_hf.py --source mops
"""

import argparse
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def upload(source: str | None = None) -> None:
    hf_token = None
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token_path.exists():
        hf_token = token_path.read_text().strip()

    if not hf_token:
        hf_token = os.environ.get("HF_TOKEN")

    if not hf_token:
        print("No HF token found. Set HF_TOKEN or run `huggingface-cli login`")
        return

    from huggingface_hub import HfApi

    repo_id = os.environ.get("HF_REPO", "jchilling/taiwan-earnings-calls")
    api = HfApi(token=hf_token)
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)

    base_dir = Path("data/audio")

    # Build allow_patterns based on source filter
    patterns = []
    if source in (None, "alphamemo"):
        patterns += ["alphamemo/*/*.json", "alphamemo/*/*.mp3", "alphamemo/download_results.json"]
    if source in (None, "twse"):
        patterns += ["twse_webpro/*/*.mp3", "twse_webpro/download_results.json"]
    if source in (None, "mops"):
        patterns += ["twse_mops/*/*.mp3", "twse_mops/download_results.json"]

    # Ignore the .cache dir and other non-data files
    ignore = ["overnight.log", "alphamemo_upload.log", "*.log"]

    source_label = source or "all sources"
    print(f"Uploading {source_label} from {base_dir}/ via upload_large_folder...")
    print(f"Patterns: {patterns}")

    api.upload_large_folder(
        folder_path=str(base_dir),
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=patterns,
        ignore_patterns=ignore,
    )

    print("HuggingFace upload complete!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["alphamemo", "twse", "mops"], help="Upload only one source")
    args = parser.parse_args()
    upload(source=args.source)
