#!/usr/bin/env python3
"""Upload the hf/ folder to the Hugging Face Hub as a gated dataset.

Needs a write token: run `hf auth login` first, or set HF_TOKEN.

Usage:
  python scripts/upload_hf.py --repo social-atoms/hugagent --tag data-v1.0
"""
import argparse
from pathlib import Path
from huggingface_hub import HfApi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="namespace/name on the Hub")
    ap.add_argument("--folder", default=Path(__file__).resolve().parent.parent / "hf", type=Path)
    ap.add_argument("--tag", default=None, help="git tag to create on the Hub, e.g. data-v1.0")
    ap.add_argument("--private", action="store_true", help="create as private (default: public, gated)")
    args = ap.parse_args()

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", private=args.private, exist_ok=True)
    api.upload_folder(repo_id=args.repo, repo_type="dataset", folder_path=str(args.folder),
                      commit_message="Data release", ignore_patterns=[".DS_Store"])
    if not args.private:
        # Gated with automatic approval: users accept the terms in the card, access is logged.
        api.update_repo_settings(repo_id=args.repo, repo_type="dataset", gated="auto")
    if args.tag:
        api.create_tag(args.repo, repo_type="dataset", tag=args.tag, tag_message=f"HugAgent {args.tag}", exist_ok=True)
    print(f"https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
