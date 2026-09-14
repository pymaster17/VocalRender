#!/usr/bin/env python
"""Publish the web demo to the Hugging Face Space.

The GitHub repository is the single source of truth; the Space is a flattened
build product:

    <Space root>/            <- demo/  (app.py, assets/, requirements.txt, README.md, .gitattributes)
    <Space root>/src/vocalrender/  <- src/vocalrender/  (the package app.py imports)

Usage (needs a write token for the Space, e.g. HF_TOKEN in the environment):

    python scripts/sync_space.py                 # upload to pymaster/VocalRender-demo
    python scripts/sync_space.py --dry-run       # only list what would be uploaded
    python scripts/sync_space.py --repo-id user/other-space

Run automatically by .github/workflows/sync-space.yml on pushes to main.
"""
from __future__ import annotations

import argparse
import fnmatch
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo"
PACKAGE = ROOT / "src" / "vocalrender"

# Development-only files that must not land in the Space.
DEMO_EXCLUDE = ["Dockerfile*", "tests", ".gitignore", ".venv*", "__pycache__", ".pytest_cache"]
PACKAGE_EXCLUDE = ["__pycache__", "*.pyc"]


def _ignore(patterns):
    def ignore(_dir, names):
        return {n for n in names if any(fnmatch.fnmatch(n, p) for p in patterns)}

    return ignore


def assemble(target: Path) -> None:
    shutil.copytree(DEMO, target, ignore=_ignore(DEMO_EXCLUDE), dirs_exist_ok=True)
    shutil.copytree(PACKAGE, target / "src" / "vocalrender", ignore=_ignore(PACKAGE_EXCLUDE))
    # LFS pointers would break the Space; refuse to upload unresolved ones.
    for wav in (target / "assets").glob("**/*.wav"):
        head = wav.read_bytes()[:40]
        if head.startswith(b"version https://git-lfs"):
            raise SystemExit(f"{wav.relative_to(target)} is a Git LFS pointer; run `git lfs pull` first")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", default="pymaster/VocalRender-demo")
    parser.add_argument("--dry-run", action="store_true", help="assemble and list files, do not upload")
    parser.add_argument("--message", default=None, help="commit message on the Space (default: git describe)")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="vocalrender-space-") as tmp:
        staged = Path(tmp) / "space"
        assemble(staged)
        files = sorted(p.relative_to(staged).as_posix() for p in staged.rglob("*") if p.is_file())
        print(f"{len(files)} files staged for {args.repo_id}:")
        for f in files:
            print("  ", f)
        if args.dry_run:
            return

        from huggingface_hub import HfApi

        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        message = args.message or f"Sync from GitHub {sha.stdout.strip() or 'HEAD'}"
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        url = api.upload_folder(
            repo_id=args.repo_id,
            repo_type="space",
            folder_path=str(staged),
            commit_message=message,
            delete_patterns=["**"],  # mirror: files removed on GitHub disappear from the Space
        )
        print("uploaded:", url)


if __name__ == "__main__":
    main()
