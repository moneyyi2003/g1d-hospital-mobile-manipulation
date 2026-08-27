#!/usr/bin/env python3
"""Download LingBot-Map and SAM3.1 checkpoints.

Usage:
    python scripts/download_checkpoints.py --all
    python scripts/download_checkpoints.py --lingbot
    python scripts/download_checkpoints.py --sam3
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


CHECKPOINTS_DIR = Path(__file__).resolve().parents[1] / "checkpoints"


# ── LingBot-Map checkpoint ─────────────────────────────────────────────────────
# LingBot-Map: https://github.com/robbyant/lingbot-map
# The checkpoint is available on Hugging Face Hub.
LINGBOT_REPO = "robbyant/lingbot-map"
LINGBOT_FILE = "lingbot_map_stream.pt"


def download_lingbot_checkpoint():
    """Download the LingBot-Map model checkpoint from Hugging Face."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("huggingface_hub is required. Install with: pip install huggingface_hub")
        return None

    dest = CHECKPOINTS_DIR / "lingbot-map.pt"
    if dest.is_file():
        print(f"LingBot checkpoint already exists: {dest}")
        return dest

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading LingBot-Map checkpoint from {LINGBOT_REPO}...")
    path = hf_hub_download(
        repo_id=LINGBOT_REPO,
        filename=LINGBOT_FILE,
        local_dir=CHECKPOINTS_DIR,
        local_dir_use_symlinks=False,
    )
    # Rename to our canonical name
    downloaded = Path(path)
    if downloaded != dest:
        downloaded.rename(dest)
    print(f"  Saved to: {dest}")
    return dest


# ── SAM3.1 checkpoint ──────────────────────────────────────────────────────────
# SAM3.1 从魔搭社区 (ModelScope) 下载，使用 git clone + git-lfs。
SAM3_MODELSCOPE_URL = "https://www.modelscope.cn/facebook/sam3.1.git"


def _check_git_lfs():
    """Ensure git-lfs is installed and initialized."""
    try:
        subprocess.run(
            ["git", "lfs", "version"],
            capture_output=True, text=True, check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: git-lfs is not installed or not found.")
        print("  Install it first:")
        print("    Ubuntu/Debian: sudo apt install git-lfs")
        print("    macOS:         brew install git-lfs")
        print("    CentOS/RHEL:   sudo yum install git-lfs")
        return False

    # Initialize git-lfs globally if not already done
    subprocess.run(["git", "lfs", "install"], capture_output=True, check=False)
    return True


def download_sam3_checkpoint():
    """Clone SAM3.1 checkpoint from ModelScope (魔搭社区) via git-lfs."""
    if not _check_git_lfs():
        return None

    dest_dir = CHECKPOINTS_DIR / "sam3.1"
    if dest_dir.is_dir() and (dest_dir / "sam3.1_multiplex.pt").is_file():
        print(f"SAM3.1 checkpoint already exists: {dest_dir}")
        return dest_dir

    print(f"Cloning SAM3.1 from ModelScope (魔搭社区)...")
    print(f"  URL: {SAM3_MODELSCOPE_URL}")

    # Remove directory if it exists but is incomplete
    if dest_dir.is_dir():
        import shutil
        shutil.rmtree(dest_dir)

    try:
        subprocess.run(
            ["git", "clone", SAM3_MODELSCOPE_URL, str(dest_dir)],
            check=True,
        )
        print(f"  Saved to: {dest_dir}")

        # Verify the main checkpoint file
        expected = dest_dir / "sam3.1_multiplex.pt"
        if not expected.is_file():
            print(f"  WARNING: {expected.name} not found after clone. "
                  f"The repo may not contain the expected file.")
        else:
            size_mb = expected.stat().st_size / (1024 * 1024)
            print(f"  Main checkpoint: {expected.name} ({size_mb:.0f} MB)")

        # Remove .git directory to save space
        git_dir = dest_dir / ".git"
        if git_dir.is_dir():
            import shutil
            shutil.rmtree(git_dir)
            print(f"  Removed .git directory to save space")

        return dest_dir

    except subprocess.CalledProcessError as exc:
        print(f"  ERROR: git clone failed: {exc}")
        print(f"  Manual fallback:")
        print(f"    git lfs install")
        print(f"    git clone {SAM3_MODELSCOPE_URL} {dest_dir}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Download LingBot-SAM3 checkpoints")
    parser.add_argument("--all", action="store_true", help="Download all checkpoints")
    parser.add_argument("--lingbot", action="store_true", help="Download LingBot-Map checkpoint")
    parser.add_argument("--sam3", action="store_true", help="Download SAM3.1 checkpoint from ModelScope")
    args = parser.parse_args()

    download_all = args.all or (not args.lingbot and not args.sam3)

    if download_all or args.lingbot:
        print("\n" + "=" * 60)
        print("LingBot-Map Checkpoint (Hugging Face)")
        print("=" * 60)
        download_lingbot_checkpoint()

    if download_all or args.sam3:
        print("\n" + "=" * 60)
        print("SAM3.1 Checkpoint (魔搭社区 ModelScope)")
        print("=" * 60)
        download_sam3_checkpoint()

    print("\nDone. Checkpoints saved to:", CHECKPOINTS_DIR)


if __name__ == "__main__":
    main()
