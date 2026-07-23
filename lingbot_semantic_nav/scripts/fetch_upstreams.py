#!/usr/bin/env python3
"""Fetch exact official upstream revisions without vendoring their source.

The lock file is authoritative. Model repositories use sparse checkouts so the
workspace does not download benchmark datasets or demo media. ROS/simulator
repositories are normally installed from the official Humble packages and are
only cloned when their group is explicitly requested.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "config" / "upstreams.lock.json"


def run(*args: str, cwd: Path | None = None) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def current_commit(target: Path) -> str | None:
    if not (target / ".git").is_dir():
        return None
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=target,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def fetch(repository: dict[str, object], *, force_fetch: bool) -> None:
    target = ROOT / str(repository["checkout"])
    expected = str(repository["commit"])
    actual = current_commit(target)
    if actual == expected:
        print(f"ok  {repository['name']} {expected[:12]}")
        return
    if target.exists() and any(target.iterdir()) and actual is None:
        raise RuntimeError(f"Refusing to overwrite non-git directory: {target}")
    if actual and not force_fetch:
        raise RuntimeError(
            f"{target} is at {actual}; pass --update to move it to {expected}"
        )

    target.mkdir(parents=True, exist_ok=True)
    if not (target / ".git").is_dir():
        run("git", "init", "--quiet", str(target))
        run("git", "remote", "add", "origin", str(repository["url"]), cwd=target)

    run("git", "fetch", "--depth", "1", "--filter=blob:none", "origin", expected, cwd=target)
    sparse_paths = repository.get("sparse_paths")
    if isinstance(sparse_paths, list) and sparse_paths:
        run("git", "sparse-checkout", "init", "--no-cone", cwd=target)
        patterns = ["/*", "!/*/"]
        patterns.extend(f"/{item}/" if "." not in Path(str(item)).name else f"/{item}" for item in sparse_paths)
        run("git", "sparse-checkout", "set", "--no-cone", *patterns, cwd=target)
    run("git", "checkout", "--detach", "--force", expected, cwd=target)
    print(f"get {repository['name']} {expected[:12]} -> {target.relative_to(ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--group",
        action="append",
        choices=("models", "ros", "simulation", "all"),
        default=None,
        help="Repository group to fetch; repeat for multiple groups.",
    )
    parser.add_argument("--update", action="store_true", help="Move existing checkouts to the locked commit.")
    args = parser.parse_args()
    payload = json.loads(LOCK.read_text(encoding="utf-8"))
    selected = set(args.group or ["models"])
    if "all" in selected:
        selected = {"models", "ros", "simulation"}
    try:
        for repository in payload["repositories"]:
            if repository["group"] in selected:
                fetch(repository, force_fetch=args.update)
    except (OSError, subprocess.CalledProcessError, RuntimeError) as exc:
        print(f"upstream fetch failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
