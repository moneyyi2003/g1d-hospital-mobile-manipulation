"""Small, dependency-free loader for project-local environment settings."""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | Path | None = None) -> Path | None:
    """Load KEY=VALUE pairs without replacing variables already in the environment."""
    candidates = (
        [Path(path)]
        if path is not None
        else [Path.cwd() / ".env", Path(__file__).resolve().parents[2] / ".env"]
    )
    env_path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if env_path is None:
        return None

    for line_number, raw_line in enumerate(
        env_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ValueError(f"Invalid .env entry at {env_path}:{line_number}")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not key.replace("_", "a").isalnum() or key[0].isdigit():
            raise ValueError(f"Invalid .env key at {env_path}:{line_number}")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)
    return env_path
