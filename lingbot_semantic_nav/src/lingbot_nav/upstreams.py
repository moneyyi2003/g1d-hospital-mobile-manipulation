"""Read and verify reproducible official upstream checkouts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess

from .errors import ConfigurationError


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class UpstreamRepository:
    name: str
    url: str
    commit: str
    checkout: Path
    license: str
    group: str

    def verify(self) -> None:
        if not (self.checkout / ".git").is_dir():
            raise ConfigurationError(
                f"Missing upstream {self.name}; run scripts/fetch_upstreams.py"
            )
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.checkout,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        actual = result.stdout.strip()
        if result.returncode != 0 or actual != self.commit:
            raise ConfigurationError(
                f"Upstream {self.name} is at {actual or 'unknown'}, expected {self.commit}"
            )


def load_upstreams(path: str | Path | None = None) -> dict[str, UpstreamRepository]:
    source = Path(path) if path else PROJECT_ROOT / "config" / "upstreams.lock.json"
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
        repositories = {}
        for value in payload["repositories"]:
            checkout = PROJECT_ROOT / str(value["checkout"])
            item = UpstreamRepository(
                name=str(value["name"]),
                url=str(value["url"]),
                commit=str(value["commit"]),
                checkout=checkout,
                license=str(value["license"]),
                group=str(value["group"]),
            )
            repositories[item.name] = item
        return repositories
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"Cannot read upstream lock {source}: {exc}") from exc


def require_upstream(name: str) -> UpstreamRepository:
    repositories = load_upstreams()
    try:
        repository = repositories[name]
    except KeyError as exc:
        raise ConfigurationError(f"Unknown upstream repository: {name}") from exc
    repository.verify()
    return repository


__all__ = ["UpstreamRepository", "load_upstreams", "require_upstream"]
