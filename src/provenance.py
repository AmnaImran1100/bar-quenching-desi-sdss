"""Content fingerprints for reproducible results and validated RF caches."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path


PACKAGES = ("numpy", "pandas", "scipy", "astropy", "scikit-learn",
            "matplotlib", "pyarrow", "tqdm")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def versions():
    return {"python": platform.python_version(),
            **{name: importlib.metadata.version(name) for name in PACKAGES}}


def fingerprint(root, paths, settings=None):
    root = Path(root).resolve()
    return dict(files={str(Path(p).resolve().relative_to(root)).replace("\\", "/"):
                       sha256(p) for p in paths},
                versions=versions(), settings=settings or {})


def cache_valid(path, inputs):
    path = Path(path)
    metadata = path.with_suffix(".provenance.json")
    if not path.is_file() or not metadata.is_file():
        return False
    try:
        record = json.loads(metadata.read_text(encoding="utf-8"))
        return record == dict(inputs=inputs, output_sha256=sha256(path))
    except (ValueError, OSError):
        return False


def record_cache(path, inputs):
    path = Path(path)
    path.with_suffix(".provenance.json").write_text(
        json.dumps(dict(inputs=inputs, output_sha256=sha256(path)), indent=2)
        + "\n", encoding="utf-8")
