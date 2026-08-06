from __future__ import annotations

import hashlib
from pathlib import Path

from long_horizon.protocol import atomic_write_json


def tree_digest(root: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    for path in sorted(
        item for item in root.rglob("*") if item.is_file() and not item.is_symlink()
    ):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
        count += 1
    return digest.hexdigest(), count


def write_lock(
    destination: Path,
    *,
    source_checkout: Path,
    source_revision: str,
    source_root: Path,
    manifest_sha256: str,
    atrex_bench_root: str,
    runtime_support: list[dict[str, object]] | None = None,
    runtime_support_root: Path | None = None,
) -> dict:
    digest, count = tree_digest(source_root)
    support_digest, support_count = (
        tree_digest(runtime_support_root)
        if runtime_support_root is not None and runtime_support_root.is_dir()
        else (None, 0)
    )
    payload = {
        "schema_version": 1,
        "source_checkout": str(source_checkout),
        "source_revision": source_revision,
        "source_tree_sha256": digest,
        "source_file_count": count,
        "manifest_sha256": manifest_sha256,
        "atrex_bench_root": atrex_bench_root,
        "runtime_support": runtime_support or [],
        "runtime_support_tree_sha256": support_digest,
        "runtime_support_file_count": support_count,
    }
    atomic_write_json(destination, payload)
    return payload
