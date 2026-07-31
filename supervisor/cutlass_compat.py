from __future__ import annotations

import argparse
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


MARKER = "# ATREX Supervisor temporary CUTLASS DSL compatibility shim"
EXCLUDE_LINE = "/sitecustomize.py"

SITECUSTOMIZE = f'''{MARKER}
"""Process-local workaround for nvidia-cutlass-dsl 4.5.2 package discovery.

The upstream get_version() implementation uses pkgutil.walk_packages(), which imports both
public and private MLIR helper namespaces and can register the same value caster twice.  Hash
the installed package metadata without importing every discovered module.  This file is
materialized only while a supervised executor/sandbox command is active.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.metadata
import importlib.abc
import importlib.machinery
import sys
from pathlib import Path


def _patch_cutlass_module(cutlass_module) -> None:
    base = getattr(cutlass_module, "CutlassBaseDSL", None)
    original = getattr(base, "get_version", None) if base is not None else None
    if original is None or getattr(original, "_atrex_supervisor_compat", False):
        return

    @functools.lru_cache(maxsize=1)
    def compatible_get_version(self):
        digest = hashlib.sha256()
        try:
            version = importlib.metadata.version("nvidia-cutlass-dsl")
        except importlib.metadata.PackageNotFoundError:
            version = "unknown"
        digest.update(version.encode("utf-8"))

        package_root = Path(cutlass_module.__file__).resolve().parents[1]
        for path in sorted(package_root.rglob("*")):
            if not path.is_file() or path.suffix not in {{".py", ".pyi", ".so"}}:
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            digest.update(path.relative_to(package_root).as_posix().encode("utf-8"))
            digest.update(str(stat.st_size).encode("ascii"))
            digest.update(str(stat.st_mtime_ns).encode("ascii"))
        return digest

    compatible_get_version._atrex_supervisor_compat = True
    base.get_version = compatible_get_version


class _CutlassLoader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        create_module = getattr(self.original, "create_module", None)
        return create_module(spec) if create_module is not None else None

    def exec_module(self, module):
        self.original.exec_module(module)
        _patch_cutlass_module(module)


class _CutlassFinder(importlib.abc.MetaPathFinder):
    target = "cutlass.cutlass_dsl.cutlass"

    def find_spec(self, fullname, path, target=None):
        if fullname != self.target:
            return None
        try:
            sys.meta_path.remove(self)
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        finally:
            sys.meta_path.insert(0, self)
        if spec is not None and spec.loader is not None:
            spec.loader = _CutlassLoader(spec.loader)
        return spec


loaded = sys.modules.get(_CutlassFinder.target)
if loaded is not None:
    _patch_cutlass_module(loaded)
else:
    sys.meta_path.insert(0, _CutlassFinder())
'''


class WorkspaceCompatibility:
    """Temporarily inject a sandbox-visible compatibility module without dirtying Git."""

    def __init__(self, workspace: Path):
        self.workspace = workspace.resolve()
        self.sitecustomize = self.workspace / "sitecustomize.py"
        self.git_exclude = self.workspace / ".git" / "info" / "exclude"
        self._created = False
        self._exclude_added = False

    def install(self) -> None:
        if self.sitecustomize.exists():
            existing = self.sitecustomize.read_text(encoding="utf-8", errors="replace")
            if MARKER not in existing:
                raise RuntimeError(
                    f"refusing to overwrite workspace-owned compatibility file: {self.sitecustomize}"
                )
        else:
            self.sitecustomize.write_text(SITECUSTOMIZE, encoding="utf-8")
            self._created = True

        if self.git_exclude.parent.is_dir():
            existing = self.git_exclude.read_text(encoding="utf-8") if self.git_exclude.exists() else ""
            if EXCLUDE_LINE not in existing.splitlines():
                separator = "" if not existing or existing.endswith("\n") else "\n"
                self.git_exclude.write_text(
                    existing + separator + EXCLUDE_LINE + "\n",
                    encoding="utf-8",
                )
                self._exclude_added = True

    def remove(self) -> None:
        if self._created and self.sitecustomize.exists():
            existing = self.sitecustomize.read_text(encoding="utf-8", errors="replace")
            if MARKER in existing:
                self.sitecustomize.unlink()
        if self._exclude_added and self.git_exclude.exists():
            lines = self.git_exclude.read_text(encoding="utf-8").splitlines()
            removed = False
            kept: list[str] = []
            for line in lines:
                if not removed and line == EXCLUDE_LINE:
                    removed = True
                    continue
                kept.append(line)
            text = "\n".join(kept)
            self.git_exclude.write_text(text + ("\n" if text else ""), encoding="utf-8")

    def __enter__(self) -> "WorkspaceCompatibility":
        self.install()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.remove()


@contextmanager
def workspace_compatibility(workspace: Path) -> Iterator[None]:
    compatibility = WorkspaceCompatibility(workspace)
    compatibility.install()
    try:
        yield
    finally:
        compatibility.remove()


def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run one command with the temporary Supervisor CUTLASS compatibility shim."
    )
    parser.add_argument("--workspace", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a command is required after --")
    return args, command


def main(argv: list[str] | None = None) -> int:
    args, command = _parse_args(list(argv) if argv is not None else sys.argv[1:])
    workspace = Path(args.workspace).resolve()
    env = os.environ.copy()
    with workspace_compatibility(workspace):
        result = subprocess.run(command, cwd=str(workspace), env=env)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
