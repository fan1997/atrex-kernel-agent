from __future__ import annotations

import ast
import hashlib
import re
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

from .lockfile import tree_digest
from .manifest import RuntimeSupportWheel


def canonical_distribution(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _safe_member(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe support wheel member: {value!r}")
    return path.as_posix()


def _wheel_metadata(
    archive: zipfile.ZipFile, expected_distribution: str
) -> tuple[str, str, str]:
    candidates = []
    for name in archive.namelist():
        path = PurePosixPath(name)
        if len(path.parts) == 2 and path.parts[0].endswith(".dist-info"):
            if path.parts[1] == "METADATA":
                message = BytesParser().parsebytes(archive.read(name))
                distribution = str(message.get("Name", ""))
                version = str(message.get("Version", ""))
                if canonical_distribution(distribution) == canonical_distribution(
                    expected_distribution
                ):
                    candidates.append((path.parts[0], distribution, version))
    if len(candidates) != 1:
        raise ValueError(
            f"wheel must contain exactly one METADATA for {expected_distribution}"
        )
    return candidates[0]


def _local_imports(source: bytes, package: str) -> set[str]:
    try:
        tree = ast.parse(source.decode("utf-8"))
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise ValueError(
            "support wheel Python members must be valid UTF-8 source"
        ) from exc
    result: set[str] = set()
    package_prefix = package + "."
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
            if node.module == package:
                names.extend(f"{package}.{alias.name}" for alias in node.names)
        else:
            continue
        for name in names:
            if name.startswith(package_prefix):
                result.add(name.replace(".", "/") + ".py")
    return result


def wheel_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_support_imports(
    source_root: Path,
    configs: tuple[RuntimeSupportWheel, ...],
    support_root: Path,
) -> None:
    """Ensure the vendored source's direct support-package imports were selected."""
    for config in configs:
        required: set[str] = set()
        for path in source_root.rglob("*.py"):
            required.update(_local_imports(path.read_bytes(), config.package))
        package_path = config.package.replace(".", "/")
        missing = []
        for module_path in sorted(required):
            if module_path == f"{package_path}.py":
                continue
            module = support_root / module_path
            package = support_root / module_path.removesuffix(".py") / "__init__.py"
            if not module.is_file() and not package.is_file():
                missing.append(module_path)
        if missing:
            raise ValueError(
                f"source imports omitted {config.distribution} support members: "
                + ", ".join(missing)
            )


def extract_support_wheel(
    config: RuntimeSupportWheel,
    wheel: Path,
    destination: Path,
) -> dict[str, object]:
    wheel = wheel.resolve()
    if not wheel.is_file():
        raise FileNotFoundError(f"support wheel not found: {wheel}")
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(wheel) as archive:
        dist_info, actual_distribution, actual_version = _wheel_metadata(
            archive, config.distribution
        )
        if actual_version != config.version:
            raise ValueError(
                f"support wheel version mismatch: {actual_distribution}={actual_version}, "
                f"required {config.version}"
            )
        available = set(archive.namelist())
        extracted: list[str] = []
        sources: dict[str, bytes] = {}
        member_data: dict[str, bytes] = {}
        package_init = config.package.replace(".", "/") + "/__init__.py"
        for raw in config.members:
            member = _safe_member(raw)
            if config.generate_minimal_init and member == package_init:
                raise ValueError(
                    f"{package_init} is generated; do not list it as an extracted member"
                )
            if member not in available:
                raise ValueError(f"support wheel member is missing: {member}")
            data = archive.read(member)
            if member.endswith(".py"):
                _local_imports(data, config.package)
                sources[member] = data
            member_data[member] = data
            extracted.append(member)

        missing_closure: set[str] = set()
        for data in sources.values():
            for imported in _local_imports(data, config.package):
                if imported in available and imported not in sources:
                    missing_closure.add(imported)
        if missing_closure:
            missing = ", ".join(sorted(missing_closure))
            raise ValueError(
                f"runtime support members omit local import closure: {missing}"
            )

        dist_info_data: dict[str, bytes] = {}
        dist_info_extracted = []
        for raw in config.dist_info_members:
            leaf = _safe_member(raw)
            member = f"{dist_info}/{leaf}"
            if member not in available:
                raise ValueError(f"support wheel dist-info member is missing: {member}")
            dist_info_data[member] = archive.read(member)
            dist_info_extracted.append(member)

        for member, data in {**member_data, **dist_info_data}.items():
            target = destination / member
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)

        generated: dict[str, str] = {}
        if config.generate_minimal_init:
            init_text = (
                '"""Generated minimal package shim for Repository Horizon runtime support."""\n'
                f'__version__ = "{config.version}"\n'
            )
            target = destination / package_init
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(init_text, encoding="utf-8")
            generated[package_init] = "minimal_package_init"

    digest, count = tree_digest(destination)
    return {
        "distribution": actual_distribution,
        "version": actual_version,
        "wheel_filename": wheel.name,
        "wheel_sha256": wheel_sha256(wheel),
        "members": extracted,
        "dist_info_members": dist_info_extracted,
        "generated": generated,
        "tree_sha256": digest,
        "file_count": count,
    }
