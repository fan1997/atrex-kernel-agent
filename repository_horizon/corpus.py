from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from .manifest import RepositorySearchConfig


CORPUS_RELATIVE = ".repository_horizon_runtime/source_corpus.git"
CATALOG_NAME = "source_corpus.json"


def _git(path: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(path),
        check=check,
        capture_output=True,
        text=True,
    )


def _bare(path: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", f"--git-dir={path}", *args],
        check=check,
        capture_output=True,
        text=True,
    )


def _resolve(source: Path, value: str) -> str:
    result = _git(source, "rev-parse", "--verify", f"{value}^{{commit}}")
    return result.stdout.strip()


def _promisor_remote(source: Path) -> str | None:
    remotes = _git(source, "remote").stdout.splitlines()
    for remote in remotes:
        promisor = _git(
            source,
            "config",
            "--bool",
            "--get",
            f"remote.{remote}.promisor",
            check=False,
        )
        if promisor.returncode != 0 or promisor.stdout.strip() != "true":
            continue
        url = _git(
            source, "remote", "get-url", remote, check=False
        ).stdout.strip()
        if url:
            return url
    return None


def _fetch_commit(
    source: Path, destination: Path, commit: str, local_name: str
) -> str:
    refspec = f"{commit}:refs/heads/{local_name}"
    local = subprocess.run(
        [
            "git",
            f"--git-dir={destination}",
            "fetch",
            "--no-tags",
            "--force",
            str(source),
            refspec,
        ],
        capture_output=True,
        text=True,
    )
    if local.returncode == 0:
        return "source_checkout"
    # A partial/promisor checkout cannot serve objects it has not hydrated:
    # upload-pack deliberately disables lazy fetching. Fetching the exact
    # locked commit directly from the same promisor remote preserves the
    # bounded-ref contract while avoiding a forced full local clone.
    remote_url = _promisor_remote(source)
    if remote_url:
        remote = subprocess.run(
            [
                "git",
                f"--git-dir={destination}",
                "fetch",
                "--no-tags",
                "--force",
                remote_url,
                refspec,
            ],
            capture_output=True,
            text=True,
        )
        if remote.returncode == 0:
            return "promisor_remote"
        remote_error = f"fetch exited {remote.returncode}"
    else:
        remote_error = "source checkout has no promisor remote"
    local_error = local.stderr.strip()[-2000:]
    raise RuntimeError(
        "could not materialize bounded source corpus from local checkout "
        f"({local_error}) or its promisor remote ({remote_error})"
    )


def _digest_lines(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8") + b"\n")
    return digest.hexdigest()


def _object_inventory(destination: Path) -> list[str]:
    result = _bare(
        destination,
        "cat-file",
        "--batch-all-objects",
        "--batch-check=%(objectname) %(objecttype)",
    )
    return sorted(line for line in result.stdout.splitlines() if line)


def build_source_corpus(
    source_checkout: Path,
    revision: str,
    config: RepositorySearchConfig,
    destination: Path,
) -> dict[str, object] | None:
    """Build a physically bounded Git corpus for repository-side research.

    The editable source remains a flat, outer-repository snapshot.  This bare
    corpus is untracked runtime data used only for local source archaeology and
    is never included in Agate staging.
    """

    if config.mode == "snapshot":
        return None
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init", "--bare", str(destination)],
        check=True,
        capture_output=True,
        text=True,
    )
    requested: list[tuple[str, str]] = [("r0", revision)]
    if config.mode == "allowlist":
        requested.extend((f"ref_{index:04d}", ref) for index, ref in enumerate(config.refs))

    resolved_refs: dict[str, dict[str, str]] = {}
    for local_name, requested_ref in requested:
        commit = _resolve(source_checkout, requested_ref)
        transport = _fetch_commit(source_checkout, destination, commit, local_name)
        resolved_refs[local_name] = {
            "requested": requested_ref,
            "commit": commit,
            "transport": transport,
        }

    for excluded in config.excluded_commits:
        present = _bare(
            destination, "cat-file", "-e", f"{excluded}^{{commit}}", check=False
        )
        if present.returncode == 0:
            raise RuntimeError(
                f"excluded commit is physically present in source corpus: {excluded}"
            )

    commits = _bare(destination, "rev-list", "--all", "--topo-order").stdout.splitlines()
    if not commits or revision not in commits:
        raise RuntimeError("source corpus does not contain its locked R0 revision")
    if config.mode == "replay_strict":
        # `rev-list <revision>` is exactly the set of commits reachable from
        # the locked R0 revision.  Computing it once avoids spawning one
        # `git merge-base --is-ancestor` process per corpus commit, which made
        # large replay-strict repositories spend minutes in preflight.
        ancestors = set(
            _bare(destination, "rev-list", revision).stdout.splitlines()
        )
        unexpected = [commit for commit in commits if commit not in ancestors]
        if unexpected:
            raise RuntimeError(
                "replay_strict corpus contains non-ancestor commits: "
                + ", ".join(unexpected[:4])
            )

    # Fetch can leave unreachable loose objects or reflog entries. Canonicalize
    # the physical object database before recording its integrity boundary.
    _bare(destination, "reflog", "expire", "--expire=now", "--all")
    _bare(destination, "gc", "--prune=now")
    objects = _object_inventory(destination)
    return {
        "schema_version": 1,
        "mode": config.mode,
        "r0_revision": revision,
        "refs": resolved_refs,
        "excluded_commits": list(config.excluded_commits),
        "commit_count": len(commits),
        "commit_set_sha256": _digest_lines(sorted(commits)),
        "object_count": len(objects),
        "object_set_sha256": _digest_lines(objects),
        "runtime_git_dir": CORPUS_RELATIVE,
        "require_report": config.require_report,
    }


def validate_source_corpus(
    workspace: Path,
    catalog: dict[str, object],
    *,
    expected_runtime: Path | None = None,
) -> list[str]:
    """Return integrity violations for the episode-visible bounded corpus."""

    violations: list[str] = []
    corpus = workspace / CORPUS_RELATIVE
    if not corpus.is_dir():
        return ["bounded source corpus is missing"]
    if expected_runtime is not None:
        try:
            if corpus.resolve() != expected_runtime.resolve():
                violations.append("episode source corpus points outside campaign runtime")
        except OSError:
            violations.append("episode source corpus cannot be resolved")
            return violations
    r0 = str(catalog.get("r0_revision", ""))
    refs = catalog.get("refs")
    if not r0 or not isinstance(refs, dict):
        return violations + ["source corpus catalog is malformed"]
    for name, record in refs.items():
        if not isinstance(name, str) or not isinstance(record, dict):
            violations.append("source corpus ref catalog is malformed")
            continue
        expected = str(record.get("commit", ""))
        actual = _bare(corpus, "rev-parse", f"refs/heads/{name}", check=False)
        if actual.returncode != 0 or actual.stdout.strip() != expected:
            violations.append(f"source corpus ref changed: {name}")
    commits = _bare(corpus, "rev-list", "--all", "--topo-order", check=False)
    commit_lines = commits.stdout.splitlines() if commits.returncode == 0 else []
    if (
        len(commit_lines) != catalog.get("commit_count")
        or _digest_lines(sorted(commit_lines)) != catalog.get("commit_set_sha256")
    ):
        violations.append("source corpus reachable commit set changed")
    objects = _object_inventory(corpus)
    if (
        len(objects) != catalog.get("object_count")
        or _digest_lines(objects) != catalog.get("object_set_sha256")
    ):
        violations.append("source corpus physical object set changed")
    for excluded in catalog.get("excluded_commits") or []:
        present = _bare(corpus, "cat-file", "-e", f"{excluded}^{{commit}}", check=False)
        if present.returncode == 0:
            violations.append(f"excluded commit is present in source corpus: {excluded}")
    return violations


def corpus_has_commit(workspace: Path, commit: str) -> bool:
    corpus = workspace / CORPUS_RELATIVE
    return (
        _bare(corpus, "cat-file", "-e", f"{commit}^{{commit}}", check=False).returncode
        == 0
    )


def corpus_has_path(workspace: Path, commit: str, path: str) -> bool:
    corpus = workspace / CORPUS_RELATIVE
    return (
        _bare(corpus, "cat-file", "-e", f"{commit}:{path}", check=False).returncode
        == 0
    )


def read_catalog(workspace: Path) -> dict[str, object] | None:
    path = workspace / CATALOG_NAME
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None
