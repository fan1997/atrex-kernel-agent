from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from long_horizon.git_episode import git_head, working_changes
from long_horizon.protocol import atomic_write_json
from orchestrator.session_io import run_session

from .corpus import CORPUS_RELATIVE, read_catalog, validate_source_corpus
from .manifest import RepositoryManifest
from .runtime import link_repository_runtime


PREPLAN_ARTIFACT = PurePosixPath("plans/end_to_end_architecture_frontier.json")
PREPLAN_PROFILE_ROOT = PurePosixPath("profiles/preplan")
PREPLAN_PROMPT = Path(__file__).resolve().parent / "prompts" / "preplan.md"
REQUIRED_FAMILIES = frozenset(
    {
        "direct_native",
        "representation_transform",
        "multi_stage",
        "hybrid_dispatch",
    }
)
DECISION_VARIABLES = (
    "representation",
    "preprocessing",
    "decomposition",
    "algorithm",
    "schedule",
    "dispatch",
)
FORBIDDEN_EVIDENCE_MARKERS = (
    "private_evaluator",
    "profile_shape_id",
    ".atrex_private_profile_case",
    "repository_horizon.dev_eval",
    "shapes.json",
)
MAX_ARTIFACT_BYTES = 256 * 1024
MAX_PROFILE_FILES = 24
MAX_PROFILE_BYTES = 2 * 1024 * 1024


def _nonempty(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _nonempty_list(value: object) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(_nonempty(item) for item in value)
    )


def _object_list(value: object) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, dict) for item in value)
    )


def _require_fields(
    value: dict[str, Any], fields: tuple[str, ...], label: str
) -> list[str]:
    return [f"{label} requires {field}" for field in fields if field not in value]


def validate_preplan_artifact(path: Path) -> list[str]:
    """Validate the macro-strategy artifact without judging which route should win."""

    try:
        if path.stat().st_size > MAX_ARTIFACT_BYTES:
            return [f"preplan artifact exceeds {MAX_ARTIFACT_BYTES} bytes"]
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [f"cannot read preplan artifact: {exc}"]
    if not isinstance(document, dict):
        return ["preplan artifact must be a JSON object"]
    violations = _require_fields(
        document,
        (
            "schema_version",
            "objective",
            "constraints",
            "inherited_implementation_choices",
            "unverified_assumptions",
            "architecture_frontier",
            "probing",
            "portfolio",
        ),
        "preplan artifact",
    )
    if violations:
        return violations
    if document["schema_version"] != 1:
        violations.append("preplan artifact schema_version must be 1")

    objective = document["objective"]
    if not isinstance(objective, dict):
        violations.append("objective must be an object")
    else:
        if objective.get("metric") != "end_to_end_latency":
            violations.append("objective.metric must be end_to_end_latency")
        variables = objective.get("decision_variables")
        if not isinstance(variables, list) or set(variables) != set(
            DECISION_VARIABLES
        ):
            violations.append(
                "objective.decision_variables must cover representation, preprocessing, "
                "decomposition, algorithm, schedule, and dispatch"
            )
        if not _nonempty(objective.get("formulation")):
            violations.append(
                "objective requires a non-empty mathematical formulation"
            )

    constraints = document["constraints"]
    if not isinstance(constraints, dict):
        violations.append("constraints must be an object")
    else:
        for category in ("semantic", "interface", "policy", "hardware"):
            entries = constraints.get(category)
            if not _object_list(entries):
                violations.append(
                    f"constraints.{category} must be a non-empty object list"
                )
                continue
            for index, entry in enumerate(entries):
                if not _nonempty(entry.get("statement")) or not _nonempty(
                    entry.get("evidence")
                ):
                    violations.append(
                        f"constraints.{category}[{index}] requires statement and evidence"
                    )

    choices = document["inherited_implementation_choices"]
    if not _object_list(choices):
        violations.append(
            "inherited_implementation_choices must be a non-empty object list"
        )
    else:
        for index, choice in enumerate(choices):
            for field in ("choice", "evidence", "why_not_a_constraint"):
                if not _nonempty(choice.get(field)):
                    violations.append(
                        f"inherited_implementation_choices[{index}] requires {field}"
                    )

    assumptions = document["unverified_assumptions"]
    if not _object_list(assumptions):
        violations.append(
            "unverified_assumptions must be a non-empty object list"
        )
    else:
        seen_assumptions: set[str] = set()
        for index, assumption in enumerate(assumptions):
            for field in (
                "id",
                "statement",
                "consequence_if_false",
                "falsification_test",
            ):
                if not _nonempty(assumption.get(field)):
                    violations.append(
                        f"unverified_assumptions[{index}] requires {field}"
                    )
            identifier = assumption.get("id")
            if isinstance(identifier, str):
                if identifier in seen_assumptions:
                    violations.append(f"duplicate assumption id: {identifier}")
                seen_assumptions.add(identifier)

    routes = document["architecture_frontier"]
    route_ids: set[str] = set()
    families: set[str] = set()
    if not _object_list(routes):
        violations.append(
            "architecture_frontier must be a non-empty object list"
        )
    else:
        for index, route in enumerate(routes):
            label = f"architecture_frontier[{index}]"
            for field in (
                "id",
                "family",
                "hypothesis",
                "data_representation",
                "pipeline",
                "transformation_cost",
                "unlocked_fast_path",
                "winning_regimes",
                "losing_regimes",
                "required_mechanisms",
                "risks",
                "falsification_tests",
            ):
                if field not in route:
                    violations.append(f"{label} requires {field}")
            identifier = route.get("id")
            if not _nonempty(identifier):
                violations.append(f"{label}.id must be non-empty")
            elif identifier in route_ids:
                violations.append(
                    f"duplicate architecture route id: {identifier}"
                )
            else:
                route_ids.add(identifier)
            family = route.get("family")
            if family not in REQUIRED_FAMILIES:
                violations.append(f"{label}.family is unsupported: {family}")
            else:
                families.add(family)
            for field in (
                "hypothesis",
                "data_representation",
                "unlocked_fast_path",
            ):
                if not _nonempty(route.get(field)):
                    violations.append(f"{label}.{field} must be non-empty")
            for field in (
                "pipeline",
                "winning_regimes",
                "losing_regimes",
                "required_mechanisms",
                "risks",
            ):
                if not _nonempty_list(route.get(field)):
                    violations.append(
                        f"{label}.{field} must be a non-empty string list"
                    )
            cost = route.get("transformation_cost")
            if not isinstance(cost, dict) or cost.get("status") not in {
                "measured",
                "estimated",
                "unknown",
            }:
                violations.append(
                    f"{label}.transformation_cost requires "
                    "measured/estimated/unknown status"
                )
            elif not _nonempty(cost.get("evidence")):
                violations.append(
                    f"{label}.transformation_cost requires evidence"
                )
            tests = route.get("falsification_tests")
            if not _object_list(tests):
                violations.append(
                    f"{label}.falsification_tests must be a non-empty object list"
                )
            else:
                for test_index, test in enumerate(tests):
                    for field in (
                        "question",
                        "method",
                        "success_criterion",
                        "failure_action",
                    ):
                        if not _nonempty(test.get(field)):
                            violations.append(
                                f"{label}.falsification_tests[{test_index}] "
                                f"requires {field}"
                            )
        missing = REQUIRED_FAMILIES - families
        if missing:
            violations.append(
                "architecture_frontier misses required families: "
                + ", ".join(sorted(missing))
            )

    probing = document["probing"]
    if not isinstance(probing, dict) or not isinstance(
        probing.get("experiments"), list
    ):
        violations.append("probing.experiments must be a list")
    else:
        for index, experiment in enumerate(probing["experiments"]):
            if not isinstance(experiment, dict):
                violations.append(
                    f"probing.experiments[{index}] must be an object"
                )
                continue
            for field in (
                "id",
                "kind",
                "hypothesis",
                "status",
                "evidence",
                "interpretation",
            ):
                if not _nonempty(experiment.get(field)):
                    violations.append(
                        f"probing.experiments[{index}] requires {field}"
                    )
            if experiment.get("kind") not in {
                "static",
                "gpu_probe",
                "micro_prototype",
            }:
                violations.append(
                    f"probing.experiments[{index}].kind is unsupported"
                )
            if experiment.get("status") not in {
                "measured",
                "deferred",
                "failed",
            }:
                violations.append(
                    f"probing.experiments[{index}].status is unsupported"
                )

    portfolio = document["portfolio"]
    if not isinstance(portfolio, dict):
        violations.append("portfolio must be an object")
    else:
        ranked = portfolio.get("ranked_route_ids")
        if (
            not isinstance(ranked, list)
            or set(ranked) != route_ids
            or len(ranked) != len(route_ids)
        ):
            violations.append(
                "portfolio.ranked_route_ids must rank every route exactly once"
            )
        primary = portfolio.get("primary_route_id")
        if primary not in route_ids:
            violations.append(
                "portfolio.primary_route_id must reference a frontier route"
            )
        hedges = portfolio.get("hedge_route_ids")
        if (
            not isinstance(hedges, list)
            or not hedges
            or any(item not in route_ids for item in hedges)
        ):
            violations.append(
                "portfolio.hedge_route_ids must reference at least one frontier route"
            )
        if primary in (hedges or []):
            violations.append(
                "portfolio primary route cannot also be a hedge route"
            )
        if not _nonempty(portfolio.get("selection_rationale")):
            violations.append("portfolio requires selection_rationale")
        if not _object_list(portfolio.get("next_experiments")):
            violations.append(
                "portfolio.next_experiments must be a non-empty object list"
            )

    serialized = json.dumps(document, sort_keys=True).casefold()
    for marker in FORBIDDEN_EVIDENCE_MARKERS:
        if marker in serialized:
            violations.append(
                "preplan evidence contains forbidden private-evaluator marker: "
                + marker
            )
    return violations


def _profile_violations(workspace: Path) -> list[str]:
    root = workspace / PREPLAN_PROFILE_ROOT
    if not root.exists():
        return []
    files = [
        path for path in root.rglob("*") if path.is_file() or path.is_symlink()
    ]
    violations: list[str] = []
    if len(files) > MAX_PROFILE_FILES:
        violations.append(
            f"preplan prototypes exceed {MAX_PROFILE_FILES} files"
        )
    total = 0
    for path in files:
        if path.is_symlink():
            violations.append(
                "preplan prototype may not be a symlink: "
                + str(path.relative_to(workspace))
            )
            continue
        total += path.stat().st_size
        try:
            content = path.read_text(encoding="utf-8").casefold()
        except (UnicodeError, OSError):
            content = ""
        for marker in FORBIDDEN_EVIDENCE_MARKERS:
            if marker in content:
                violations.append(
                    "preplan prototype contains forbidden private-evaluator marker: "
                    + marker
                )
    if total > MAX_PROFILE_BYTES:
        violations.append(
            f"preplan prototypes exceed {MAX_PROFILE_BYTES} bytes"
        )
    return violations


def render_preplan_prompt(
    campaign: Any, manifest: RepositoryManifest, workspace: Path
) -> str:
    public_dev = (
        "python tools/sandbox.py --kind dev "
        f"--hardware {campaign.sandbox_hardware}"
    )
    if campaign.sandbox_profile:
        public_dev += f" --gateway-profile {campaign.sandbox_profile}"
    if campaign.sandbox_url:
        public_dev += f" --url {campaign.sandbox_url}"
    public_dev += (
        " --no-sync --input vendor/flash_attention --input kernel.py "
        "--input input.py --input reference.py -- "
        "python profiles/preplan/<driver>.py"
    )
    values = {
        "WORKSPACE": workspace,
        "PLATFORM": campaign.platform,
        "FRAMEWORK": campaign.framework,
        "SOURCE_NAME": manifest.source_name,
        "SOURCE_REVISION": manifest.revision,
        "EDITABLE_ROOTS": ", ".join(
            f"`{value}`" for value in manifest.editable_workspace_roots
        ),
        "SOURCE_CORPUS": (
            CORPUS_RELATIVE if read_catalog(workspace) is not None else "unavailable"
        ),
        "ARTIFACT": PREPLAN_ARTIFACT.as_posix(),
        "PUBLIC_DEV_COMMAND": public_dev,
        "SANDBOX": campaign._sandbox_directive(),
    }
    prompt = PREPLAN_PROMPT.read_text(encoding="utf-8")
    for key, value in values.items():
        prompt = prompt.replace("{{" + key + "}}", str(value))
    unresolved = [
        part.split("}}", 1)[0]
        for part in prompt.split("{{")[1:]
        if "}}" in part
    ]
    if unresolved:
        raise RuntimeError(
            "unresolved preplan prompt placeholders: " + ", ".join(unresolved)
        )
    return prompt


@dataclass
class PreplanRunner:
    campaign: Any
    manifest: RepositoryManifest
    timeout: int = 7200

    def run(self) -> Path:
        canonical = self.campaign.workspace
        base_commit = git_head(canonical)
        run_id = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-"
            + uuid.uuid4().hex[:8]
        )
        worktree = (
            canonical.parent
            / ".atrex_preplan_worktrees"
            / canonical.name
            / run_id
        )
        worktree.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(worktree), base_commit],
            cwd=str(canonical),
            check=True,
            capture_output=True,
            text=True,
        )
        link_repository_runtime(self.campaign, worktree, self.manifest)
        corpus = read_catalog(worktree)
        if corpus is not None:
            corpus_violations = validate_source_corpus(
                worktree,
                corpus,
                expected_runtime=canonical / CORPUS_RELATIVE,
            )
            if corpus_violations:
                raise RuntimeError(
                    "preplan source corpus is invalid: "
                    + "; ".join(corpus_violations)
                )

        prompt = render_preplan_prompt(self.campaign, self.manifest, worktree)
        print(
            "[repository-horizon] starting PREPLAN-only session "
            f"worktree={worktree}",
            flush=True,
        )
        result = run_session(
            worktree,
            prompt,
            timeout=self.timeout,
            agent_cli=self.campaign.agent_cli,
            sandbox_hardware=self.campaign.sandbox_hardware,
            sandbox_profile=self.campaign.sandbox_profile,
            sandbox_url=self.campaign.sandbox_url,
            sandbox_timeout=self.campaign.sandbox_timeout,
            reasoning_effort="max",
            extra_environment=self.campaign.agent_environment(),
        )

        violations: list[str] = []
        if result.timed_out:
            violations.append("preplan session timed out")
        if result.exit_status != 0:
            violations.append(
                f"preplan session exited with status {result.exit_status}"
            )
        if git_head(worktree) != base_commit:
            violations.append(
                "preplan changed Git HEAD; implementation commits are forbidden"
            )
        dirty = working_changes(worktree)
        if dirty:
            violations.append(
                "preplan modified campaign source: " + ", ".join(dirty[:12])
            )
        artifact = worktree / PREPLAN_ARTIFACT
        violations.extend(validate_preplan_artifact(artifact))
        violations.extend(_profile_violations(worktree))
        if violations:
            raise RuntimeError(
                "PREPLAN REJECTED: "
                + "; ".join(violations)
                + f"; forensic_worktree={worktree}; "
                + f"stderr={result.stderr_tail[-1000:]}"
            )

        canonical_artifact = canonical / PREPLAN_ARTIFACT
        canonical_artifact.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(artifact, canonical_artifact)
        digest = hashlib.sha256(canonical_artifact.read_bytes()).hexdigest()
        atomic_write_json(
            canonical / "plans" / "preplan_run.json",
            {
                "schema_version": 1,
                "status": "PASS",
                "source_revision": self.manifest.revision,
                "incumbent_commit": base_commit,
                "artifact": PREPLAN_ARTIFACT.as_posix(),
                "artifact_sha256": digest,
                "session_id": result.session_id,
                "tokens": result.tokens,
                "worktree": str(worktree),
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "formal_episode_started": False,
                "candidate_created": False,
            },
        )
        print(
            f"[repository-horizon] PREPLAN PASS artifact={canonical_artifact} "
            f"sha256={digest} formal_episodes=0",
            flush=True,
        )
        return canonical_artifact
