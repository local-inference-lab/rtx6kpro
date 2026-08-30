#!/usr/bin/env python3
"""Validate and render Local Inference Lab Docker image publication records."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse

SCHEMA_VERSION = 4
WIKI_REPOSITORY = "https://github.com/local-inference-lab/rtx6kpro"
UNKNOWN = "UNKNOWN — needs verification"
NOT_TESTED = "Not tested"
NA = "N/A"

RELEASE_CLASSES = {"experimental", "community-derivative", "official"}
DISTRIBUTION_ROLES = {"recommended", "custom"}
QUALIFICATION_STATUSES = {"implemented", "qualified", "research-only", "unsupported"}
MAINTENANCE_STATUSES = {"maintainer-supported", "author-supported", "ephemeral", "superseded"}
THREAD_STATUSES = {"active", "superseded"}
SUPPORT_COMMITMENTS = {"maintained", "ephemeral", "superseded"}
BOT_LISTINGS = {"listed", "not-listed", "not-applicable"}
VALIDATION_STATUSES = {"passed", "failed", "not-tested"}
VALIDATION_CATEGORIES = {"smoke", "correctness", "stability", "performance", "other"}
EVIDENCE_CLASSES = {"exploratory", "confirmatory", "qualification-evidence"}

SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
GIT_RE = re.compile(r"^[0-9a-f]{40}$")
IMAGE_REF_RE = re.compile(r"^.+@sha256:[0-9a-f]{64}$")
PLACEHOLDER_RE = re.compile(r"(?i)\b(?:required|optional|todo|tbd|change[-_ ]?me|replace[-_ ]?me)\b")
SECRET_RE = re.compile(
    r"(?i)(?:bearer\s+[A-Za-z0-9._~+/-]{12,}|"
    r"(?:api[_-]?key|access[_-]?token|secret|password)\s*[:=]\s*[^\s]+|"
    r"(?:ghp_|github_pat_|sk-)[A-Za-z0-9_-]{12,})"
)
PERSONAL_PATH_RE = re.compile(
    r"(?i)(?:^|[\s`'\"])(?:/home/[^/\s]+/|/Users/[^/\s]+/|/root/|/mnt/[^/\s]+/|[A-Z]:\\Users\\[^\\\s]+\\)"
)
PRIVATE_NETWORK_RE = re.compile(
    r"(?i)(?:https?://)?(?:"
    r"10(?:\.\d{1,3}){3}|"
    r"192\.168(?:\.\d{1,3}){2}|"
    r"172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2}|"
    r"[A-Za-z0-9-]+\.(?:internal|lan|local)(?::\d+)?"
    r")"
)


@dataclass(frozen=True)
class Issue:
    severity: str
    path: str
    message: str

    def render(self) -> str:
        return f"{self.severity.upper():7} {self.path}: {self.message}"


class Report:
    def __init__(self, strict: bool = False) -> None:
        self.strict = strict
        self.issues: list[Issue] = []

    def error(self, path: str, message: str) -> None:
        self.issues.append(Issue("error", path, message))

    def warning(self, path: str, message: str) -> None:
        self.issues.append(Issue("error" if self.strict else "warning", path, message))

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "warning"]


def load_record(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(data, dict):
        raise ValueError("record must be a JSON object")
    return data


def iter_strings(value: Any, path: str = "$") -> Iterable[tuple[str, str]]:
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, Mapping):
        for key, child in value.items():
            yield from iter_strings(child, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            yield from iter_strings(child, f"{path}[{index}]")


def is_unknown(value: Any) -> bool:
    return isinstance(value, str) and value.strip() == UNKNOWN


def is_na(value: Any) -> bool:
    return isinstance(value, str) and value.strip() == NA


def is_not_tested(value: Any) -> bool:
    return isinstance(value, str) and value.strip() == NOT_TESTED


def is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip() or bool(PLACEHOLDER_RE.search(value))
    if isinstance(value, (list, dict, tuple)):
        return len(value) == 0
    return False


def get_map(value: Any, report: Report, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        report.error(path, "object is required")
        return {}
    return value


def get_list(value: Any, report: Report, path: str) -> list[Any]:
    if not isinstance(value, list):
        report.error(path, "array is required")
        return []
    return value


def require_value(
    value: Any,
    report: Report,
    path: str,
    *,
    allow_na: bool = False,
    allow_not_tested: bool = False,
) -> None:
    if is_empty(value):
        report.error(path, "value is required")
        return
    if is_unknown(value):
        report.warning(path, f"replace exact marker {UNKNOWN!r} before publication")
    if is_na(value) and not allow_na:
        report.error(path, "N/A is not valid for this required field")
    if is_not_tested(value) and not allow_not_tested:
        report.error(path, "Not tested is not valid for this field")


def require_enum(value: Any, allowed: set[str], report: Report, path: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        report.error(path, f"must be one of: {', '.join(sorted(allowed))}")
        return ""
    return value


def is_public_url(value: Any) -> bool:
    if not isinstance(value, str) or is_unknown(value) or is_na(value) or is_not_tested(value):
        return False
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return False
    host = (parsed.hostname or "").lower()
    if host in {"localhost", "127.0.0.1", "::1"}:
        return False
    if PRIVATE_NETWORK_RE.search(value):
        return False
    return True


def validate_image_identity(obj: dict[str, Any], report: Report, path: str) -> None:
    repository = obj.get("repository")
    tag = obj.get("tag")
    digest = obj.get("digest")
    reference = obj.get("reference")
    for key, value in (("repository", repository), ("tag", tag), ("digest", digest), ("reference", reference)):
        require_value(value, report, f"{path}.{key}")
    if isinstance(digest, str) and not is_unknown(digest) and not SHA256_RE.fullmatch(digest):
        report.error(f"{path}.digest", "must be sha256:<64 lowercase hex>")
    if isinstance(reference, str) and not is_unknown(reference) and not IMAGE_REF_RE.fullmatch(reference):
        report.error(f"{path}.reference", "must be an immutable image@sha256:<digest> reference")
    if all(isinstance(x, str) and not is_unknown(x) for x in (repository, tag, digest, reference)):
        expected = f"{repository}:{tag}@{digest}"
        if reference != expected:
            report.error(f"{path}.reference", f"must equal {expected!r}")


def validate_community_wiki(value: Any, report: Report) -> None:
    obj = get_map(value, report, "community_wiki")
    if not obj:
        return
    repository = obj.get("repository")
    commit = obj.get("commit")
    runbook_path = obj.get("runbook_path")
    runbook_url = obj.get("runbook_url")
    relationship = obj.get("relationship")

    for field, field_value in (
        ("repository", repository),
        ("commit", commit),
        ("runbook_path", runbook_path),
        ("runbook_url", runbook_url),
        ("relationship", relationship),
    ):
        require_value(field_value, report, f"community_wiki.{field}")

    if isinstance(repository, str) and not is_unknown(repository):
        if repository.rstrip("/") != WIKI_REPOSITORY:
            report.error(
                "community_wiki.repository",
                f"must identify the community wiki {WIKI_REPOSITORY}",
            )

    if isinstance(commit, str) and not is_unknown(commit) and not GIT_RE.fullmatch(commit):
        report.error("community_wiki.commit", "full 40-character wiki commit is required")

    if isinstance(runbook_path, str) and not is_unknown(runbook_path):
        if (
            runbook_path.startswith("/")
            or ".." in Path(runbook_path).parts
            or not runbook_path.endswith(".md")
        ):
            report.error(
                "community_wiki.runbook_path",
                "use a repository-relative Markdown path without parent traversal",
            )

    if (
        isinstance(commit, str)
        and not is_unknown(commit)
        and isinstance(runbook_path, str)
        and not is_unknown(runbook_path)
        and isinstance(runbook_url, str)
        and not is_unknown(runbook_url)
    ):
        expected = f"{WIKI_REPOSITORY}/blob/{commit}/{runbook_path}"
        if not (runbook_url == expected or runbook_url.startswith(expected + "#")):
            report.error(
                "community_wiki.runbook_url",
                f"must be a commit-pinned URL beginning {expected!r}",
            )


def validate_component(component: Any, report: Report, path: str) -> None:
    obj = get_map(component, report, path)
    if not obj:
        return
    name = obj.get("name")
    require_value(name, report, f"{path}.name")
    repository = obj.get("repository")
    commit = obj.get("commit")
    release_reference = obj.get("release_reference")
    if is_na(repository) and is_na(commit) and is_na(release_reference):
        return
    require_value(repository, report, f"{path}.repository")
    if isinstance(repository, str) and not is_unknown(repository) and not is_public_url(repository):
        report.error(f"{path}.repository", "public repository URL is required")
    commit_valid = isinstance(commit, str) and bool(GIT_RE.fullmatch(commit))
    release_valid = isinstance(release_reference, str) and not any(
        (is_empty(release_reference), is_unknown(release_reference), is_na(release_reference))
    )
    if not commit_valid and not release_valid:
        if is_unknown(commit) or is_unknown(release_reference):
            report.warning(
                f"{path}.commit",
                "verify a full 40-character commit or a release_reference",
            )
        else:
            report.error(
                f"{path}.commit",
                "provide a full 40-character commit or a verified release_reference",
            )

    for key in ("pull_requests", "patches", "overlays"):
        items = get_list(obj.get(key, []), report, f"{path}.{key}")
        for index, item in enumerate(items):
            child = get_map(item, report, f"{path}.{key}[{index}]")
            if not child:
                continue
            if all(is_na(v) for v in child.values() if not isinstance(v, list)):
                continue
            if key == "pull_requests":
                url = child.get("url")
                head = child.get("head_commit")
                require_value(url, report, f"{path}.{key}[{index}].url")
                if isinstance(url, str) and not is_unknown(url) and not is_public_url(url):
                    report.error(f"{path}.{key}[{index}].url", "public PR URL is required")
                if not (isinstance(head, str) and GIT_RE.fullmatch(head)):
                    if is_unknown(head):
                        report.warning(f"{path}.{key}[{index}].head_commit", "verify the full PR head commit")
                    else:
                        report.error(f"{path}.{key}[{index}].head_commit", "full PR head commit is required")
                require_value(child.get("title"), report, f"{path}.{key}[{index}].title")
                authors = get_list(child.get("authors"), report, f"{path}.{key}[{index}].authors")
                if not authors or all(is_na(a) for a in authors):
                    report.error(f"{path}.{key}[{index}].authors", "human authors are required")
            elif key == "patches":
                require_value(child.get("path_or_url"), report, f"{path}.{key}[{index}].path_or_url")
                digest = child.get("sha256")
                if not (isinstance(digest, str) and SHA256_RE.fullmatch(digest)):
                    if is_unknown(digest):
                        report.warning(f"{path}.{key}[{index}].sha256", "verify the patch SHA-256")
                    else:
                        report.error(f"{path}.{key}[{index}].sha256", "patch SHA-256 is required")
                require_value(child.get("purpose"), report, f"{path}.{key}[{index}].purpose")
                authors = get_list(child.get("authors"), report, f"{path}.{key}[{index}].authors")
                if not authors or all(is_na(a) for a in authors):
                    report.error(f"{path}.{key}[{index}].authors", "human authors are required")
            else:
                for field in ("source", "destination", "old_sha256", "new_sha256", "purpose"):
                    value = child.get(field)
                    require_value(value, report, f"{path}.{key}[{index}].{field}")
                for field in ("old_sha256", "new_sha256"):
                    value = child.get(field)
                    if not (isinstance(value, str) and SHA256_RE.fullmatch(value)):
                        if is_unknown(value):
                            report.warning(f"{path}.{key}[{index}].{field}", "verify the SHA-256")
                        else:
                            report.error(f"{path}.{key}[{index}].{field}", "SHA-256 is required")


def validate_tested_configuration(config: Any, report: Report, path: str) -> None:
    obj = get_map(config, report, path)
    fields = (
        "name", "hardware", "topology", "power_and_clocks", "driver", "cuda_runtime",
        "pytorch", "nccl", "engine_source", "model_revision", "quantization",
        "parallelism", "kv_cache", "speculative_mode", "graph_mode",
        "scheduler_limits", "cache_policy", "launch_command",
    )
    for field in fields:
        require_value(obj.get(field), report, f"{path}.{field}", allow_not_tested=True)


def validate_validation(record: dict[str, Any], report: Report) -> None:
    validation = get_map(record.get("validation"), report, "validation")
    commands = get_list(validation.get("commands"), report, "validation.commands")
    if not commands:
        report.error("validation.commands", "at least one command or exact Not tested marker is required")
    for index, command in enumerate(commands):
        require_value(command, report, f"validation.commands[{index}]", allow_not_tested=True)

    results = get_list(validation.get("results"), report, "validation.results")
    if not results:
        report.error("validation.results", "at least one validation result is required")
    passed_categories: set[str] = set()
    for index, item in enumerate(results):
        path = f"validation.results[{index}]"
        obj = get_map(item, report, path)
        require_value(obj.get("name"), report, f"{path}.name")
        category = require_enum(obj.get("category"), VALIDATION_CATEGORIES, report, f"{path}.category")
        status = require_enum(obj.get("status"), VALIDATION_STATUSES, report, f"{path}.status")
        for field in ("conditions", "measurement", "result", "conclusion"):
            require_value(obj.get(field), report, f"{path}.{field}", allow_not_tested=status == "not-tested")
        evidence_url = obj.get("evidence_url")
        evidence_sha = obj.get("evidence_sha256")
        if status == "passed":
            passed_categories.add(category)
            if not is_public_url(evidence_url):
                if is_unknown(evidence_url):
                    report.warning(f"{path}.evidence_url", "verify the public evidence URL")
                else:
                    report.error(f"{path}.evidence_url", "passed result requires a public evidence URL")
            if not (isinstance(evidence_sha, str) and SHA256_RE.fullmatch(evidence_sha)):
                if is_unknown(evidence_sha):
                    report.warning(f"{path}.evidence_sha256", "verify the evidence SHA-256")
                else:
                    report.error(f"{path}.evidence_sha256", "passed result requires evidence SHA-256")
        elif status == "not-tested":
            if obj.get("result") != NOT_TESTED:
                report.error(f"{path}.result", f"not-tested result must be exactly {NOT_TESTED!r}")

    performance_claims = get_list(
        validation.get("performance_claims"), report, "validation.performance_claims"
    )
    qualification = record.get("qualification_status")
    actual_claims: list[dict[str, Any]] = []
    for index, item in enumerate(performance_claims):
        path = f"validation.performance_claims[{index}]"
        obj = get_map(item, report, path)
        if not obj or is_na(obj.get("name")):
            continue
        actual_claims.append(obj)
        required = (
            "name", "candidate_image", "baseline_image", "benchmark_repository",
            "benchmark_commit", "candidate_command", "baseline_command", "hardware",
            "model_revision", "concurrency", "input_lengths", "output_length_or_duration",
            "experimental_unit", "run_order", "stopping_rule", "exclusions",
            "falsification_condition", "repeated_control_variation", "aggregation",
            "absolute_effect", "relative_effect", "uncertainty", "raw_results_url",
            "raw_results_sha256", "result", "conclusion",
        )
        for field in required:
            require_value(
                obj.get(field),
                report,
                f"{path}.{field}",
                allow_not_tested=(
                    field == "repeated_control_variation" and qualification != "qualified"
                ),
            )
        evidence_class = require_enum(
            obj.get("evidence_class"), EVIDENCE_CLASSES, report, f"{path}.evidence_class"
        )
        for field in ("candidate_image", "baseline_image"):
            value = obj.get(field)
            if not (isinstance(value, str) and IMAGE_REF_RE.fullmatch(value)):
                if is_unknown(value):
                    report.warning(f"{path}.{field}", "verify the digest-pinned image reference")
                else:
                    report.error(f"{path}.{field}", "digest-pinned image reference is required")
        commit = obj.get("benchmark_commit")
        if not (isinstance(commit, str) and GIT_RE.fullmatch(commit)):
            if is_unknown(commit):
                report.warning(f"{path}.benchmark_commit", "verify the full benchmark commit")
            else:
                report.error(f"{path}.benchmark_commit", "full benchmark commit is required")
        benchmark_repository = obj.get("benchmark_repository")
        if not is_public_url(benchmark_repository):
            if is_unknown(benchmark_repository):
                report.warning(f"{path}.benchmark_repository", "verify the public benchmark repository URL")
            else:
                report.error(f"{path}.benchmark_repository", "public benchmark repository URL is required")
        raw_results_url = obj.get("raw_results_url")
        if not is_public_url(raw_results_url):
            if is_unknown(raw_results_url):
                report.warning(f"{path}.raw_results_url", "verify the public raw-results URL")
            else:
                report.error(f"{path}.raw_results_url", "public raw-results URL is required")
        raw_sha = obj.get("raw_results_sha256")
        if not (isinstance(raw_sha, str) and SHA256_RE.fullmatch(raw_sha)):
            if is_unknown(raw_sha):
                report.warning(f"{path}.raw_results_sha256", "verify the raw-results SHA-256")
            else:
                report.error(f"{path}.raw_results_sha256", "raw-results SHA-256 is required")
        runs = obj.get("runs")
        if not (isinstance(runs, int) and runs >= 1):
            report.error(f"{path}.runs", "positive integer run count is required")
        for field, message in (
            ("changed_variables", "list every deliberately changed variable"),
            ("nuisance_variables", "list material nuisance variables"),
            ("rival_explanations", "list the rival explanations reviewed"),
        ):
            values = get_list(obj.get(field), report, f"{path}.{field}")
            if not values:
                report.error(f"{path}.{field}", message)
            for value_index, value in enumerate(values):
                require_value(value, report, f"{path}.{field}[{value_index}]")
        if qualification == "qualified" and evidence_class != "qualification-evidence":
            report.error(
                f"{path}.evidence_class",
                "qualified image performance claims must use qualification-evidence",
            )

    distribution = record.get("distribution_role")
    if qualification == "qualified" and any(
        isinstance(command, str) and command == NOT_TESTED for command in commands
    ):
        report.error("validation.commands", "qualified records cannot contain Not tested commands")
    if distribution == "recommended":
        missing_categories = {"correctness", "stability", "performance"} - passed_categories
        if missing_categories:
            report.error(
                "validation.results",
                "recommended image requires passed correctness, stability, and performance categories; missing "
                + ", ".join(sorted(missing_categories)),
            )
    if actual_claims and "performance" not in passed_categories:
        report.error("validation.results", "performance claims require a passed performance result")


def validate_record(record: dict[str, Any], strict: bool = False) -> Report:
    report = Report(strict=strict)
    required_top = {
        "schema_version", "record_id", "title", "summary", "model_family",
        "release_class", "distribution_role", "qualification_status", "maintenance_status",
        "image", "base_image", "recommended_image", "community_wiki", "build", "changes",
        "tested_configurations", "validation", "limitations", "support", "publication",
    }
    for key in sorted(required_top - set(record)):
        report.error(key, "required top-level field is missing")

    if record.get("schema_version") != SCHEMA_VERSION:
        report.error("schema_version", f"must equal {SCHEMA_VERSION}")
    for field in ("record_id", "title", "summary", "model_family"):
        require_value(record.get(field), report, field)

    release = require_enum(record.get("release_class"), RELEASE_CLASSES, report, "release_class")
    distribution = require_enum(
        record.get("distribution_role"), DISTRIBUTION_ROLES, report, "distribution_role"
    )
    qualification = require_enum(
        record.get("qualification_status"), QUALIFICATION_STATUSES, report, "qualification_status"
    )
    maintenance = require_enum(
        record.get("maintenance_status"), MAINTENANCE_STATUSES, report, "maintenance_status"
    )

    image = get_map(record.get("image"), report, "image")
    base = get_map(record.get("base_image"), report, "base_image")
    validate_image_identity(image, report, "image")
    validate_image_identity(base, report, "base_image")
    require_value(base.get("credit"), report, "base_image.credit")

    recommended = get_map(record.get("recommended_image"), report, "recommended_image")
    recommended_ref = recommended.get("reference")
    require_value(recommended_ref, report, "recommended_image.reference")
    if isinstance(recommended_ref, str) and not is_unknown(recommended_ref) and not IMAGE_REF_RE.fullmatch(recommended_ref):
        report.error("recommended_image.reference", "digest-pinned recommended image reference is required")

    validate_community_wiki(record.get("community_wiki"), report)

    build = get_map(record.get("build"), report, "build")
    recipe_url = build.get("recipe_url")
    if not is_public_url(recipe_url):
        if is_unknown(recipe_url):
            report.warning("build.recipe_url", "public recipe URL must be verified")
        else:
            report.error("build.recipe_url", "public Dockerfile or build-script URL is required")
    recipe_commit = build.get("recipe_commit")
    if not (isinstance(recipe_commit, str) and GIT_RE.fullmatch(recipe_commit)):
        if is_unknown(recipe_commit):
            report.warning("build.recipe_commit", "full recipe commit must be verified")
        else:
            report.error("build.recipe_commit", "full 40-character recipe commit is required")
    require_value(build.get("build_command"), report, "build.build_command")
    components = get_list(build.get("components"), report, "build.components")
    if not components:
        report.error("build.components", "at least one source component is required")
    for index, component in enumerate(components):
        validate_component(component, report, f"build.components[{index}]")
    for field in ("package_changes", "build_arguments", "environment_defaults", "entrypoint_changes"):
        items = get_list(build.get(field), report, f"build.{field}")
        if not items:
            report.error(f"build.{field}", "use N/A rather than omitting this field")
    for field, pattern in (("result_tree", GIT_RE), ("integration_patch_sha256", SHA256_RE)):
        value = build.get(field)
        require_value(value, report, f"build.{field}", allow_na=True)
        if isinstance(value, str) and not is_na(value) and not is_unknown(value) and not pattern.fullmatch(value):
            report.error(f"build.{field}", "identity has the wrong format")

    changes = get_map(record.get("changes"), report, "changes")
    for field in ("inherited", "introduced", "compatibility_impact"):
        values = get_list(changes.get(field), report, f"changes.{field}")
        if not values:
            report.error(f"changes.{field}", "at least one explicit statement is required")
        for index, value in enumerate(values):
            require_value(value, report, f"changes.{field}[{index}]", allow_na=field == "compatibility_impact")

    configs = get_list(record.get("tested_configurations"), report, "tested_configurations")
    if not configs:
        report.error("tested_configurations", "at least one tested or Not tested configuration is required")
    for index, config in enumerate(configs):
        validate_tested_configuration(config, report, f"tested_configurations[{index}]")

    validate_validation(record, report)

    limitations = get_map(record.get("limitations"), report, "limitations")
    for field in ("known", "untested", "unsupported"):
        values = get_list(limitations.get(field), report, f"limitations.{field}")
        if not values:
            report.error(f"limitations.{field}", "use N/A or an explicit statement rather than omitting")
        for index, value in enumerate(values):
            require_value(
                value,
                report,
                f"limitations.{field}[{index}]",
                allow_na=True,
                allow_not_tested=field == "untested",
            )

    support = get_map(record.get("support"), report, "support")
    for field in ("owner", "contact", "support_thread", "triage_policy"):
        require_value(support.get(field), report, f"support.{field}")
    issue_tracker = support.get("issue_tracker")
    require_value(issue_tracker, report, "support.issue_tracker", allow_na=True)
    if not is_na(issue_tracker) and not is_unknown(issue_tracker) and not is_public_url(issue_tracker):
        report.error("support.issue_tracker", "public issue-tracker URL or N/A is required")
    support_thread = support.get("support_thread")
    if isinstance(support_thread, str) and not is_unknown(support_thread) and not is_public_url(support_thread):
        report.error("support.support_thread", "public support-thread URL is required")
    thread_status = require_enum(support.get("thread_status"), THREAD_STATUSES, report, "support.thread_status")
    commitment = require_enum(
        support.get("support_commitment"), SUPPORT_COMMITMENTS, report, "support.support_commitment"
    )
    superseded_by = support.get("superseded_by")
    require_value(superseded_by, report, "support.superseded_by", allow_na=True)

    publication = get_map(record.get("publication"), report, "publication")
    record_url = publication.get("record_url")
    if not is_public_url(record_url):
        if is_unknown(record_url):
            report.warning("publication.record_url", "public machine-readable record URL is required")
        else:
            report.error("publication.record_url", "public machine-readable record URL is required")
    main_link = publication.get("main_channel_link")
    require_value(main_link, report, "publication.main_channel_link", allow_na=True)
    count = publication.get("main_channel_link_count")
    if not isinstance(count, int) or count not in {0, 1}:
        report.error("publication.main_channel_link_count", "must be 0 or 1")
    if count == 1 and (is_na(main_link) or not is_public_url(main_link)):
        report.error("publication.main_channel_link", "one posted link requires its public message URL")
    bot_listing = require_enum(publication.get("bot_listing"), BOT_LISTINGS, report, "publication.bot_listing")
    approval = publication.get("maintainer_approval_url")
    require_value(approval, report, "publication.maintainer_approval_url", allow_na=True)

    if distribution == "recommended":
        if release != "official":
            report.error("release_class", "recommended image must use official release class")
        if qualification != "qualified":
            report.error("qualification_status", "recommended image must be qualified")
        if maintenance != "maintainer-supported":
            report.error("maintenance_status", "recommended image must be maintainer-supported")
        if commitment != "maintained":
            report.error("support.support_commitment", "recommended image must use maintained commitment")
        if recommended_ref != image.get("reference"):
            report.error("recommended_image.reference", "recommended record must identify itself as the recommended image")
        if bot_listing != "listed":
            report.error("publication.bot_listing", "recommended image must be listed")
        if not is_public_url(approval):
            report.error("publication.maintainer_approval_url", "recommended image needs public maintainer approval")
        if count != 1:
            report.error("publication.main_channel_link_count", "recommended image must have one main announcement")
    elif distribution == "custom":
        if release == "official":
            report.error("release_class", "custom image cannot claim official release class")
        if maintenance not in {"ephemeral", "author-supported", "superseded"}:
            report.error("maintenance_status", "custom image must be ephemeral, author-supported, or superseded")
        expected_commitment = {
            "ephemeral": "ephemeral",
            "author-supported": "maintained",
            "superseded": "superseded",
        }.get(maintenance)
        if expected_commitment and commitment != expected_commitment:
            report.error(
                "support.support_commitment",
                f"{maintenance} custom image requires {expected_commitment} commitment",
            )
        if bot_listing != "not-applicable":
            report.error("publication.bot_listing", "custom images are not bot-listed recommended images")
        if not is_na(approval):
            report.error("publication.maintainer_approval_url", "custom image should use N/A for maintainer approval")

    if maintenance == "superseded":
        if thread_status != "superseded":
            report.error("support.thread_status", "superseded image must mark its thread superseded")
        if not (isinstance(superseded_by, str) and IMAGE_REF_RE.fullmatch(superseded_by)):
            if is_unknown(superseded_by):
                report.warning("support.superseded_by", "verify the replacing recommended digest")
            else:
                report.error("support.superseded_by", "superseded image needs the replacing recommended digest")
    elif thread_status == "superseded":
        report.error("support.thread_status", "active image cannot have a superseded thread")

    for path, text in iter_strings(record):
        if SECRET_RE.search(text):
            report.error(path, "possible credential or token detected")
        if PERSONAL_PATH_RE.search(text):
            report.error(path, "personal or non-portable filesystem path detected")
        if PRIVATE_NETWORK_RE.search(text):
            report.error(path, "private hostname or network address detected")

    return report


def bullet_list(values: Any) -> str:
    if not isinstance(values, list) or not values:
        return f"- {NA}"
    return "\n".join(f"- {value}" for value in values)


def scalar(value: Any) -> str:
    if value is None or value == "":
        return UNKNOWN
    return str(value)


def render_components(components: Any) -> str:
    if not isinstance(components, list) or not components:
        return f"- {NA}"
    lines: list[str] = []
    for component in components:
        if not isinstance(component, dict):
            continue
        lines.append(
            f"- **{scalar(component.get('name'))}:** {scalar(component.get('repository'))} "
            f"@ `{scalar(component.get('commit'))}`; release `{scalar(component.get('release_reference'))}`"
        )
        for pr in component.get("pull_requests", []) or []:
            if isinstance(pr, dict) and not is_na(pr.get("url")):
                lines.append(
                    f"  - PR: {scalar(pr.get('url'))} @ `{scalar(pr.get('head_commit'))}` — "
                    f"{scalar(pr.get('title'))}; authors: {', '.join(map(str, pr.get('authors', [])))}"
                )
        for patch in component.get("patches", []) or []:
            if isinstance(patch, dict) and not is_na(patch.get("path_or_url")):
                lines.append(
                    f"  - Patch: {scalar(patch.get('path_or_url'))} — `{scalar(patch.get('sha256'))}` — "
                    f"{scalar(patch.get('purpose'))}; authors: {', '.join(map(str, patch.get('authors', [])))}"
                )
        for overlay in component.get("overlays", []) or []:
            if isinstance(overlay, dict) and not is_na(overlay.get("source")):
                lines.append(
                    f"  - Overlay: {scalar(overlay.get('source'))} → {scalar(overlay.get('destination'))}; "
                    f"{scalar(overlay.get('old_sha256'))} → {scalar(overlay.get('new_sha256'))}; "
                    f"{scalar(overlay.get('purpose'))}"
                )
    return "\n".join(lines) or f"- {NA}"


def render_configurations(configs: Any) -> str:
    if not isinstance(configs, list) or not configs:
        return f"- {NA}"
    sections: list[str] = []
    for config in configs:
        if not isinstance(config, dict):
            continue
        sections.append(
            "\n".join(
                [
                    f"### {scalar(config.get('name'))}",
                    f"- Hardware: {scalar(config.get('hardware'))}",
                    f"- Topology: {scalar(config.get('topology'))}",
                    f"- Power/clocks: {scalar(config.get('power_and_clocks'))}",
                    f"- Driver/runtime: {scalar(config.get('driver'))}; CUDA {scalar(config.get('cuda_runtime'))}; "
                    f"PyTorch {scalar(config.get('pytorch'))}; NCCL {scalar(config.get('nccl'))}",
                    f"- Engine: {scalar(config.get('engine_source'))}",
                    f"- Model/quant: {scalar(config.get('model_revision'))}; {scalar(config.get('quantization'))}",
                    f"- Parallelism: {scalar(config.get('parallelism'))}",
                    f"- KV/speculation: {scalar(config.get('kv_cache'))}; {scalar(config.get('speculative_mode'))}",
                    f"- Graph/scheduler: {scalar(config.get('graph_mode'))}; {scalar(config.get('scheduler_limits'))}",
                    f"- Cache/JIT: {scalar(config.get('cache_policy'))}",
                    "- Launch command:",
                    "```bash",
                    scalar(config.get("launch_command")),
                    "```",
                ]
            )
        )
    return "\n\n".join(sections) or f"- {NA}"


def render_validation(validation: dict[str, Any]) -> str:
    commands = validation.get("commands", [])
    command_text = "\n".join(f"- `{command}`" for command in commands) if commands else f"- {NA}"
    rows: list[str] = []
    for item in validation.get("results", []) or []:
        if not isinstance(item, dict):
            continue
        rows.append(
            f"- **{scalar(item.get('category'))} / {scalar(item.get('name'))}:** "
            f"{scalar(item.get('status'))}. Conditions: {scalar(item.get('conditions'))}. "
            f"Measurement: {scalar(item.get('measurement'))}. Result: {scalar(item.get('result'))}. "
            f"Conclusion: {scalar(item.get('conclusion'))}. Evidence: {scalar(item.get('evidence_url'))} "
            f"(`{scalar(item.get('evidence_sha256'))}`)."
        )
    return f"### Commands\n{command_text}\n\n### Results\n" + ("\n".join(rows) or f"- {NA}")


def render_performance(claims: Any) -> str:
    actual = [c for c in claims or [] if isinstance(c, dict) and not is_na(c.get("name"))]
    if not actual:
        return f"- {NA}"
    sections: list[str] = []
    for claim in actual:
        sections.append(
            "\n".join(
                [
                    f"### {scalar(claim.get('name'))}",
                    f"- Candidate: `{scalar(claim.get('candidate_image'))}`",
                    f"- Control: `{scalar(claim.get('baseline_image'))}`",
                    f"- Benchmark: {scalar(claim.get('benchmark_repository'))} @ `{scalar(claim.get('benchmark_commit'))}`",
                    f"- Hardware/model: {scalar(claim.get('hardware'))}; {scalar(claim.get('model_revision'))}",
                    f"- Concurrency/lengths: {scalar(claim.get('concurrency'))}; {scalar(claim.get('input_lengths'))}; "
                    f"{scalar(claim.get('output_length_or_duration'))}",
                    f"- Evidence class: `{scalar(claim.get('evidence_class'))}`",
                    f"- Experimental unit: {scalar(claim.get('experimental_unit'))}",
                    f"- Independent repetitions: {scalar(claim.get('runs'))}",
                    f"- Run order: {scalar(claim.get('run_order'))}",
                    f"- Stopping rule: {scalar(claim.get('stopping_rule'))}",
                    f"- Exclusions: {scalar(claim.get('exclusions'))}",
                    f"- Aggregation: {scalar(claim.get('aggregation'))}",
                    f"- Changed variables: {', '.join(map(str, claim.get('changed_variables', [])))}",
                    f"- Nuisance variables: {', '.join(map(str, claim.get('nuisance_variables', [])))}",
                    f"- Rival explanations: {', '.join(map(str, claim.get('rival_explanations', [])))}",
                    f"- Falsification condition: {scalar(claim.get('falsification_condition'))}",
                    f"- Repeated-control variation: {scalar(claim.get('repeated_control_variation'))}",
                    f"- Absolute effect: {scalar(claim.get('absolute_effect'))}",
                    f"- Relative effect: {scalar(claim.get('relative_effect'))}",
                    f"- Uncertainty: {scalar(claim.get('uncertainty'))}",
                    "- Candidate command:",
                    "```bash",
                    scalar(claim.get("candidate_command")),
                    "```",
                    "- Control command:",
                    "```bash",
                    scalar(claim.get("baseline_command")),
                    "```",
                    f"- Raw results: {scalar(claim.get('raw_results_url'))} (`{scalar(claim.get('raw_results_sha256'))}`)",
                    f"- Result: {scalar(claim.get('result'))}",
                    f"- Conclusion: {scalar(claim.get('conclusion'))}",
                ]
            )
        )
    return "\n\n".join(sections)


def render_thread(record: dict[str, Any]) -> str:
    image = record.get("image", {})
    base = record.get("base_image", {})
    community_wiki = record.get("community_wiki", {})
    build = record.get("build", {})
    changes = record.get("changes", {})
    validation = record.get("validation", {})
    limitations = record.get("limitations", {})
    support = record.get("support", {})
    publication = record.get("publication", {})
    return f"""# {scalar(record.get('title'))}

{scalar(record.get('summary'))}

## Identity and status

- Release class: `{scalar(record.get('release_class'))}`
- Distribution role: `{scalar(record.get('distribution_role'))}`
- Qualification: `{scalar(record.get('qualification_status'))}`
- Maintenance: `{scalar(record.get('maintenance_status'))}`
- Model family: {scalar(record.get('model_family'))}
- Image and digest: `{scalar(image.get('reference'))}`
- Recommended image/control: `{scalar(record.get('recommended_image', {}).get('reference'))}`

## Community runbook

- Wiki: {scalar(community_wiki.get('repository'))} @ `{scalar(community_wiki.get('commit'))}`
- Runbook: [{scalar(community_wiki.get('runbook_path'))}]({scalar(community_wiki.get('runbook_url'))})
- Relationship: {scalar(community_wiki.get('relationship'))}

## Based on

- Base image and digest: `{scalar(base.get('reference'))}`
- Base credit: {scalar(base.get('credit'))}

## Build recipe

- Public recipe: {scalar(build.get('recipe_url'))}
- Recipe commit: `{scalar(build.get('recipe_commit'))}`
- Complete build command:
```bash
{scalar(build.get('build_command'))}
```

## Source commits, PRs, patches, and overlays

{render_components(build.get('components'))}

### Package changes
{bullet_list(build.get('package_changes'))}

### Build arguments
{bullet_list(build.get('build_arguments'))}

### Environment defaults
{bullet_list(build.get('environment_defaults'))}

### Entrypoint changes
{bullet_list(build.get('entrypoint_changes'))}

- Result tree: `{scalar(build.get('result_tree'))}`
- Integration patch: `{scalar(build.get('integration_patch_sha256'))}`

## Changes from the base image

### Inherited
{bullet_list(changes.get('inherited'))}

### Introduced
{bullet_list(changes.get('introduced'))}

### Compatibility impact
{bullet_list(changes.get('compatibility_impact'))}

## Tested configuration

{render_configurations(record.get('tested_configurations'))}

## Validation results

{render_validation(validation)}

## Performance claims

{render_performance(validation.get('performance_claims'))}

## Known limitations

{bullet_list(limitations.get('known'))}

## Untested configurations

{bullet_list(limitations.get('untested'))}

## Unsupported configurations

{bullet_list(limitations.get('unsupported'))}

## Support and issue routing

- Support owner: {scalar(support.get('owner'))}
- Contact: {scalar(support.get('contact'))}
- Support commitment: `{scalar(support.get('support_commitment'))}`
- Support thread: {scalar(support.get('support_thread'))}
- Thread status: `{scalar(support.get('thread_status'))}`
- Issue tracker: {scalar(support.get('issue_tracker'))}
- Upstream escalation: {scalar(support.get('triage_policy'))}
- Superseded by: `{scalar(support.get('superseded_by'))}`

## Publication record

- Machine-readable record: {scalar(publication.get('record_url'))}
- Main-channel link: {scalar(publication.get('main_channel_link'))}
- Automated listing: `{scalar(publication.get('bot_listing'))}`
- Maintainer approval: {scalar(publication.get('maintainer_approval_url'))}
"""


def render_main_link(record: dict[str, Any]) -> str:
    if record.get("distribution_role") != "custom":
        raise ValueError("render-main-link is only for custom images")
    image = record.get("image", {})
    base = record.get("base_image", {})
    support = record.get("support", {})
    community_wiki = record.get("community_wiki", {})
    validation = record.get("validation", {})
    result = next((r for r in validation.get("results", []) if isinstance(r, dict) and r.get("status") == "passed"), None)
    tested = scalar(result.get("result")) if result else NOT_TESTED
    return f"""[Custom image] {scalar(record.get('title'))}

Image: `{scalar(image.get('reference'))}`
Based on: `{scalar(base.get('reference'))}`
Status: `{scalar(record.get('qualification_status'))}`; `{scalar(record.get('maintenance_status'))}`
Tested: {tested}
Current community runbook: {scalar(community_wiki.get('runbook_url'))}
Support and complete provenance: {scalar(support.get('support_thread'))}
"""


def render_recommended(record: dict[str, Any]) -> str:
    if record.get("distribution_role") != "recommended":
        raise ValueError("render-recommended is only for recommended images")
    image = record.get("image", {})
    base = record.get("base_image", {})
    build = record.get("build", {})
    support = record.get("support", {})
    publication = record.get("publication", {})
    community_wiki = record.get("community_wiki", {})
    validation = record.get("validation", {})
    result_summary = "; ".join(
        f"{r.get('category')}: {r.get('result')}"
        for r in validation.get("results", [])
        if isinstance(r, dict) and r.get("status") == "passed"
    ) or NOT_TESTED
    return f"""# Recommended Community Image: {scalar(record.get('model_family'))}

**Image and digest:** `{scalar(image.get('reference'))}`
**Replaces/base:** `{scalar(base.get('reference'))}`
**Build recipe:** {scalar(build.get('recipe_url'))} @ `{scalar(build.get('recipe_commit'))}`
**Source identities:** See {scalar(publication.get('record_url'))}
**Community runbook:** {scalar(community_wiki.get('runbook_url'))} @ `{scalar(community_wiki.get('commit'))}`
**Qualified configuration:** {scalar(record.get('tested_configurations', [{}])[0].get('name'))}
**Correctness/stability/performance gates:** {result_summary}
**Known limitations:** {', '.join(map(str, record.get('limitations', {}).get('known', [NA])))}
**Support route:** {scalar(support.get('support_thread'))}
**Machine-readable record:** {scalar(publication.get('record_url'))}
"""


def write_output(text: str, output: str | None) -> None:
    if output:
        Path(output).write_text(text.rstrip() + "\n", encoding="utf-8")
    else:
        print(text.rstrip())


def command_validate(args: argparse.Namespace) -> int:
    record = load_record(Path(args.record))
    report = validate_record(record, strict=args.strict)
    for issue in report.issues:
        print(issue.render())
    if report.errors:
        print(f"FAIL: {len(report.errors)} error(s)")
        return 1
    print(f"PASS: {len(report.warnings)} warning(s)")
    return 0


def command_render(args: argparse.Namespace, renderer: Any) -> int:
    record = load_record(Path(args.record))
    report = validate_record(record, strict=args.strict)
    for issue in report.issues:
        print(issue.render(), file=sys.stderr)
    if report.errors:
        print(f"FAIL: {len(report.errors)} error(s)", file=sys.stderr)
        return 1
    try:
        text = renderer(record)
    except ValueError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    write_output(text, args.output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="validate an image publication record")
    validate.add_argument("record")
    validate.add_argument("--strict", action="store_true")
    validate.set_defaults(func=command_validate)

    for name, renderer, help_text in (
        ("render-thread", render_thread, "render the complete required support thread"),
        ("render-main-link", render_main_link, "render the one allowed custom-image model-channel link"),
        ("render-recommended", render_recommended, "render the recommended-image release announcement"),
    ):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("record")
        command.add_argument("--strict", action="store_true")
        command.add_argument("--output")
        command.set_defaults(func=lambda args, r=renderer: command_render(args, r))

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except ValueError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
