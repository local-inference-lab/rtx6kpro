#!/usr/bin/env python3
"""Create, finalize, validate, and archive portable contribution packages.

The format is intentionally small and uses only the Python standard library.
A package is suitable for Discord/file sharing and does not require GitHub.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import stat
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse

SCHEMA_VERSION = 2
WIKI_REPOSITORY = "https://github.com/local-inference-lab/rtx6kpro"
REQUIRED_ROOT_FILES = {"README.md", "MANIFEST.json", "TESTING.md", "SHA256SUMS"}
CONTENT_DIRS = {"patches", "files", "evidence"}
ALLOWED_KINDS = {
    "patch-set",
    "source-change",
    "file-set",
    "experiment",
    "bug-reproducer",
    "benchmark-addon",
    "container-overlay",
}
ALLOWED_STATUS = {"implemented", "qualified", "research-only", "unsupported"}
ALLOWED_BASE_TYPES = {"git", "image", "archive", "files"}
ALLOWED_ARTIFACT_KINDS = {
    "git-format-patch",
    "git-diff",
    "git-bundle",
    "replacement-file",
    "additional-file",
    "script",
    "benchmark-result",
    "test-log",
    "reproducer",
    "report",
    "screenshot",
    "video",
    "evidence",
}
CHANGE_ARTIFACT_KINDS = {
    "git-format-patch",
    "git-diff",
    "git-bundle",
    "replacement-file",
    "additional-file",
    "script",
}
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
GIT_RE = re.compile(r"^[0-9a-f]{40}$")
PACKAGE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
PLACEHOLDER_RE = re.compile(
    r"(?i)(?:UNKNOWN\s+[—-]\s+needs verification|\bREQUIRED\b|\bOPTIONAL:\b|"
    r"\bTODO\b|\bTBD\b|CHANGE[-_ ]?ME|REPLACE[-_ ]?ME)"
)
SECRET_RE = re.compile(
    r"(?i)(?:bearer\s+[A-Za-z0-9._~+/-]{12,}|"
    r"(?:api[_-]?key|access[_-]?token|secret|password)\s*[:=]\s*[^\s]+|"
    r"(?:ghp_|github_pat_|sk-)[A-Za-z0-9_-]{12,})"
)
ABSOLUTE_PATH_RE = re.compile(
    r"(?i)(?:^|[\s`'\"])(?:/home/|/root/|/mnt/|file://|[A-Z]:\\)"
)
TEXT_SUFFIXES = {
    ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".py", ".sh", ".bash", ".zsh", ".ps1", ".patch", ".diff", ".csv",
    ".log", ".html", ".js", ".ts", ".tsx", ".jsx", ".css",
}


@dataclass(frozen=True)
class Issue:
    severity: str
    path: str
    message: str

    def render(self) -> str:
        return f"{self.severity.upper():7} {self.path}: {self.message}"


class Reporter:
    def __init__(self, strict: bool = False) -> None:
        self.strict = strict
        self.issues: list[Issue] = []

    def error(self, path: str, message: str) -> None:
        self.issues.append(Issue("error", path, message))

    def warning(self, path: str, message: str) -> None:
        self.issues.append(Issue("error" if self.strict else "warning", path, message))

    @property
    def errors(self) -> list[Issue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [issue for issue in self.issues if issue.severity == "warning"]


class PackageError(Exception):
    """Raised when a package cannot be read or safely processed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def load_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PackageError(f"cannot read {path}: {exc}") from exc
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PackageError(
            f"invalid JSON in {path} at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(value, dict):
        raise PackageError(f"{path} must contain a JSON object")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip() or bool(PLACEHOLDER_RE.search(value))
    if isinstance(value, (list, tuple, dict)):
        return len(value) == 0
    return False


def iter_strings(value: Any, path: str = "$") -> Iterable[tuple[str, str]]:
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, Mapping):
        for key, child in value.items():
            yield from iter_strings(child, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            yield from iter_strings(child, f"{path}[{index}]")


def safe_relative_path(value: Any, reporter: Reporter, path: str) -> str | None:
    if not isinstance(value, str) or not value.strip():
        reporter.error(path, "non-empty package-relative path is required")
        return None
    if "\\" in value:
        reporter.error(path, "use forward slashes in package paths")
        return None
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        reporter.error(path, "path must be relative and must not contain '.' or '..'")
        return None
    return pure.as_posix()


def package_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            continue
        if path.is_file() and path.name != "SHA256SUMS":
            files.append(path)
    return sorted(files, key=lambda p: p.relative_to(root).as_posix())


def parse_checksums(path: Path, reporter: Reporter) -> dict[str, str]:
    checksums: dict[str, str] = {}
    if not path.exists():
        reporter.error("SHA256SUMS", "file is missing")
        return checksums
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        reporter.error("SHA256SUMS", f"cannot read: {exc}")
        return checksums
    for lineno, line in enumerate(lines, 1):
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match:
            reporter.error(f"SHA256SUMS:{lineno}", "expected '<64 hex><two spaces><relative path>'")
            continue
        rel = match.group(2)
        if rel in checksums:
            reporter.error(f"SHA256SUMS:{lineno}", f"duplicate path {rel!r}")
            continue
        checksums[rel] = f"sha256:{match.group(1)}"
    return checksums


def validate_url_or_identity(reporter: Reporter, value: Any, path: str) -> None:
    if missing(value):
        reporter.error(path, "repository URL or durable repository identity is required")
        return
    if not isinstance(value, str):
        reporter.error(path, "must be a string")
        return
    if value.startswith(("http://", "https://")):
        parsed = urlparse(value)
        if not parsed.netloc:
            reporter.error(path, "invalid repository URL")


def require_list(reporter: Reporter, value: Any, path: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        reporter.error(path, "must be a non-empty list")
        return []
    return value


def require_text(reporter: Reporter, mapping: Mapping[str, Any], field: str, prefix: str = "") -> None:
    path = f"{prefix}.{field}" if prefix else field
    if missing(mapping.get(field)):
        reporter.error(path, "required text is missing or still contains a placeholder")


def validate_manifest(root: Path, manifest: Mapping[str, Any], reporter: Reporter) -> set[str]:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        reporter.error("MANIFEST.json.schema_version", f"must equal {SCHEMA_VERSION}")

    package_id = manifest.get("package_id")
    if missing(package_id):
        reporter.error("MANIFEST.json.package_id", "durable package ID is required")
    elif not isinstance(package_id, str) or not PACKAGE_ID_RE.fullmatch(package_id):
        reporter.error("MANIFEST.json.package_id", "use letters, numbers, dots, underscores, and hyphens")

    for field in ("title", "summary"):
        require_text(reporter, manifest, field, "MANIFEST.json")

    kind = manifest.get("kind")
    if kind not in ALLOWED_KINDS:
        reporter.error("MANIFEST.json.kind", "must be one of " + ", ".join(sorted(ALLOWED_KINDS)))

    status_value = manifest.get("qualification_status")
    if status_value not in ALLOWED_STATUS:
        reporter.error(
            "MANIFEST.json.qualification_status",
            "must be one of " + ", ".join(sorted(ALLOWED_STATUS)),
        )

    if manifest.get("delivery_route") != "shareable-package":
        reporter.error("MANIFEST.json.delivery_route", "must equal 'shareable-package'")

    community_wiki = manifest.get("community_wiki")
    if not isinstance(community_wiki, Mapping):
        reporter.error("MANIFEST.json.community_wiki", "must be an object")
        community_wiki = {}
    for field in ("repository", "commit", "runbook_path", "runbook_url", "relationship"):
        require_text(reporter, community_wiki, field, "MANIFEST.json.community_wiki")
    wiki_repository = community_wiki.get("repository")
    wiki_commit = community_wiki.get("commit")
    wiki_path = community_wiki.get("runbook_path")
    wiki_url = community_wiki.get("runbook_url")
    if isinstance(wiki_repository, str) and not missing(wiki_repository):
        if wiki_repository.rstrip("/") != WIKI_REPOSITORY:
            reporter.error(
                "MANIFEST.json.community_wiki.repository",
                f"must identify {WIKI_REPOSITORY}",
            )
    if isinstance(wiki_commit, str) and not missing(wiki_commit) and not GIT_RE.fullmatch(wiki_commit):
        reporter.error(
            "MANIFEST.json.community_wiki.commit",
            "full 40-character lowercase wiki commit is required",
        )
    if isinstance(wiki_path, str) and not missing(wiki_path):
        pure_wiki_path = PurePosixPath(wiki_path)
        if (
            pure_wiki_path.is_absolute()
            or any(part in {"", ".", ".."} for part in pure_wiki_path.parts)
            or not wiki_path.endswith(".md")
        ):
            reporter.error(
                "MANIFEST.json.community_wiki.runbook_path",
                "use a repository-relative Markdown path without parent traversal",
            )
    if all(isinstance(value, str) and not missing(value) for value in (wiki_commit, wiki_path, wiki_url)):
        expected_wiki_url = f"{WIKI_REPOSITORY}/blob/{wiki_commit}/{wiki_path}"
        if not (wiki_url == expected_wiki_url or wiki_url.startswith(expected_wiki_url + "#")):
            reporter.error(
                "MANIFEST.json.community_wiki.runbook_url",
                f"must be a commit-pinned URL beginning {expected_wiki_url!r}",
            )

    authors = require_list(reporter, manifest.get("authors"), "MANIFEST.json.authors")
    for index, raw_author in enumerate(authors):
        path = f"MANIFEST.json.authors[{index}]"
        if not isinstance(raw_author, Mapping):
            reporter.error(path, "must be an object")
            continue
        require_text(reporter, raw_author, "name", path)

    base = manifest.get("base")
    if not isinstance(base, Mapping):
        reporter.error("MANIFEST.json.base", "must be an object")
        base = {}
    base_type = base.get("type")
    if base_type not in ALLOWED_BASE_TYPES:
        reporter.error("MANIFEST.json.base.type", "must be one of " + ", ".join(sorted(ALLOWED_BASE_TYPES)))
    require_text(reporter, base, "description", "MANIFEST.json.base")
    if base_type == "git":
        validate_url_or_identity(reporter, base.get("repository"), "MANIFEST.json.base.repository")
        commit = base.get("commit")
        if not isinstance(commit, str) or not GIT_RE.fullmatch(commit):
            reporter.error("MANIFEST.json.base.commit", "full 40-character lowercase Git commit is required")
    elif base_type == "image":
        require_text(reporter, base, "image", "MANIFEST.json.base")
        digest = base.get("digest")
        if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
            reporter.error("MANIFEST.json.base.digest", "sha256 image digest is required")
    elif base_type in {"archive", "files"}:
        archive_sha = base.get("archive_sha256")
        if not isinstance(archive_sha, str) or not SHA256_RE.fullmatch(archive_sha):
            reporter.error(
                "MANIFEST.json.base.archive_sha256",
                "sha256 identity for the exact base snapshot/archive is required",
            )

    changes = manifest.get("changes")
    if not isinstance(changes, Mapping):
        reporter.error("MANIFEST.json.changes", "must be an object")
        changes = {}
    for field in ("inherited", "introduced", "compatibility_impact"):
        require_list(reporter, changes.get(field), f"MANIFEST.json.changes.{field}")

    declared: set[str] = set()
    artifacts = require_list(reporter, manifest.get("artifacts"), "MANIFEST.json.artifacts")
    change_count = 0
    for index, raw_artifact in enumerate(artifacts):
        path = f"MANIFEST.json.artifacts[{index}]"
        if not isinstance(raw_artifact, Mapping):
            reporter.error(path, "must be an object")
            continue
        rel = safe_relative_path(raw_artifact.get("path"), reporter, f"{path}.path")
        kind_value = raw_artifact.get("kind")
        if kind_value not in ALLOWED_ARTIFACT_KINDS:
            reporter.error(f"{path}.kind", "must be one of " + ", ".join(sorted(ALLOWED_ARTIFACT_KINDS)))
        if kind_value in CHANGE_ARTIFACT_KINDS:
            change_count += 1
            require_text(reporter, raw_artifact, "apply_command", path)
            require_text(reporter, raw_artifact, "revert_command", path)
        require_text(reporter, raw_artifact, "purpose", path)
        sha_value = raw_artifact.get("sha256")
        if not isinstance(sha_value, str) or not SHA256_RE.fullmatch(sha_value):
            reporter.error(f"{path}.sha256", "must be sha256: followed by 64 lowercase hex characters")
        if rel is None:
            continue
        if rel in declared:
            reporter.error(f"{path}.path", "artifact path is declared more than once")
            continue
        declared.add(rel)
        artifact_file = root / PurePosixPath(rel)
        try:
            resolved = artifact_file.resolve(strict=False)
            resolved.relative_to(root.resolve())
        except (OSError, ValueError):
            reporter.error(f"{path}.path", "artifact resolves outside the package")
            continue
        if artifact_file.is_symlink():
            reporter.error(f"{path}.path", "symlink artifacts are not allowed")
        elif not artifact_file.is_file():
            reporter.error(f"{path}.path", "declared artifact file does not exist")
        elif isinstance(sha_value, str) and SHA256_RE.fullmatch(sha_value):
            actual = sha256_file(artifact_file)
            if actual != sha_value:
                reporter.error(f"{path}.sha256", f"hash mismatch; actual {actual}")
    if change_count == 0:
        reporter.error("MANIFEST.json.artifacts", "at least one change artifact is required")

    validation = manifest.get("validation")
    if not isinstance(validation, Mapping):
        reporter.error("MANIFEST.json.validation", "must be an object")
        validation = {}
    require_list(reporter, validation.get("commands"), "MANIFEST.json.validation.commands")
    results = require_list(reporter, validation.get("results"), "MANIFEST.json.validation.results")
    for index, raw_result in enumerate(results):
        path = f"MANIFEST.json.validation.results[{index}]"
        if not isinstance(raw_result, Mapping):
            reporter.error(path, "must be an object")
            continue
        for field in ("name", "conditions", "measurement", "result", "conclusion"):
            require_text(reporter, raw_result, field, path)
        evidence = require_list(reporter, raw_result.get("evidence"), f"{path}.evidence")
        for evidence_index, evidence_path in enumerate(evidence):
            rel = safe_relative_path(evidence_path, reporter, f"{path}.evidence[{evidence_index}]")
            if rel is None:
                continue
            declared.add(rel)
            file_path = root / PurePosixPath(rel)
            if file_path.is_symlink() or not file_path.is_file():
                reporter.error(f"{path}.evidence[{evidence_index}]", "evidence file does not exist")

    limitations = manifest.get("limitations")
    if not isinstance(limitations, Mapping):
        reporter.error("MANIFEST.json.limitations", "must be an object")
        limitations = {}
    for field in ("known", "untested", "unsupported"):
        require_list(reporter, limitations.get(field), f"MANIFEST.json.limitations.{field}")

    support = manifest.get("support")
    if not isinstance(support, Mapping):
        reporter.error("MANIFEST.json.support", "must be an object")
    else:
        require_text(reporter, support, "statement", "MANIFEST.json.support")

    for string_path, text in iter_strings(manifest, "MANIFEST.json"):
        if SECRET_RE.search(text):
            reporter.error(string_path, "possible credential or secret detected")
        if ABSOLUTE_PATH_RE.search(text):
            reporter.error(string_path, "absolute local path makes the package non-portable")

    return declared


def scan_text_files(root: Path, reporter: Reporter, strict: bool) -> None:
    for path in package_files(root):
        rel = path.relative_to(root).as_posix()
        if path.stat().st_size > 10 * 1024 * 1024:
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {"README.md", "TESTING.md"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if SECRET_RE.search(text):
            reporter.error(rel, "possible credential or secret detected")
        if rel in {"README.md", "TESTING.md", "MANIFEST.json"}:
            if PLACEHOLDER_RE.search(text):
                reporter.error(rel, "required template placeholders remain")
            if ABSOLUTE_PATH_RE.search(text):
                reporter.error(rel, "absolute local path makes the package non-portable")
        elif strict and rel.endswith((".md", ".txt", ".json")) and ABSOLUTE_PATH_RE.search(text):
            reporter.warning(rel, "contains an absolute local path; confirm it is intentional evidence")


def validate_package(root: Path, strict: bool = False) -> Reporter:
    reporter = Reporter(strict=strict)
    if not root.exists() or not root.is_dir():
        reporter.error("$", "package path must be an existing directory")
        return reporter

    for required in sorted(REQUIRED_ROOT_FILES):
        if not (root / required).is_file():
            reporter.error(required, "required root file is missing")

    for path in root.rglob("*"):
        if path.is_symlink():
            reporter.error(path.relative_to(root).as_posix(), "symlinks are not allowed")

    manifest: dict[str, Any] = {}
    if (root / "MANIFEST.json").is_file():
        try:
            manifest = load_json(root / "MANIFEST.json")
        except PackageError as exc:
            reporter.error("MANIFEST.json", str(exc))

    declared: set[str] = set()
    if manifest:
        declared = validate_manifest(root, manifest, reporter)

    actual_files = {path.relative_to(root).as_posix(): path for path in package_files(root)}
    checksums = parse_checksums(root / "SHA256SUMS", reporter)
    for rel, path in actual_files.items():
        expected = checksums.get(rel)
        if expected is None:
            reporter.error(f"SHA256SUMS:{rel}", "file is missing from SHA256SUMS")
            continue
        actual = sha256_file(path)
        if actual != expected:
            reporter.error(f"SHA256SUMS:{rel}", f"hash mismatch; actual {actual}")
    for rel in checksums:
        if rel not in actual_files:
            reporter.error(f"SHA256SUMS:{rel}", "checksum references a missing file")

    if strict:
        for directory in CONTENT_DIRS:
            content_root = root / directory
            if not content_root.exists():
                continue
            for path in content_root.rglob("*"):
                if not path.is_file() or path.name == "README.md":
                    continue
                rel = path.relative_to(root).as_posix()
                if rel not in declared:
                    reporter.error(rel, "content file is not declared as an artifact or validation evidence")

    scan_text_files(root, reporter, strict)
    return reporter


def finalize_package(root: Path) -> None:
    if not root.exists() or not root.is_dir():
        raise PackageError("package path must be an existing directory")
    manifest_path = root / "MANIFEST.json"
    manifest = load_json(manifest_path)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise PackageError("MANIFEST.json.artifacts must be a list")
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict):
            raise PackageError(f"MANIFEST.json.artifacts[{index}] must be an object")
        raw_path = artifact.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise PackageError(f"MANIFEST.json.artifacts[{index}].path is missing")
        pure = PurePosixPath(raw_path)
        if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
            raise PackageError(f"unsafe artifact path: {raw_path!r}")
        file_path = root / pure
        if file_path.is_symlink() or not file_path.is_file():
            raise PackageError(f"artifact file does not exist: {raw_path}")
        artifact["sha256"] = sha256_file(file_path)
    write_json(manifest_path, manifest)

    lines: list[str] = []
    for path in package_files(root):
        rel = path.relative_to(root).as_posix()
        lines.append(f"{sha256_file(path).split(':', 1)[1]}  {rel}")
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def deterministic_zip(root: Path, output: Path) -> None:
    root_name = root.name
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(root.rglob("*"), key=lambda p: p.relative_to(root).as_posix()):
            if path.is_symlink():
                raise PackageError(f"symlink not allowed: {path}")
            if not path.is_file():
                continue
            rel = path.relative_to(root).as_posix()
            info = zipfile.ZipInfo(f"{root_name}/{rel}", date_time=(1980, 1, 1, 0, 0, 0))
            mode = path.stat().st_mode
            permissions = 0o755 if mode & stat.S_IXUSR else 0o644
            info.external_attr = permissions << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())


def print_report(reporter: Reporter) -> None:
    for issue in reporter.issues:
        print(issue.render())
    print(
        f"Validation complete: {len(reporter.errors)} error(s), "
        f"{len(reporter.warnings)} warning(s)."
    )


def command_init(args: argparse.Namespace) -> int:
    destination: Path = args.destination
    if not PACKAGE_ID_RE.fullmatch(args.package_id):
        print(
            "ERROR   $: package ID must use letters, numbers, dots, underscores, and hyphens",
            file=sys.stderr,
        )
        return 1
    if destination.exists():
        if not destination.is_dir():
            print(f"ERROR   $: destination is not a directory: {destination}", file=sys.stderr)
            return 1
        if any(destination.iterdir()):
            print(f"ERROR   $: destination is not empty: {destination}", file=sys.stderr)
            return 1
    template = Path(__file__).resolve().parents[1] / "assets" / "package"
    if not template.is_dir():
        print(f"ERROR   $: template directory is missing: {template}", file=sys.stderr)
        return 1
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copytree(template, destination, dirs_exist_ok=True)
    for source_name, target_name in (
        ("README.template.md", "README.md"),
        ("TESTING.template.md", "TESTING.md"),
        ("MANIFEST.template.json", "MANIFEST.json"),
    ):
        source = destination / source_name
        target = destination / target_name
        source.replace(target)
    manifest = load_json(destination / "MANIFEST.json")
    manifest["package_id"] = args.package_id
    write_json(destination / "MANIFEST.json", manifest)
    (destination / "SHA256SUMS").write_text("", encoding="utf-8")
    print(f"Initialized package at {destination}")
    return 0


def command_finalize(args: argparse.Namespace) -> int:
    try:
        finalize_package(args.package)
    except PackageError as exc:
        print(f"ERROR   $: {exc}", file=sys.stderr)
        return 1
    print(f"Updated manifest hashes and SHA256SUMS in {args.package}")
    return 0


def command_validate(args: argparse.Namespace) -> int:
    reporter = validate_package(args.package, strict=args.strict)
    print_report(reporter)
    return 1 if reporter.errors else 0


def command_archive(args: argparse.Namespace) -> int:
    root = args.package.resolve()
    output = args.output.resolve()
    try:
        output.relative_to(root)
    except ValueError:
        pass
    else:
        print("ERROR   $: archive output must be outside the package directory", file=sys.stderr)
        return 1
    if output.exists() and not args.force:
        print(f"ERROR   $: output already exists: {output}; use --force", file=sys.stderr)
        return 1
    reporter = validate_package(root, strict=True)
    if reporter.errors:
        print_report(reporter)
        print("Archive refused because strict validation failed.", file=sys.stderr)
        return 1
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    try:
        deterministic_zip(root, output)
    except (OSError, PackageError) as exc:
        print(f"ERROR   $: cannot create archive: {exc}", file=sys.stderr)
        return 1
    print(f"Created {output}")
    print(f"Archive SHA-256: {sha256_file(output)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="copy the package template")
    init_parser.add_argument("destination", type=Path)
    init_parser.add_argument("--package-id", required=True)
    init_parser.set_defaults(func=command_init)

    finalize_parser = subparsers.add_parser("finalize", help="update hashes and SHA256SUMS")
    finalize_parser.add_argument("package", type=Path)
    finalize_parser.set_defaults(func=command_finalize)

    validate_parser = subparsers.add_parser("validate", help="validate a package")
    validate_parser.add_argument("package", type=Path)
    validate_parser.add_argument("--strict", action="store_true")
    validate_parser.set_defaults(func=command_validate)

    archive_parser = subparsers.add_parser("archive", help="create a deterministic ZIP")
    archive_parser.add_argument("package", type=Path)
    archive_parser.add_argument("--output", type=Path, required=True)
    archive_parser.add_argument("--force", action="store_true")
    archive_parser.set_defaults(func=command_archive)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
