#!/usr/bin/env python3
"""Validate this Agent Skills collection against the open format conventions."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml
except ImportError as exc:  # pragma: no cover - explicit runtime message
    raise SystemExit("PyYAML is required for this repository validator: pip install pyyaml") from exc

ALLOWED_TOP_LEVEL = {
    "name",
    "description",
    "license",
    "compatibility",
    "metadata",
    "allowed-tools",
}
NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
XML_RE = re.compile(r"<[^>]+>")
MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
LOCAL_PATH_RE = re.compile(r"^(?:references|assets|scripts|agents)/")


@dataclass(frozen=True)
class Finding:
    severity: str
    path: str
    message: str

    def render(self) -> str:
        return f"{self.severity.upper():7} {self.path}: {self.message}"


def split_frontmatter(text: str, path: Path) -> tuple[dict[str, Any], str]:
    if not text.startswith("---\n"):
        raise ValueError(f"{path}: SKILL.md must start with YAML frontmatter")
    end = text.find("\n---\n", 4)
    if end < 0:
        raise ValueError(f"{path}: closing frontmatter delimiter is missing")
    raw = text[4:end]
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: frontmatter must be a mapping")
    return data, text[end + 5 :]


def iter_skill_dirs(root: Path) -> Iterable[Path]:
    skills = root / "skills"
    if not skills.is_dir():
        return []
    return sorted(
        (p for p in skills.iterdir() if p.is_dir() and not p.name.startswith((".", "_"))),
        key=lambda p: p.name,
    )


def validate_skill(skill: Path) -> list[Finding]:
    findings: list[Finding] = []
    skill_file = skill / "SKILL.md"
    rel_skill = skill.as_posix()
    if not skill_file.is_file():
        return [Finding("error", rel_skill, "SKILL.md is required")]

    try:
        text = skill_file.read_text(encoding="utf-8")
        meta, body = split_frontmatter(text, skill_file)
    except (OSError, UnicodeError, ValueError, yaml.YAMLError) as exc:
        return [Finding("error", skill_file.as_posix(), str(exc))]

    unknown_fields = set(meta) - ALLOWED_TOP_LEVEL
    if unknown_fields:
        findings.append(
            Finding(
                "error",
                skill_file.as_posix(),
                "unsupported top-level frontmatter fields: " + ", ".join(sorted(unknown_fields)),
            )
        )

    name = meta.get("name")
    if not isinstance(name, str) or not NAME_RE.fullmatch(name):
        findings.append(Finding("error", f"{skill_file}:name", "invalid skill name"))
    else:
        if len(name) > 64:
            findings.append(Finding("error", f"{skill_file}:name", "name exceeds 64 characters"))
        if name != skill.name:
            findings.append(Finding("error", f"{skill_file}:name", "name must match parent directory"))

    description = meta.get("description")
    if not isinstance(description, str) or not description.strip():
        findings.append(Finding("error", f"{skill_file}:description", "non-empty description is required"))
    else:
        if len(description) > 1024:
            findings.append(Finding("error", f"{skill_file}:description", "description exceeds 1024 characters"))
        if XML_RE.search(description):
            findings.append(Finding("error", f"{skill_file}:description", "description contains an XML tag"))
        lower = description.lower()
        if "use " not in lower and "when " not in lower:
            findings.append(
                Finding("error", f"{skill_file}:description", "description must state when the skill is used")
            )

    compatibility = meta.get("compatibility")
    if compatibility is not None and (
        not isinstance(compatibility, str) or not 1 <= len(compatibility) <= 500
    ):
        findings.append(
            Finding("error", f"{skill_file}:compatibility", "compatibility must be a 1-500 character string")
        )

    metadata = meta.get("metadata")
    if metadata is not None:
        if not isinstance(metadata, dict):
            findings.append(Finding("error", f"{skill_file}:metadata", "metadata must be a mapping"))
        else:
            for key, value in metadata.items():
                if not isinstance(key, str) or not isinstance(value, str):
                    findings.append(
                        Finding(
                            "error",
                            f"{skill_file}:metadata",
                            "metadata must map string keys to string values",
                        )
                    )
                    break

    allowed_tools = meta.get("allowed-tools")
    if allowed_tools is not None and not isinstance(allowed_tools, str):
        findings.append(
            Finding("error", f"{skill_file}:allowed-tools", "allowed-tools must be a space-separated string")
        )

    if not body.strip():
        findings.append(Finding("error", skill_file.as_posix(), "Markdown body is empty"))
    if len(body.splitlines()) > 500:
        findings.append(Finding("error", skill_file.as_posix(), "body exceeds 500 lines"))

    for forbidden in ("README.md", "CHANGELOG.md", "INSTALLATION_GUIDE.md", "QUICK_REFERENCE.md"):
        if (skill / forbidden).exists():
            findings.append(
                Finding("error", (skill / forbidden).as_posix(), "auxiliary documentation belongs at repository root")
            )

    for link in MARKDOWN_LINK_RE.findall(body):
        target = link.split("#", 1)[0]
        if not target or "://" in target or target.startswith("mailto:"):
            continue
        target_path = (skill / target).resolve()
        try:
            target_path.relative_to(skill.resolve())
        except ValueError:
            findings.append(Finding("error", skill_file.as_posix(), f"link escapes skill directory: {link}"))
            continue
        if not target_path.exists():
            findings.append(Finding("error", skill_file.as_posix(), f"broken local link: {link}"))
        if target.startswith("references/"):
            parts = Path(target).parts
            if len(parts) != 2:
                findings.append(
                    Finding("error", skill_file.as_posix(), f"reference must be one level deep: {link}")
                )

    # Reference files may link to anchors or public URLs, but not to further local reference files.
    references = skill / "references"
    if references.is_dir():
        for ref in references.glob("*.md"):
            ref_text = ref.read_text(encoding="utf-8")
            for link in MARKDOWN_LINK_RE.findall(ref_text):
                target = link.split("#", 1)[0]
                if target and "://" not in target and not target.startswith("#"):
                    findings.append(
                        Finding("error", ref.as_posix(), f"nested local reference chain is not allowed: {link}")
                    )

    agents = skill / "agents" / "openai.yaml"
    if not agents.is_file():
        findings.append(Finding("error", agents.as_posix(), "agents/openai.yaml is required by this collection"))
    else:
        try:
            agent_meta = yaml.safe_load(agents.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError) as exc:
            findings.append(Finding("error", agents.as_posix(), f"invalid YAML: {exc}"))
        else:
            interface = agent_meta.get("interface") if isinstance(agent_meta, dict) else None
            if not isinstance(interface, dict):
                findings.append(Finding("error", agents.as_posix(), "interface mapping is required"))
            else:
                for field in ("display_name", "short_description", "default_prompt"):
                    if not isinstance(interface.get(field), str) or not interface[field].strip():
                        findings.append(Finding("error", agents.as_posix(), f"interface.{field} is required"))
                short_description = interface.get("short_description")
                if isinstance(short_description, str) and not 25 <= len(short_description) <= 64:
                    findings.append(
                        Finding(
                            "error",
                            agents.as_posix(),
                            "interface.short_description must be 25-64 characters",
                        )
                    )
                default_prompt = interface.get("default_prompt")
                if (
                    isinstance(name, str)
                    and isinstance(default_prompt, str)
                    and f"${name}" not in default_prompt
                ):
                    findings.append(
                        Finding(
                            "error",
                            agents.as_posix(),
                            f"interface.default_prompt must explicitly mention ${name}",
                        )
                    )

    return findings


def validate_collection(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    skill_dirs = list(iter_skill_dirs(root))
    if not skill_dirs:
        findings.append(Finding("error", (root / "skills").as_posix(), "no skills found"))
        return findings
    if (root / "SKILL.md").exists():
        findings.append(Finding("error", (root / "SKILL.md").as_posix(), "collection root must not masquerade as a skill"))

    plugin_manifest = root / ".codex-plugin" / "plugin.json"
    if not plugin_manifest.is_file():
        findings.append(Finding("error", plugin_manifest.as_posix(), "skill-only plugin manifest is required by this collection"))
    else:
        try:
            plugin = json.loads(plugin_manifest.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            findings.append(Finding("error", plugin_manifest.as_posix(), f"invalid JSON: {exc}"))
        else:
            if not isinstance(plugin, dict):
                findings.append(Finding("error", plugin_manifest.as_posix(), "plugin manifest must be an object"))
            else:
                for field in ("name", "version", "description", "skills"):
                    if not isinstance(plugin.get(field), str) or not plugin[field].strip():
                        findings.append(Finding("error", plugin_manifest.as_posix(), f"{field} is required"))
                plugin_name = plugin.get("name")
                if isinstance(plugin_name, str) and not NAME_RE.fullmatch(plugin_name):
                    findings.append(Finding("error", plugin_manifest.as_posix(), "plugin name must be kebab-case"))
                if plugin.get("skills") != "./skills/":
                    findings.append(Finding("error", plugin_manifest.as_posix(), "skills must point to ./skills/"))

    for skill in skill_dirs:
        findings.extend(validate_skill(skill))
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=Path(__file__).resolve().parents[1], type=Path)
    args = parser.parse_args(argv)
    findings = validate_collection(args.root.resolve())
    for finding in findings:
        print(finding.render())
    errors = [f for f in findings if f.severity == "error"]
    if errors:
        print(f"FAIL: {len(errors)} error(s)")
        return 1
    print(f"PASS: {len(list(iter_skill_dirs(args.root.resolve())))} skill(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
