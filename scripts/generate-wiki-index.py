#!/usr/bin/env python3
"""Generate a complete Markdown index for the RTX PRO 6000 Blackwell wiki."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "INDEX.md"

SECTION_ORDER = [
    ("landing", "Landing Pages And Hubs"),
    ("docs", "Contributor And Onboarding Guides"),
    ("models", "Model Runbooks"),
    ("benchmarks", "Benchmarks And Quality"),
    ("kld", "Distribution Fidelity"),
    ("optimization", "Optimization Notes"),
    ("hardware", "Hardware And Topology"),
    ("inference-engines", "Inference Engines"),
    ("troubleshooting", "Troubleshooting"),
    ("daily-summaries", "Daily Summaries"),
    ("other", "Other Documents"),
]

IGNORE_PARTS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "images",
    "logs",
    "data",
}


def first_heading(path: Path) -> str:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or path.stem
    return path.stem


def section_for(path: Path) -> str:
    rel = path.relative_to(ROOT)
    if len(rel.parts) == 1 and rel.name in {"README.md", "INDEX.md", "GLOSSARY.md"}:
        return "landing"
    top = rel.parts[0]
    if top in {"docs", "models", "benchmarks", "kld", "optimization", "hardware",
               "inference-engines", "troubleshooting", "daily-summaries"}:
        return top
    return "other"


def collect_pages() -> dict[str, list[tuple[str, str]]]:
    sections: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for path in sorted(ROOT.rglob("*.md")):
        rel = path.relative_to(ROOT)
        if any(part in IGNORE_PARTS for part in rel.parts):
            continue
        if rel.parts[0] == ".git":
            continue
        title = first_heading(path)
        sections[section_for(path)].append((rel.as_posix(), title))
    return sections


def render() -> str:
    sections = collect_pages()
    lines = [
        "# RTX PRO 6000 Blackwell Wiki Index",
        "",
        "This file is generated from every Markdown page in the repository.",
        "Use it when a model page or debugging note is not linked from the",
        "front page yet.",
        "",
        "Regenerate it with:",
        "",
        "```bash",
        "python3 scripts/generate-wiki-index.py > INDEX.md",
        "```",
        "",
    ]

    for key, heading in SECTION_ORDER:
        items = sections.get(key, [])
        if not items:
            continue
        lines.extend([f"## {heading}", ""])
        for rel, title in items:
            lines.append(f"- [{title}]({rel}) - `{rel}`")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    print(render(), end="")


if __name__ == "__main__":
    main()
