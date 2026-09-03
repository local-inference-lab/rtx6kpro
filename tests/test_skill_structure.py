from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "tools" / "validate_skills.py"

spec = importlib.util.spec_from_file_location("validate_skills", VALIDATOR)
assert spec and spec.loader
validator = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = validator
spec.loader.exec_module(validator)

EXPECTED_SKILLS = {
    "local-inference-lab-sharing-changes",
    "local-inference-lab-publishing-docker",
    "local-inference-lab-running-benchmarks",
    "local-inference-lab-evaluating-prompts",
    "local-inference-lab-reconciling-changes",
    "local-inference-lab-reporting-bugs",
    "local-inference-lab-github-contributions",
}


class SkillStructureTests(unittest.TestCase):
    def test_collection_passes_local_spec_validator(self) -> None:
        findings = validator.validate_collection(ROOT)
        errors = [f for f in findings if f.severity == "error"]
        self.assertEqual([], errors, "\n".join(f.render() for f in findings))

    def test_each_installable_skill_has_its_own_directory(self) -> None:
        actual = {p.name for p in (ROOT / "skills").iterdir() if p.is_dir()}
        self.assertEqual(EXPECTED_SKILLS, actual)
        self.assertFalse((ROOT / "SKILL.md").exists())
        for name in actual:
            self.assertTrue((ROOT / "skills" / name / "SKILL.md").is_file())


    def test_codex_skill_only_plugin_manifest(self) -> None:
        manifest = json.loads((ROOT / ".codex-plugin" / "plugin.json").read_text(encoding="utf-8"))
        self.assertEqual("local-inference-lab-community-skills", manifest["name"])
        self.assertEqual("1.5.0", manifest["version"])
        self.assertEqual("./skills/", manifest["skills"])

    def test_each_skill_links_the_rtx6kpro_wiki_reference(self) -> None:
        for skill in (ROOT / "skills").iterdir():
            reference = skill / "references" / "rtx6kpro-wiki.md"
            self.assertTrue(reference.is_file(), f"missing {reference}")
            skill_text = (skill / "SKILL.md").read_text(encoding="utf-8")
            self.assertIn("references/rtx6kpro-wiki.md", skill_text)
            reference_text = reference.read_text(encoding="utf-8")
            self.assertIn("https://github.com/local-inference-lab/rtx6kpro", reference_text)
            self.assertIn("commit-pinned", reference_text)

    def test_reference_files_are_one_level_deep(self) -> None:
        for skill in (ROOT / "skills").iterdir():
            refs = skill / "references"
            if not refs.exists():
                continue
            nested = [p for p in refs.rglob("*") if p.is_dir()]
            self.assertEqual([], nested, f"nested reference directories in {skill.name}: {nested}")

    def test_skill_groupings_cover_every_skill_once(self) -> None:
        data = json.loads((ROOT / "skills.sh.json").read_text(encoding="utf-8"))
        listed = [name for group in data["groupings"] for name in group["skills"]]
        self.assertEqual(len(listed), len(set(listed)))
        self.assertEqual(EXPECTED_SKILLS, set(listed))

    def test_individual_skill_folders_contain_no_auxiliary_readmes(self) -> None:
        for skill in (ROOT / "skills").iterdir():
            for name in ("README.md", "CHANGELOG.md", "INSTALLATION_GUIDE.md"):
                self.assertFalse((skill / name).exists(), f"{skill.name}/{name}")

    def test_examples_use_fictional_public_identities(self) -> None:
        example_files = sorted((ROOT / "examples").rglob("*"))
        text = "\n".join(
            path.read_text(encoding="utf-8", errors="ignore")
            for path in example_files
            if path.is_file() and path.suffix.lower() in {".md", ".json", ".txt"}
        ).lower()
        self.assertNotIn("@everyone", text)
        self.assertNotIn("@here", text)
        self.assertIn("example contributor", text)
        for url_host in ("example.org", "registry.example.org"):
            self.assertIn(url_host, text)



if __name__ == "__main__":
    unittest.main()
