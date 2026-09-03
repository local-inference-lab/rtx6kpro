from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills" / "local-inference-lab-sharing-changes"
SCRIPT = SKILL / "scripts" / "change_package.py"
TEMPLATE = SKILL / "assets" / "package"
EXAMPLE = ROOT / "examples" / "example-change-package"

spec = importlib.util.spec_from_file_location("change_package", SCRIPT)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class ChangePackageTests(unittest.TestCase):
    def test_example_passes_strict_validation(self) -> None:
        report = module.validate_package(EXAMPLE, strict=True)
        self.assertEqual([], report.errors, "\n".join(i.render() for i in report.issues))

    def test_template_uses_exact_missing_information_vocabulary(self) -> None:
        serialized = "\n".join(
            p.read_text(encoding="utf-8")
            for p in TEMPLATE.glob("*.template.*")
        )
        self.assertIn("UNKNOWN — needs verification", serialized)
        self.assertIn("Not tested", serialized)
        self.assertIn("N/A", serialized)
        self.assertNotIn("REQUIRED:", serialized)

    def test_init_creates_expected_layout_without_github(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "shared-change"
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "init", str(destination), "--package-id", "shared-change-r1"],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(0, result.returncode, result.stdout + result.stderr)
            for name in ("README.md", "MANIFEST.json", "TESTING.md", "SHA256SUMS"):
                self.assertTrue((destination / name).is_file())
            for name in ("patches", "files", "evidence"):
                self.assertTrue((destination / name).is_dir())
            manifest = json.loads((destination / "MANIFEST.json").read_text(encoding="utf-8"))
            self.assertEqual("shared-change-r1", manifest["package_id"])
            self.assertEqual("shareable-package", manifest["delivery_route"])
            self.assertNotIn("pull_request", json.dumps(manifest).lower())
            self.assertNotIn("issue_url", json.dumps(manifest).lower())
            self.assertEqual(
                "https://github.com/local-inference-lab/rtx6kpro",
                manifest["community_wiki"]["repository"],
            )

    def test_wiki_reference_must_be_commit_pinned(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "pkg"
            shutil.copytree(EXAMPLE, package)
            manifest_path = package / "MANIFEST.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["community_wiki"]["runbook_url"] = (
                "https://github.com/local-inference-lab/rtx6kpro/blob/master/README.md"
            )
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            module.finalize_package(package)
            report = module.validate_package(package, strict=True)
            self.assertTrue(
                any(i.path == "MANIFEST.json.community_wiki.runbook_url" for i in report.errors)
            )

    def test_unknown_marker_fails_strict_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "pkg"
            shutil.copytree(EXAMPLE, package)
            manifest_path = package / "MANIFEST.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["title"] = "UNKNOWN — needs verification"
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            module.finalize_package(package)
            report = module.validate_package(package, strict=True)
            self.assertTrue(any(i.path == "MANIFEST.json.title" for i in report.errors))

    def test_changed_artifact_hash_is_repaired_by_finalize(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "pkg"
            shutil.copytree(EXAMPLE, package)
            patch = package / "patches" / "0001-bound-scheduler-queue.patch"
            patch.write_text(patch.read_text(encoding="utf-8") + "\n# test mutation\n", encoding="utf-8")
            self.assertTrue(module.validate_package(package, strict=True).errors)
            module.finalize_package(package)
            report = module.validate_package(package, strict=True)
            self.assertEqual([], report.errors, "\n".join(i.render() for i in report.issues))

    def test_strict_validation_rejects_undeclared_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "pkg"
            shutil.copytree(EXAMPLE, package)
            (package / "evidence" / "undeclared.txt").write_text("extra\n", encoding="utf-8")
            module.finalize_package(package)
            report = module.validate_package(package, strict=True)
            self.assertTrue(any(i.path == "evidence/undeclared.txt" for i in report.errors))

    def test_security_scan_rejects_secrets_and_personal_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "pkg"
            shutil.copytree(EXAMPLE, package)
            readme = package / "README.md"
            readme.write_text(readme.read_text(encoding="utf-8") + "\napi_key=sk-example-secret-123456\n/home/alice/output\n", encoding="utf-8")
            module.finalize_package(package)
            report = module.validate_package(package, strict=True)
            messages = "\n".join(i.render() for i in report.errors)
            self.assertIn("secret", messages.lower())
            self.assertIn("absolute local path", messages.lower())

    def test_archive_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for parent in ("one", "two"):
                package = Path(tmp) / parent / "example-change-package"
                package.parent.mkdir()
                shutil.copytree(EXAMPLE, package)
                output = Path(tmp) / f"{parent}.zip"
                result = subprocess.run(
                    [sys.executable, str(SCRIPT), "archive", str(package), "--output", str(output)],
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(0, result.returncode, result.stdout + result.stderr)
                with zipfile.ZipFile(output) as archive:
                    self.assertIn("example-change-package/MANIFEST.json", archive.namelist())
                paths.append(output)
            self.assertEqual(paths[0].read_bytes(), paths[1].read_bytes())


if __name__ == "__main__":
    unittest.main()
