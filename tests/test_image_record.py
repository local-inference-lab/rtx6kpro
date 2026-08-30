from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills" / "local-inference-lab-publishing-docker"
SCRIPT = SKILL / "scripts" / "image_record.py"
TEMPLATE = SKILL / "assets" / "image-record.template.json"
CUSTOM = ROOT / "examples" / "example-custom-image-record.json"
RECOMMENDED = ROOT / "examples" / "example-recommended-image-record.json"

spec = importlib.util.spec_from_file_location("image_record", SCRIPT)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class ImageRecordTests(unittest.TestCase):
    def setUp(self) -> None:
        self.custom = json.loads(CUSTOM.read_text(encoding="utf-8"))
        self.recommended = json.loads(RECOMMENDED.read_text(encoding="utf-8"))

    def test_examples_pass_strict_validation(self) -> None:
        for record in (self.custom, self.recommended):
            report = module.validate_record(record, strict=True)
            self.assertEqual([], report.errors, "\n".join(i.render() for i in report.issues))

    def test_template_warns_non_strict_and_fails_strict(self) -> None:
        record = json.loads(TEMPLATE.read_text(encoding="utf-8"))
        non_strict = module.validate_record(record, strict=False)
        self.assertFalse(non_strict.errors, "\n".join(i.render() for i in non_strict.issues))
        self.assertTrue(non_strict.warnings)
        strict = module.validate_record(record, strict=True)
        self.assertTrue(strict.errors)

    def test_policy_covers_recommended_and_custom_distribution_lifecycle(self) -> None:
        policy = (SKILL / "references" / "image-policy.md").read_text(encoding="utf-8")
        required = (
            "Each model family has one recommended, maintainer-supported image.",
            "server's automated image listing or bot",
            "dedicated support thread",
            "ask a server maintainer before posting",
            "at most one link",
            "ephemeral",
            "superseded",
            "withheld or removed from main-channel linking",
        )
        for phrase in required:
            self.assertIn(phrase, policy)

    def test_wiki_reference_must_be_commit_pinned(self) -> None:
        record = copy.deepcopy(self.custom)
        record["community_wiki"]["runbook_url"] = (
            "https://github.com/local-inference-lab/rtx6kpro/blob/master/README.md"
        )
        report = module.validate_record(record, strict=True)
        self.assertTrue(any(i.path == "community_wiki.runbook_url" for i in report.errors))

    def test_renderers_include_the_current_community_runbook(self) -> None:
        thread = module.render_thread(self.custom)
        link = module.render_main_link(self.custom)
        release = module.render_recommended(self.recommended)
        for text in (thread, link, release):
            self.assertIn(self.custom["community_wiki"]["repository"], text) if text is thread else None
            self.assertIn("rtx6kpro/blob/", text)

    def test_generated_examples_match_renderer(self) -> None:
        cases = (
            (ROOT / "examples" / "example-custom-image-thread.md", module.render_thread(self.custom)),
            (ROOT / "examples" / "example-custom-image-link.md", module.render_main_link(self.custom)),
            (
                ROOT / "examples" / "example-recommended-image-thread.md",
                module.render_thread(self.recommended),
            ),
            (
                ROOT / "examples" / "example-recommended-image-release.md",
                module.render_recommended(self.recommended),
            ),
        )
        for path, rendered in cases:
            with self.subTest(path=path.name):
                self.assertEqual(rendered.rstrip(), path.read_text(encoding="utf-8").rstrip())

    def test_recommended_image_requires_all_regression_gates(self) -> None:
        record = copy.deepcopy(self.recommended)
        record["validation"]["results"] = [
            result for result in record["validation"]["results"] if result["category"] != "stability"
        ]
        report = module.validate_record(record, strict=True)
        self.assertTrue(any(i.path == "validation.results" for i in report.errors))

    def test_custom_image_cannot_claim_official_or_bot_listing(self) -> None:
        record = copy.deepcopy(self.custom)
        record["release_class"] = "official"
        record["publication"]["bot_listing"] = "listed"
        report = module.validate_record(record, strict=True)
        paths = {i.path for i in report.errors}
        self.assertIn("release_class", paths)
        self.assertIn("publication.bot_listing", paths)

    def test_custom_author_supported_status_requires_maintained_commitment(self) -> None:
        record = copy.deepcopy(self.custom)
        record["support"]["support_commitment"] = "ephemeral"
        report = module.validate_record(record, strict=True)
        self.assertTrue(any(i.path == "support.support_commitment" for i in report.errors))

    def test_custom_main_channel_link_is_bounded_to_one(self) -> None:
        record = copy.deepcopy(self.custom)
        record["publication"]["main_channel_link_count"] = 2
        report = module.validate_record(record, strict=True)
        self.assertTrue(any(i.path == "publication.main_channel_link_count" for i in report.errors))

    def test_superseded_record_requires_thread_and_replacing_digest(self) -> None:
        record = copy.deepcopy(self.custom)
        record["maintenance_status"] = "superseded"
        record["support"]["support_commitment"] = "superseded"
        record["support"]["thread_status"] = "active"
        record["support"]["superseded_by"] = "N/A"
        report = module.validate_record(record, strict=True)
        paths = {i.path for i in report.errors}
        self.assertIn("support.thread_status", paths)
        self.assertIn("support.superseded_by", paths)

    def test_performance_claim_requires_digest_pinned_control_and_exact_command(self) -> None:
        record = copy.deepcopy(self.custom)
        claim = record["validation"]["performance_claims"][0]
        claim["baseline_image"] = "registry.example.org/community/vllm:latest"
        claim["baseline_command"] = "UNKNOWN — needs verification"
        report = module.validate_record(record, strict=True)
        paths = {i.path for i in report.errors}
        self.assertIn("validation.performance_claims[0].baseline_image", paths)
        self.assertIn("validation.performance_claims[0].baseline_command", paths)

    def test_qualified_performance_claim_requires_controlled_evidence(self) -> None:
        fields = (
            "evidence_class",
            "experimental_unit",
            "runs",
            "run_order",
            "stopping_rule",
            "exclusions",
            "changed_variables",
            "nuisance_variables",
            "rival_explanations",
            "falsification_condition",
            "repeated_control_variation",
            "absolute_effect",
            "relative_effect",
            "uncertainty",
        )
        for field in fields:
            with self.subTest(field=field):
                record = copy.deepcopy(self.custom)
                record["validation"]["performance_claims"][0].pop(field)
                report = module.validate_record(record, strict=True)
                path = f"validation.performance_claims[0].{field}"
                self.assertTrue(
                    any(i.path == path for i in report.errors),
                    "\n".join(i.render() for i in report.issues),
                )

        record = copy.deepcopy(self.custom)
        claim = record["validation"]["performance_claims"][0]
        claim["evidence_class"] = "exploratory"
        claim["repeated_control_variation"] = "Not tested"
        report = module.validate_record(record, strict=True)
        paths = {i.path for i in report.errors}
        self.assertIn("validation.performance_claims[0].evidence_class", paths)
        self.assertIn("validation.performance_claims[0].repeated_control_variation", paths)

    def test_unqualified_claim_accepts_exploratory_and_confirmatory_evidence(self) -> None:
        for evidence_class in ("exploratory", "confirmatory"):
            with self.subTest(evidence_class=evidence_class):
                record = copy.deepcopy(self.custom)
                record["qualification_status"] = "implemented"
                claim = record["validation"]["performance_claims"][0]
                claim["evidence_class"] = evidence_class
                claim["repeated_control_variation"] = "Not tested"
                report = module.validate_record(record, strict=True)
                self.assertEqual([], report.errors, "\n".join(i.render() for i in report.issues))

    def test_rendered_thread_contains_every_required_section(self) -> None:
        text = module.render_thread(self.custom)
        headings = (
            "## Identity and status",
            "## Community runbook",
            "## Based on",
            "## Build recipe",
            "## Source commits, PRs, patches, and overlays",
            "## Changes from the base image",
            "## Tested configuration",
            "## Validation results",
            "## Performance claims",
            "## Known limitations",
            "## Untested configurations",
            "## Unsupported configurations",
            "## Support and issue routing",
            "## Publication record",
        )
        for heading in headings:
            self.assertIn(heading, text)
        self.assertIn(self.custom["image"]["reference"], text)
        self.assertIn(self.custom["base_image"]["reference"], text)
        claim = self.custom["validation"]["performance_claims"][0]
        expected = {
            "Evidence class": f"`{claim['evidence_class']}`",
            "Experimental unit": claim["experimental_unit"],
            "Independent repetitions": str(claim["runs"]),
            "Run order": claim["run_order"],
            "Stopping rule": claim["stopping_rule"],
            "Exclusions": claim["exclusions"],
            "Changed variables": ", ".join(claim["changed_variables"]),
            "Nuisance variables": ", ".join(claim["nuisance_variables"]),
            "Rival explanations": ", ".join(claim["rival_explanations"]),
            "Falsification condition": claim["falsification_condition"],
            "Repeated-control variation": claim["repeated_control_variation"],
            "Absolute effect": claim["absolute_effect"],
            "Relative effect": claim["relative_effect"],
            "Uncertainty": claim["uncertainty"],
        }
        for label, value in expected.items():
            self.assertIn(f"- {label}: {value}", text)

    def test_security_scan_rejects_secret_private_host_and_personal_path(self) -> None:
        record = copy.deepcopy(self.custom)
        record["summary"] = "api_key=sk-example-secret-123456 at http://10.0.0.1 /home/alice/log.txt"
        report = module.validate_record(record, strict=True)
        messages = "\n".join(i.render() for i in report.errors).lower()
        self.assertIn("credential", messages)
        self.assertIn("private hostname", messages)
        self.assertIn("filesystem path", messages)

    def test_cli_validates_and_renders_examples(self) -> None:
        for command, source in (
            ("render-main-link", CUSTOM),
            ("render-recommended", RECOMMENDED),
        ):
            result = subprocess.run(
                [sys.executable, str(SCRIPT), command, str(source), "--strict"],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(0, result.returncode, result.stdout + result.stderr)
            self.assertIn("sha256:", result.stdout)


if __name__ == "__main__":
    unittest.main()
