#!/usr/bin/env python3
"""Unit tests for daily summary validation and rendering."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


MODULE_PATH = Path(__file__).with_name("daily_summary.py")
SPEC = importlib.util.spec_from_file_location("daily_summary", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
daily_summary = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = daily_summary
SPEC.loader.exec_module(daily_summary)


def record(channel_id: str = "2", message_id: str = "3"):
    return daily_summary.MessageRecord(
        channel="#testing",
        channel_id=channel_id,
        message_id=message_id,
        timestamp="2026-09-10T12:00:00+00:00",
        author="tester",
        content="Measured 123 tok/s.",
        url=f"https://discord.com/channels/1/{channel_id}/{message_id}",
    )


class RenderSummaryTest(unittest.TestCase):
    def test_renders_only_validated_fields(self) -> None:
        source = record()
        raw = {
            "highlights": [
                {"text": "Measured 123 tok/s.", "source_urls": [source.url]}
            ],
            "channels": [
                {"description": "Performance results", "source_urls": [source.url]}
            ],
        }
        rendered = daily_summary.render_summary(raw, [source], "2026-09-10", 1900)
        self.assertTrue(rendered.startswith("# Daily Summary - 2026-09-10"))
        self.assertIn("Measured 123 tok/s.", rendered)
        self.assertNotIn("One thing before", rendered)

    def test_rejects_hallucinated_source_urls(self) -> None:
        source = record()
        raw = {
            "highlights": [
                {
                    "text": "Unsupported claim",
                    "source_urls": ["https://discord.com/channels/1/2/999"],
                }
            ],
            "channels": [],
        }
        with self.assertRaisesRegex(ValueError, "no highlight"):
            daily_summary.render_summary(raw, [source], "2026-09-10", 1900)

    def test_disables_mass_mentions(self) -> None:
        source = record()
        raw = {
            "highlights": [{"text": "@everyone test", "source_urls": [source.url]}],
            "channels": [],
        }
        rendered = daily_summary.render_summary(raw, [source], "2026-09-10", 1900)
        self.assertNotIn("@everyone", rendered)

    def test_drops_duplicate_source(self) -> None:
        source = record()
        raw = {
            "highlights": [
                {"text": "First", "source_urls": [source.url]},
                {"text": "Duplicate", "source_urls": [source.url]},
            ],
            "channels": [],
        }
        rendered = daily_summary.render_summary(raw, [source], "2026-09-10", 1900)
        self.assertIn("First", rendered)
        self.assertNotIn("Duplicate", rendered)

    def test_truncates_at_a_word_boundary(self) -> None:
        text = daily_summary.clean_summary_text("alpha beta gamma", 12)
        self.assertEqual(text, "alpha...")

    def test_prefers_a_complete_sentence(self) -> None:
        text = daily_summary.clean_summary_text(
            "A complete sentence. A second sentence that exceeds the limit.", 35
        )
        self.assertEqual(text, "A complete sentence.")


if __name__ == "__main__":
    unittest.main()
