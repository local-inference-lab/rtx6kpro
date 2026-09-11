#!/usr/bin/env python3
"""Unit tests for daily summary validation and rendering."""

from __future__ import annotations

import importlib.util
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile
import unittest


MODULE_PATH = Path(__file__).with_name("daily_summary.py")
SPEC = importlib.util.spec_from_file_location("daily_summary", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
daily_summary = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = daily_summary
SPEC.loader.exec_module(daily_summary)


def record(
    channel_id: str = "2",
    message_id: str = "3",
    content: str = "Measured 123 tok/s.",
):
    return daily_summary.MessageRecord(
        channel="#testing",
        channel_id=channel_id,
        message_id=message_id,
        timestamp="2026-09-10T12:00:00+00:00",
        author_id="4",
        author="tester",
        content=content,
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
        self.assertIsNone(
            daily_summary.render_summary(raw, [source], "2026-09-10", 1900)
        )

    def test_accepts_no_signal_result(self) -> None:
        self.assertIsNone(
            daily_summary.render_summary(
                {"highlights": [], "channels": []}, [], "2026-09-10", 1900
            )
        )

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

    def test_prefers_a_complete_clause(self) -> None:
        text = daily_summary.clean_summary_text(
            "A complete clause; trailing words that exceed the configured limit", 30
        )
        self.assertEqual(text, "A complete clause;")


class ExtractionChunkTest(unittest.TestCase):
    def test_every_record_is_primary_exactly_once(self) -> None:
        records = [record("2", str(index), "x" * 220) for index in range(100, 112)] + [
            record("5", "200", "second source")
        ]
        chunks = daily_summary.build_extraction_chunks(records, 1800, 2)
        primary_urls = [
            item.url for chunk in chunks for item, primary in chunk.records if primary
        ]
        self.assertCountEqual(primary_urls, [item.url for item in records])
        self.assertEqual(len(primary_urls), len(set(primary_urls)))
        self.assertGreater(len(chunks), 1)
        self.assertTrue(
            any(not primary for chunk in chunks for _item, primary in chunk.records)
        )
        self.assertTrue(
            all(
                daily_summary.extraction_payload_size(list(chunk.records)) <= 1800
                for chunk in chunks
            )
        )

    def test_rejects_a_single_oversized_record(self) -> None:
        source = record(content="x" * 2_000)
        with self.assertRaisesRegex(ValueError, "exceeds the extraction chunk limit"):
            daily_summary.build_extraction_chunks([source], 1_000, 0)


class SelectionBatchTest(unittest.TestCase):
    def test_every_event_is_present_in_exactly_one_batch(self) -> None:
        events = [
            {
                "text": f"event-{index}-" + "x" * 300,
                "source_urls": [record(message_id=str(index)).url],
            }
            for index in range(12)
        ]
        batches = daily_summary.build_selection_batches(events, 1_500)
        flattened = [event for batch in batches for event in batch]
        self.assertEqual(flattened, events)
        self.assertGreater(len(batches), 1)
        self.assertTrue(
            all(
                daily_summary.selection_payload_size(batch) <= 1_500
                for batch in batches
            )
        )

    def test_deduplicates_shortlists_by_source_set(self) -> None:
        source = record()
        events = [
            {"text": "first", "source_urls": [source.url]},
            {"text": "FIRST", "source_urls": [source.url]},
        ]
        self.assertEqual(
            daily_summary.deduplicate_selection_events(events), [events[0]]
        )


class SummaryPolicyTest(unittest.TestCase):
    def test_scoped_rule_and_expiry(self) -> None:
        policy_data = {
            "schema_version": 1,
            "rules": [
                {
                    "id": "scope-example",
                    "action": "downrank",
                    "reason": "Operator-reviewed low-signal source",
                    "channel_ids": ["2"],
                    "author_ids": ["4"],
                    "expires_at": "2026-10-01T00:00:00+00:00",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            path.write_text(json.dumps(policy_data), encoding="utf-8")
            policy = daily_summary.SummaryPolicy.load(path)
        actions = policy.actions_for(
            record(), datetime(2026, 9, 10, tzinfo=timezone.utc)
        )
        self.assertEqual(actions, ("downrank",))
        expired = policy.actions_for(
            record(), datetime(2026, 10, 2, tzinfo=timezone.utc)
        )
        self.assertEqual(expired, ())

    def test_rejects_unscoped_rule(self) -> None:
        policy_data = {
            "schema_version": 1,
            "rules": [{"id": "bad", "action": "suppress", "reason": "No scope"}],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            path.write_text(json.dumps(policy_data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "requires a channel or author"):
                daily_summary.SummaryPolicy.load(path)


class PerformanceClaimTest(unittest.TestCase):
    def test_rejects_cross_author_performance_claim(self) -> None:
        first = record(message_id="10")
        second = daily_summary.MessageRecord(
            **{
                **record(message_id="11").__dict__,
                "author_id": "different-author",
            }
        )
        records = {first.url: first, second.url: second}
        self.assertTrue(
            daily_summary.combines_independent_performance_claims(
                "Measured 100 tok/s and 50 tok/s", list(records), records
            )
        )

    def test_keeps_single_author_comparison(self) -> None:
        first = record(message_id="10")
        second = record(message_id="11")
        records = {first.url: first, second.url: second}
        self.assertFalse(
            daily_summary.combines_independent_performance_claims(
                "Measured 100 tok/s and 50 tok/s", list(records), records
            )
        )


if __name__ == "__main__":
    unittest.main()
