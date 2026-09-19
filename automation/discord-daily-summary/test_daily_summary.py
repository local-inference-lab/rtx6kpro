#!/usr/bin/env python3
"""Unit tests for daily summary validation and rendering."""

from __future__ import annotations

import importlib.util
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch


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
    @staticmethod
    def item(source, text: str = "Measured 123 tok/s.", event_id: str = "e0001"):
        return {
            "section": "benchmarks_and_implementation_findings",
            "text": text,
            "event_ids": [event_id],
            "source_urls": [source.url],
        }

    def test_renders_only_validated_fields(self) -> None:
        source = record()
        raw = {"items": [self.item(source)]}
        rendered = daily_summary.render_summary(raw, [source], "2026-09-10")
        self.assertTrue(rendered.startswith("# Daily Summary - 2026-09-10"))
        self.assertIn("Measured 123 tok/s.", rendered)
        self.assertIn("## Benchmarks and implementation findings", rendered)
        self.assertNotIn("One thing before", rendered)

    def test_rejects_hallucinated_source_urls(self) -> None:
        source = record()
        raw = {
            "items": [
                {
                    **self.item(source, "Unsupported claim"),
                    "source_urls": ["https://discord.com/channels/1/2/999"],
                }
            ]
        }
        with self.assertRaisesRegex(ValueError, "outside the daily record set"):
            daily_summary.render_summary(raw, [source], "2026-09-10")

    def test_accepts_no_signal_result(self) -> None:
        self.assertIsNone(daily_summary.render_summary({"items": []}, [], "2026-09-10"))

    def test_disables_mass_mentions(self) -> None:
        source = record()
        raw = {"items": [self.item(source, "@everyone test")]}
        rendered = daily_summary.render_summary(raw, [source], "2026-09-10")
        self.assertNotIn("@everyone", rendered)

    def test_rejects_duplicate_event(self) -> None:
        source = record()
        raw = {
            "items": [
                self.item(source, "First"),
                self.item(source, "Duplicate"),
            ],
        }
        with self.assertRaisesRegex(ValueError, "repeated event identifiers"):
            daily_summary.render_summary(raw, [source], "2026-09-10")

    def test_splits_without_dropping_bullets(self) -> None:
        source = record()
        raw = {
            "items": [
                self.item(source, f"Finding {index} " + "x" * 50, f"e{index:04d}")
                for index in range(1, 8)
            ]
        }
        rendered = daily_summary.render_summary(raw, [source], "2026-09-10")
        parts = daily_summary.split_discord_messages(rendered, 180)
        self.assertGreater(len(parts), 1)
        self.assertTrue(all(len(part) <= 180 for part in parts))
        for index in range(1, 8):
            self.assertEqual(sum(f"Finding {index} " in part for part in parts), 1)

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

    def test_reuses_a_qualified_checkpoint_for_identical_input(self) -> None:
        source = record(content="B12X measured 123 tok/s on RTX hardware.")
        chunk = daily_summary.ExtractionChunk("chunk-001", ((source, True),))
        response = {
            "record_audit": [
                {
                    "url": source.url,
                    "disposition": "event",
                    "reason": "Measured inference performance.",
                }
            ],
            "events": [
                {
                    "text": "B12X measured 123 tok/s on RTX hardware.",
                    "status": "measured",
                    "kind": "performance",
                    "importance": 3,
                    "source_urls": [source.url],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_directory = Path(directory)
            first = daily_summary.LocalModelClient("http://model", "model")
            first.complete_json = Mock(return_value=(response, {"total_tokens": 20}))
            extracted = first.extract_events(
                [chunk], {}, "2026-09-13", 1, checkpoint_directory
            )
            self.assertEqual(len(extracted["events"]), 1)
            self.assertTrue((checkpoint_directory / "chunk-001.json").exists())

            second = daily_summary.LocalModelClient("http://model", "model")
            second.complete_json = Mock(side_effect=AssertionError("model was called"))
            reused = second.extract_events(
                [chunk], {}, "2026-09-13", 1, checkpoint_directory
            )

        self.assertEqual(reused, extracted)
        second.complete_json.assert_not_called()


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


class EditorialValidationTest(unittest.TestCase):
    def test_recovers_known_events_from_an_item_with_an_unknown_event(self) -> None:
        source = record(message_id="10")
        events = [
            {
                "id": "e0001",
                "text": "A release was published.",
                "source_urls": [source.url],
            },
            {
                "id": "e0002",
                "text": "A benchmark was measured.",
                "source_urls": [source.url],
            },
        ]
        raw = {
            "audit": [
                {"event_id": "e0001", "disposition": "publish", "reason": "Release"},
                {"event_id": "e0002", "disposition": "publish", "reason": "Benchmark"},
            ],
            "items": [
                {
                    "section": "releases_and_fixes",
                    "text": "A release was published.",
                    "event_ids": ["e0001", "e9999"],
                    "source_urls": [source.url],
                },
                {
                    "section": "benchmarks_and_implementation_findings",
                    "text": "A benchmark was measured.",
                    "event_ids": ["e0002"],
                    "source_urls": [source.url],
                },
            ],
        }

        validated = daily_summary.validate_editorial_output(
            raw, events, {source.url: source}, require_publication_coverage=False
        )

        self.assertEqual(validated["_missing_publish_event_ids"], ["e0001"])
        self.assertEqual(validated["items"][0]["event_ids"], ["e0002"])
        with self.assertRaisesRegex(ValueError, "publication coverage mismatch"):
            daily_summary.validate_editorial_output(raw, events, {source.url: source})

    def test_constrains_editorial_citations_to_the_items_events(self) -> None:
        source = record(message_id="10")
        unrelated = record(message_id="11")
        events = [
            {
                "id": "e0001",
                "text": "Measured 123 tok/s.",
                "source_urls": [source.url],
            }
        ]
        raw = {
            "audit": [
                {"event_id": "e0001", "disposition": "publish", "reason": "Useful"}
            ],
            "items": [
                {
                    "section": "benchmarks_and_implementation_findings",
                    "text": "Measured 123 tok/s.",
                    "event_ids": ["e0001"],
                    "source_urls": [unrelated.url],
                }
            ],
        }

        validated = daily_summary.validate_editorial_output(
            raw,
            events,
            {source.url: source, unrelated.url: unrelated},
        )

        self.assertEqual(validated["items"][0]["source_urls"], [source.url])

    def test_requires_an_audit_decision_for_every_event(self) -> None:
        source = record()
        events = [
            {
                "id": "e0001",
                "text": "Measured 123 tok/s.",
                "source_urls": [source.url],
            },
            {
                "id": "e0002",
                "text": "A second finding.",
                "source_urls": [source.url],
            },
        ]
        raw = {
            "audit": [
                {"event_id": "e0001", "disposition": "publish", "reason": "Useful"}
            ],
            "items": [
                {
                    "section": "key_highlights",
                    "text": "Measured 123 tok/s.",
                    "event_ids": ["e0001"],
                    "source_urls": [source.url],
                }
            ],
        }
        with self.assertRaisesRegex(ValueError, "omitted extracted events"):
            daily_summary.validate_editorial_output(raw, events, {source.url: source})

    def test_preserves_secondary_technical_events(self) -> None:
        source = record()
        events = [
            {
                "id": "e0001",
                "text": "A release was published.",
                "source_urls": [source.url],
            },
            {
                "id": "e0002",
                "text": "A benchmark was measured.",
                "source_urls": [source.url],
            },
        ]
        raw = {
            "audit": [
                {"event_id": "e0001", "disposition": "publish", "reason": "Release"},
                {"event_id": "e0002", "disposition": "publish", "reason": "Benchmark"},
            ],
            "items": [
                {
                    "section": "releases_and_fixes",
                    "text": "A release was published.",
                    "event_ids": ["e0001"],
                    "source_urls": [source.url],
                },
                {
                    "section": "benchmarks_and_implementation_findings",
                    "text": "A benchmark was measured.",
                    "event_ids": ["e0002"],
                    "source_urls": [source.url],
                },
            ],
        }
        validated = daily_summary.validate_editorial_output(
            raw, events, {source.url: source}
        )
        self.assertEqual(len(validated["items"]), 2)


class ModelRequestTest(unittest.TestCase):
    @patch.object(daily_summary.requests, "post")
    def test_uses_reasoning_without_an_output_token_cap(self, post: Mock) -> None:
        response = Mock()
        response.json.return_value = {
            "choices": [
                {"finish_reason": "stop", "message": {"content": '{"value":1}'}}
            ],
            "usage": {},
        }
        response.raise_for_status.return_value = None
        post.return_value = response
        client = daily_summary.LocalModelClient("http://model", "model")

        client.complete_json("system", "user", {"type": "json_object"}, timeout=10)

        payload = post.call_args.kwargs["json"]
        self.assertNotIn("max_tokens", payload)
        self.assertEqual(payload["temperature"], 1.0)
        self.assertEqual(payload["top_p"], 1.0)
        self.assertEqual(
            payload["chat_template_kwargs"],
            {"thinking": True, "reasoning_effort": "high"},
        )

    @patch.object(daily_summary.time, "sleep")
    @patch.object(daily_summary.requests, "post")
    def test_retries_a_null_structured_response(self, post: Mock, sleep: Mock) -> None:
        empty_response = Mock()
        empty_response.raise_for_status.return_value = None
        empty_response.json.return_value = {
            "choices": [{"finish_reason": "stop", "message": {"content": None}}]
        }
        valid_response = Mock()
        valid_response.raise_for_status.return_value = None
        valid_response.json.return_value = {
            "choices": [
                {"finish_reason": "stop", "message": {"content": '{"value":1}'}}
            ],
            "usage": {"prompt_tokens": 10},
        }
        post.side_effect = [empty_response, valid_response]
        client = daily_summary.LocalModelClient("http://model", "model")

        result, usage = client.complete_json(
            "system", "user", {"type": "json_object"}, timeout=10
        )

        self.assertEqual(result, {"value": 1})
        self.assertEqual(usage, {"prompt_tokens": 10})
        self.assertEqual(post.call_count, 2)
        sleep.assert_called_once_with(1)

    def test_keeps_single_author_comparison(self) -> None:
        first = record(message_id="10")
        second = record(message_id="11")
        records = {first.url: first, second.url: second}
        self.assertFalse(
            daily_summary.combines_independent_performance_claims(
                "Measured 100 tok/s and 50 tok/s", list(records), records
            )
        )


class CitationVerificationTest(unittest.TestCase):
    def test_recovers_repaired_text_that_contains_a_url(self) -> None:
        source = record(message_id="10", content="B12X added a planner.")
        candidates = {
            "audit": [{"event_id": "e0001", "disposition": "publish", "reason": "Fix"}],
            "items": [
                {
                    "section": "releases_and_fixes",
                    "text": "B12X added a planner.",
                    "event_ids": ["e0001"],
                    "source_urls": [source.url],
                }
            ],
        }
        client = daily_summary.LocalModelClient("http://model", "model")
        client.complete_json = Mock(
            side_effect=[
                (
                    {
                        "candidates": [
                            {
                                "id": "i0001",
                                "keep": False,
                                "reason": "Rewrite the statement.",
                                "used_source_numbers": [],
                            }
                        ]
                    },
                    {},
                ),
                (
                    {
                        "candidates": [
                            {
                                "id": "i0001",
                                "keep": True,
                                "reason": "Supported.",
                                "text": "B12X added a planner. https://example.com",
                                "used_source_numbers": [0],
                            }
                        ]
                    },
                    {},
                ),
                (
                    {
                        "candidates": [
                            {
                                "id": "i0001",
                                "keep": True,
                                "reason": "Supported.",
                                "text": "B12X added a planner.",
                                "used_source_numbers": [0],
                            }
                        ]
                    },
                    {},
                ),
                (
                    {
                        "candidates": [
                            {
                                "id": "i0001",
                                "keep": True,
                                "reason": "Supported.",
                                "used_source_numbers": [0],
                            }
                        ]
                    },
                    {},
                ),
            ]
        )

        result = client.verify_candidates(candidates, [source])

        self.assertEqual(len(result["items"]), 1)
        self.assertEqual(result["items"][0]["text"], "B12X added a planner.")
        self.assertEqual(client.complete_json.call_count, 4)

    def test_recovers_a_retained_candidate_without_valid_evidence(self) -> None:
        source = record(message_id="10", content="B12X added a planner.")
        candidates = {
            "audit": [{"event_id": "e0001", "disposition": "publish", "reason": "Fix"}],
            "items": [
                {
                    "section": "releases_and_fixes",
                    "text": "B12X added a planner.",
                    "event_ids": ["e0001"],
                    "source_urls": [source.url],
                }
            ],
        }
        client = daily_summary.LocalModelClient("http://model", "model")
        client.complete_json = Mock(
            side_effect=[
                (
                    {
                        "candidates": [
                            {
                                "id": "i0001",
                                "keep": True,
                                "reason": "Supported.",
                                "used_source_numbers": [],
                            }
                        ]
                    },
                    {},
                ),
                (
                    {
                        "candidates": [
                            {
                                "id": "i0001",
                                "keep": True,
                                "reason": "Supported.",
                                "used_source_numbers": [0],
                            }
                        ]
                    },
                    {},
                ),
            ]
        )

        result = client.verify_candidates(candidates, [source])

        self.assertEqual(len(result["items"]), 1)
        self.assertEqual(result["items"][0]["source_urls"], [source.url])
        self.assertEqual(client.complete_json.call_count, 2)

    def test_recovers_candidate_ids_omitted_by_a_verifier_pass(self) -> None:
        first = record(message_id="10", content="B12X added a planner.")
        second = record(message_id="11", content="vLLM fixed a loader.")
        candidates = {
            "audit": [
                {"event_id": "e0001", "disposition": "publish", "reason": "Fix"},
                {"event_id": "e0002", "disposition": "publish", "reason": "Fix"},
            ],
            "items": [
                {
                    "section": "releases_and_fixes",
                    "text": "B12X added a planner.",
                    "event_ids": ["e0001"],
                    "source_urls": [first.url],
                },
                {
                    "section": "releases_and_fixes",
                    "text": "vLLM fixed a loader.",
                    "event_ids": ["e0002"],
                    "source_urls": [second.url],
                },
            ],
        }
        client = daily_summary.LocalModelClient("http://model", "model")
        client.complete_json = Mock(
            side_effect=[
                (
                    {
                        "candidates": [
                            {
                                "id": "i0001",
                                "keep": True,
                                "reason": "Supported.",
                                "used_source_numbers": [0],
                            }
                        ]
                    },
                    {},
                ),
                (
                    {
                        "candidates": [
                            {
                                "id": "i0002",
                                "keep": True,
                                "reason": "Supported.",
                                "used_source_numbers": [0],
                            }
                        ]
                    },
                    {},
                ),
            ]
        )

        result = client.verify_candidates(candidates, [first, second])

        self.assertEqual(len(result["items"]), 2)
        self.assertEqual(client.complete_json.call_count, 2)


class DiscordPublicationTest(unittest.TestCase):
    def test_replace_posts_every_part_before_deleting_prior_messages(self) -> None:
        client = daily_summary.DiscordClient("token", "guild")
        client.publish = Mock(side_effect=["new-1", "new-2"])
        client.delete_message = Mock()

        result = client.publish_many(
            "summary-channel",
            ["first part", "second part"],
            ["prior-1", "prior-2"],
            replace=True,
        )

        self.assertEqual(result, ["new-1", "new-2"])
        self.assertEqual(client.publish.call_count, 2)
        self.assertEqual(
            client.delete_message.call_args_list,
            [
                unittest.mock.call("summary-channel", "prior-1"),
                unittest.mock.call("summary-channel", "prior-2"),
            ],
        )

    def test_replace_removes_partial_posts_and_preserves_prior_messages(self) -> None:
        client = daily_summary.DiscordClient("token", "guild")
        client.publish = Mock(side_effect=["new-1", RuntimeError("post failed")])
        client.delete_message = Mock()

        with self.assertRaisesRegex(RuntimeError, "post failed"):
            client.publish_many(
                "summary-channel",
                ["first part", "second part"],
                ["prior-1"],
                replace=True,
            )

        client.delete_message.assert_called_once_with("summary-channel", "new-1")


class PublicationRecoveryTest(unittest.TestCase):
    def initialize_repository(self, directory: str) -> Path:
        repository = Path(directory)
        subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=repository,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"], cwd=repository, check=True
        )
        index_path = repository / "daily-summaries" / "README.md"
        index_path.parent.mkdir(parents=True)
        index_path.write_text("index\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=repository, check=True)
        subprocess.run(["git", "commit", "-qm", "Initial"], cwd=repository, check=True)
        return repository

    def test_recovers_only_expected_publication_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = self.initialize_repository(directory)
            index_path = repository / "daily-summaries" / "README.md"
            summary_path = repository / "daily-summaries" / "2026-09" / "2026-09-12.md"
            index_path.write_text("changed\n", encoding="utf-8")
            summary_path.parent.mkdir(parents=True)
            summary_path.write_text("summary\n", encoding="utf-8")

            daily_summary.recover_interrupted_publication(
                repository,
                {},
                {
                    "daily-summaries/README.md",
                    "daily-summaries/2026-09/2026-09-12.md",
                },
            )

            self.assertEqual(index_path.read_text(encoding="utf-8"), "index\n")
            self.assertFalse(summary_path.exists())
            self.assertEqual(
                daily_summary.changed_repository_paths(repository, {}), set()
            )

    def test_rejects_changes_outside_publication_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = self.initialize_repository(directory)
            unexpected = repository / "unrelated.txt"
            unexpected.write_text("keep me\n", encoding="utf-8")

            with self.assertRaisesRegex(
                RuntimeError, "changes outside the publication paths"
            ):
                daily_summary.recover_interrupted_publication(
                    repository,
                    {},
                    {"daily-summaries/README.md"},
                )

            self.assertTrue(unexpected.exists())


if __name__ == "__main__":
    unittest.main()
