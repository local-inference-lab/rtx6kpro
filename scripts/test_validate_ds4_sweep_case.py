#!/usr/bin/env python3
"""Unit tests for the DeepSeek-V4 sweep artifact contract."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("validate-ds4-sweep-case.py")
SPEC = importlib.util.spec_from_file_location("validate_ds4_sweep_case", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def decode_data(**overrides):
    row = {
        "context_tokens": 0,
        "concurrency": 1,
        "aggregate_tps": 150.0,
        "request_count": 1,
        "num_completed": 1,
        "num_errors": 0,
    }
    row.update(overrides)
    return {
        "results": [row],
        "coding_peak": {
            "runs_requested": 5,
            "runs_ok": 5,
            "summary": {
                "median_generation_tok_s": 300.0,
                "cjk_runs": 0,
            },
        },
    }


class DecodeValidationTest(unittest.TestCase):
    def test_accepts_successful_duration_sample(self):
        MODULE.validate_decode(decode_data(completed_request_count=0), [1])

    def test_rejects_zero_throughput(self):
        with self.assertRaisesRegex(MODULE.ValidationError, "nonpositive"):
            MODULE.validate_decode(decode_data(aggregate_tps=0.0), [1])

    def test_rejects_empty_sample_set(self):
        with self.assertRaisesRegex(MODULE.ValidationError, "no request samples"):
            MODULE.validate_decode(decode_data(request_count=0), [1])

    def test_rejects_failed_stream(self):
        with self.assertRaisesRegex(MODULE.ValidationError, "1 error"):
            MODULE.validate_decode(decode_data(num_errors=1), [1])

    def test_accepts_sparse_warmup_at_all_measured_loads(self):
        data = decode_data(context_tokens=4094)
        data["results"] = [
            {**data["results"][0], "concurrency": concurrency}
            for concurrency in (1, 16, 32, 64)
        ]
        MODULE.validate_sparse_decode_warmup(data, [1, 16, 32, 64], 4096)

    def test_rejects_empty_sparse_warmup(self):
        with self.assertRaisesRegex(MODULE.ValidationError, "near context 4096"):
            MODULE.validate_sparse_decode_warmup({"results": []}, [1], 4096)


class PrefillValidationTest(unittest.TestCase):
    def test_accepts_tokenizer_length_tolerance(self):
        MODULE.validate_prefill(
            {
                "prefill": {
                    "65472": {
                        "prompt_tokens": 65472,
                        "tok_per_sec": 5000.0,
                        "ttft_seconds": 13.0,
                        "samples": 1,
                    }
                }
            },
            [65536],
        )

    def test_rejects_implausible_failed_request_rate(self):
        with self.assertRaisesRegex(MODULE.ValidationError, "missing prefill"):
            MODULE.validate_prefill(
                {
                    "prefill": {
                        "65536": {
                            "tok_per_sec": 2_000_000.0,
                            "ttft_seconds": 0.001,
                            "samples": 1,
                        }
                    }
                },
                [65536],
            )


if __name__ == "__main__":
    unittest.main()
