#!/usr/bin/env python3
"""Unit tests for DeepSeek-V4 runtime JIT log qualification."""

from __future__ import annotations

import datetime as dt
import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("validate-ds4-runtime-log.py")
SPEC = importlib.util.spec_from_file_location("validate_ds4_runtime_log", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


BOUNDARY = dt.datetime(2026, 8, 13, 7, 30, 0, tzinfo=dt.UTC)


class RuntimeLogValidationTest(unittest.TestCase):
    def test_accepts_pre_measurement_warning_appended_after_boundary_lines(self):
        log = "\n".join(
            (
                "INFO 08-13 07:30:05 measured request completed",
                "WARNING 08-13 07:29:59 [jit_monitor.py:135] "
                "CuTeDSL JIT compilation during inference: call_pertok.",
            )
        )
        self.assertEqual(MODULE.runtime_jit_warnings(log, BOUNDARY), [])

    def test_rejects_warning_during_measurement(self):
        warning = (
            "WARNING 08-13 07:30:01 [jit_monitor.py:135] "
            "Triton kernel JIT compilation during inference: kernel."
        )
        self.assertEqual(MODULE.runtime_jit_warnings(warning, BOUNDARY), [warning])

    def test_rejects_unparseable_jit_warning(self):
        with self.assertRaisesRegex(MODULE.ValidationError, "no parseable timestamp"):
            MODULE.runtime_jit_warnings(
                "JIT compilation during inference: unknown format", BOUNDARY
            )

    def test_resolves_timestamp_across_year_boundary(self):
        boundary = dt.datetime(2027, 1, 1, 0, 0, 1, tzinfo=dt.UTC)
        warning = (
            "WARNING 12-31 23:59:59 [jit_monitor.py:135] "
            "CuTeDSL JIT compilation during inference: kernel."
        )
        self.assertEqual(MODULE.runtime_jit_warnings(warning, boundary), [])


if __name__ == "__main__":
    unittest.main()
