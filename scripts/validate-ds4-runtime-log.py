#!/usr/bin/env python3
"""Reject DeepSeek-V4 measurements that triggered runtime JIT compilation."""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from pathlib import Path


JIT_WARNING = "JIT compilation during inference"
LOG_TIMESTAMP = re.compile(
    r"\b(?:DEBUG|INFO|WARNING|ERROR|CRITICAL) "
    r"(?P<month>\d{2})-(?P<day>\d{2}) "
    r"(?P<hour>\d{2}):(?P<minute>\d{2}):(?P<second>\d{2})\b"
)


class ValidationError(ValueError):
    """A server log does not satisfy the measured-runtime contract."""


def parse_utc_boundary(value: str) -> dt.datetime:
    """Parse an ISO-8601 measurement boundary and require a UTC offset."""
    try:
        boundary = dt.datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as error:
        raise ValidationError(f"invalid UTC measurement boundary: {value!r}") from error
    if boundary.tzinfo is None or boundary.utcoffset() != dt.timedelta(0):
        raise ValidationError("measurement boundary must include a UTC offset")
    return boundary.astimezone(dt.UTC)


def _event_time(match: re.Match[str], boundary: dt.datetime) -> dt.datetime:
    values = {name: int(match.group(name)) for name in LOG_TIMESTAMP.groupindex}
    candidates = []
    for year in (boundary.year - 1, boundary.year, boundary.year + 1):
        try:
            candidates.append(dt.datetime(year=year, tzinfo=dt.UTC, **values))
        except ValueError as error:
            raise ValidationError(
                "runtime JIT warning has an invalid timestamp"
            ) from error
    return min(candidates, key=lambda value: abs(value - boundary))


def runtime_jit_warnings(log_text: str, boundary: dt.datetime) -> list[str]:
    """Return timestamped JIT warnings emitted during measured requests.

    vLLM's JIT monitor emits one timestamped warning for each Triton or CuTeDSL
    compile. Worker diagnostics can be appended out of chronological order, so
    file position is not a valid measurement boundary.
    """
    violations = []
    for line in log_text.splitlines():
        if JIT_WARNING not in line:
            continue
        match = LOG_TIMESTAMP.search(line)
        if match is None:
            raise ValidationError("runtime JIT warning has no parseable timestamp")
        if _event_time(match, boundary) >= boundary:
            violations.append(line)
    return violations


def validate_runtime_log(log_path: Path, boundary_path: Path) -> None:
    if not log_path.exists():
        raise ValidationError(f"missing server log: {log_path}")
    if not boundary_path.exists():
        raise ValidationError(f"missing measurement boundary: {boundary_path}")
    boundary = parse_utc_boundary(boundary_path.read_text(encoding="utf-8"))
    violations = runtime_jit_warnings(
        log_path.read_text(encoding="utf-8", errors="replace"), boundary
    )
    if violations:
        details = "\n".join(violations)
        raise ValidationError(
            f"{len(violations)} runtime JIT compilation warning(s) at or after "
            f"{boundary.isoformat()}:\n{details}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("server_log", type=Path)
    parser.add_argument("measurement_start_utc", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        validate_runtime_log(args.server_log, args.measurement_start_utc)
    except ValidationError as error:
        print(f"invalid benchmark runtime: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
