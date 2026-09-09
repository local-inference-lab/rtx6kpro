"""Resolve the physical TP4 device set reserved for a qualification process."""

from __future__ import annotations

import os


def qualification_gpus(value: str | None = None) -> tuple[int, ...]:
    """Accept an explicitly named quartet supported by qualification_devices.sh."""
    value = os.environ.get("QUAL_GPU_IDS", "4,5,6,7") if value is None else value
    groups = {",".join(map(str, range(start, start + 4))) for start in (0, 4, 8, 12)}
    groups.update({"0,1,12,13", "3,12,13,14"})
    if value not in groups:
        raise ValueError(f"Expected an ordered physical GPU quartet; got {value!r}")
    return tuple(map(int, value.split(",")))


def require_container_devices(container: dict) -> None:
    """Reject a container outside the explicitly selected qualification quartet."""
    requests = container["HostConfig"]["DeviceRequests"]
    expected = [str(index) for index in qualification_gpus()]
    if len(requests) != 1 or requests[0]["DeviceIDs"] != expected:
        raise ValueError(f"Container device selection does not match {expected}")
