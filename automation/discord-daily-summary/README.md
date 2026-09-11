# Discord Daily Summary Service

Status: **implemented**. The service reads public community messages from the
preceding 24 hours, including active and recently archived public threads. It
extracts technical events from bounded conversation chunks, selects a daily
digest from the complete event set, validates every cited Discord URL against
the original message, renders deterministic Markdown, and publishes the same
document to Discord and the `rtx6kpro` repository.

The model receives no tools. Discord messages are serialized as untrusted JSON
records, and every model stage returns constrained JSON rather than Markdown.
Each eligible record is primary evidence in exactly one extraction chunk;
adjacent chunks may repeat records as explicitly marked conversation context.
No record is discarded to fit a single model prompt. A citation-verification
call rewrites or rejects every selected candidate using only its cited original
records. The Python renderer accepts only source URLs present in the fetched
records. Discord publication disables mention parsing.

A full summary execution stores `records.json`, `coverage.json`,
`model-output.json`, `summary.md`, and `status.json` under
`/var/lib/discord-summary/runs/YYYY-MM-DD`. `coverage.json` records discovered
sources, archived-thread discovery, fetch failures, message counts, truncation,
policy application, extraction chunk count, and exact primary-record coverage.
A production run fails before publication when source coverage is incomplete.
A complete run with no independently meaningful event stores
`status=no_signal` and does not publish filler to Discord or GitHub.

`policy.json` is an operator-managed policy file. A rule may suppress,
downrank, or prioritize records and must identify a channel, an author, or
both. Every rule requires a reason and may have an ISO-8601 expiration time.
The model cannot create or modify policy. Empty rules are the default.

```json
{
  "schema_version": 1,
  "rules": [
    {
      "id": "example-expiring-channel-policy",
      "action": "downrank",
      "reason": "Operator-reviewed source is outside the digest scope",
      "channel_ids": ["123456789012345678"],
      "author_ids": [],
      "expires_at": "2026-10-01T00:00:00+00:00"
    }
  ]
}
```

The systemd unit runs as the dedicated `discord-summary` account. The account
cannot access `/root`; systemd exposes the Discord and GitHub tokens as
read-only runtime credentials. Persistent state, reproducibility artifacts,
the dedicated Git checkout, and publication receipts are stored under
`/var/lib/discord-summary`.

## Installation

Install `daily_summary.py` and `git-askpass.sh` in `/opt/discord-summary`, copy
`config.json` and `policy.json` to `/etc/discord-summary`, and install the
service and timer units under `/etc/systemd/system`. Store the two credentials
as root-only files:

```text
/etc/discord-summary/discord-token
/etc/discord-summary/github-token
```

Run a non-publishing qualification through a transient systemd unit before
enabling the timer:

```bash
sudo systemd-run --wait --pipe --collect \
  --property=User=discord-summary \
  --property=StateDirectory=discord-summary \
  --property=LoadCredential=discord-token:/etc/discord-summary/discord-token \
  --property=LoadCredential=github-token:/etc/discord-summary/github-token \
  /usr/bin/python3 /opt/discord-summary/daily_summary.py \
  --config /etc/discord-summary/config.json --dry-run
```

Pass `--records-file /var/lib/discord-summary/runs/YYYY-MM-DD/records.json` to
repeat model and renderer qualification without reading Discord again.
Use `--fetch-only` to verify Discord source discovery and write the coverage
artifacts without invoking the model or either publisher.

The production schedule is 08:07 UTC with a maximum randomized delay of one
minute. `systemctl start discord-daily-summary.service` is a publishing run;
use `--dry-run` for validation.
