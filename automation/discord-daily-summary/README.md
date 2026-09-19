# Discord Daily Summary Service

Status: **implemented**. The service reads public community messages from the
preceding 24 hours, including active and recently archived public threads. It
extracts technical events from bounded conversation chunks, audits every
extracted event, validates every publication item against its cited original
messages, renders deterministic Markdown, and publishes the same document to
Discord and the `rtx6kpro` repository.

The model receives no tools. Discord messages are serialized as untrusted JSON
records, and every model stage returns constrained JSON rather than Markdown.
Each eligible record is primary evidence in exactly one extraction chunk;
adjacent chunks may repeat records as explicitly marked conversation context.
The extractor records an explicit disposition and reason for every primary
record and fails when any record is omitted. Input chunking bounds individual
model requests without dropping records. The editorial audit marks
each extracted event as `publish`, `duplicate`, `low_signal`, or `unsupported`
and records a reason. Ranking determines section placement rather than imposing
an item-count limit. An item that names an unknown event enters editorial
recovery; publication still requires an item for every `publish` event. A
citation-verification call accepts or rejects each publication item without
rewriting it. Rejected items enter a separate repair
pass with the verifier's reason and original evidence, and repaired text must
pass the citation verifier again. The Python renderer accepts only source URLs
present in the fetched records. Before verification, editorial citations are
constrained to the recorded sources of the extracted events assigned to each
item; unsupported model-selected URLs cannot enter the report.

Every model call uses the checkpoint's supported sampling contract
(`temperature=1`, `top_p=1`, deterministic `seed=0`) with thinking enabled and
`reasoning_effort=high`. Requests omit `max_tokens`; the model's context window
is the only generation ceiling. Independent extraction chunks run concurrently
according to `model_concurrency`, while the editorial and citation-verification
passes each receive the complete extracted event set.

The report separates key highlights, releases and fixes, regressions and user
reports, benchmarks and implementation findings, and active work. GitHub stores
the complete report. Discord receives the same report split at line boundaries
across as many messages as required by Discord's 2,000-character message
limit. No publication item is removed to satisfy that transport limit. Mention
parsing is disabled for every Discord message.

The GitHub filename and index entry use the UTC publication date. The report
heading identifies the completed calendar day covered by the scheduled run.
For example, the run published on `2026-09-13` stores
`daily-summaries/2026-09/2026-09-13.md` with the heading
`Daily Summary - 2026-09-12`.

A full summary execution stores `records.json`, `coverage.json`,
`model-output.json`, `summary.md`, `status.json`, and qualified per-chunk
extraction checkpoints under
`/var/lib/discord-summary/runs/YYYY-MM-DD`. `coverage.json` records discovered
sources, archived-thread discovery, fetch failures, message counts, truncation,
policy application, extraction chunk count, and exact primary-record coverage.
A production run fails before publication when source coverage is incomplete.
A retry with identical records and policy reuses a chunk checkpoint only when
its SHA-256 input identity and exact primary-record/event audit validate.
A malformed, empty, or transiently failed structured model response is retried
twice before the run fails; retries do not add an output-token ceiling. A
citation decision that retains an item without identifying valid source
evidence, or a citation repair that puts a URL in publication text, is
incomplete and enters the bounded candidate-audit recovery path.
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
artifacts without invoking the model or either publisher. Pass
`--replace-discord-messages` only when replacing a published report: the bot
posts all replacement parts successfully before deleting the prior message
set. `--publish-existing-run /var/lib/discord-summary/runs/YYYY-MM-DD` publishes
an already qualified `summary.md` without repeating collection or model calls;
the run's `status.json` must have status `ready` or `published`.

The production schedule is 08:07 UTC with a maximum randomized delay of one
minute. `systemctl start discord-daily-summary.service` is a publishing run;
use `--dry-run` for validation.
