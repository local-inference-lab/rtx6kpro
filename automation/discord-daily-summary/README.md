# Discord Daily Summary Service

Status: **implemented**. The service reads public community messages from the
preceding 24 hours, asks the configured local vLLM model to select technical
highlights, validates every cited Discord URL, renders deterministic Markdown,
and publishes the same document to Discord and the `rtx6kpro` repository.

The model receives no tools. Discord messages are serialized as untrusted JSON
records, and the model returns a constrained JSON object rather than Markdown.
A second constrained model call rewrites or rejects every candidate using only
its cited source records. The Python renderer accepts only source URLs present
in the fetched records. Discord publication disables mention parsing.

The systemd unit runs as the dedicated `discord-summary` account. The account
cannot access `/root`; systemd exposes the Discord and GitHub tokens as
read-only runtime credentials. Persistent state, reproducibility artifacts,
the dedicated Git checkout, and publication receipts are stored under
`/var/lib/discord-summary`.

## Installation

Install `daily_summary.py` and `git-askpass.sh` in `/opt/discord-summary`, copy
`config.json` to `/etc/discord-summary/config.json`, and install the service and
timer units under `/etc/systemd/system`. Store the two credentials as root-only
files:

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

The production schedule is 08:07 UTC with a maximum randomized delay of one
minute. `systemctl start discord-daily-summary.service` is a publishing run;
use `--dry-run` for validation.
