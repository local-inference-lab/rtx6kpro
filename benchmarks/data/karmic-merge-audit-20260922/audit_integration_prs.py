"""Simulate PR merges in Git objects without changing a branch or worktree.

Conflict results are recorded and skipped so the report can expose independent
gaps too. A simulation with any conflict is not a reconstructed integration.
"""

import argparse
import json
import subprocess


def run(command, *, cwd=None, input=None, check=True):
    return subprocess.run(
        command, cwd=cwd, input=input, text=True, capture_output=True, check=check
    )


parser = argparse.ArgumentParser()
parser.add_argument("--repo", required=True)
parser.add_argument("--github-repo", required=True)
parser.add_argument("--canonical", required=True)
parser.add_argument("--integration", required=True)
parser.add_argument("--prs", type=int, nargs="+", required=True)
args = parser.parse_args()


def git(*arguments, **kwargs):
    return run(["git", *arguments], cwd=args.repo, **kwargs)


canonical = git("rev-parse", args.canonical).stdout.strip()
integration = git("rev-parse", args.integration).stdout.strip()
state = canonical
records = []
for number in args.prs:
    pr = json.loads(run([
        "gh", "api", f"repos/{args.github_repo}/pulls/{number}"
    ]).stdout)
    ref = f"refs/audit/20260922/pr-{number}"
    git("fetch", "--no-tags", f"https://github.com/{args.github_repo}.git",
        f"+refs/pull/{number}/head:{ref}")
    head = git("rev-parse", ref).stdout.strip()
    if head != pr["head"]["sha"]:
        raise RuntimeError(f"PR #{number} changed during the audit; rerun")
    merged = git("merge-tree", "--write-tree", state, head, check=False)
    lines = merged.stdout.splitlines()
    record = {
        "number": number, "url": pr["html_url"], "title": pr["title"],
        "head": head, "base_branch": pr["base"]["ref"],
        "head_repository": pr["head"]["repo"]["full_name"],
        "head_branch": pr["head"]["ref"], "author": pr["user"]["login"],
        "merged": pr["merged"], "state": pr["state"],
        "clean": merged.returncode == 0,
        "merge_tree_output": merged.stdout,
        "simulation_parent": state,
    }
    if merged.returncode == 0:
        tree = lines[0]
        state = git(
            "commit-tree", tree, "-p", state, "-p", head,
            input=f"Audit-only merge of {args.github_repo}#{number}\n"
        ).stdout.strip()
        record.update(tree=tree, simulation_commit=state)
    records.append(record)

print(json.dumps({
    "github_repository": args.github_repo,
    "canonical_commit": canonical, "integration_commit": integration,
    "pull_requests": records, "simulation_commit": state,
    "simulation_tree": git("rev-parse", f"{state}^{{tree}}").stdout.strip(),
    "integration_tree": git("rev-parse", f"{integration}^{{tree}}").stdout.strip(),
    "unresolved_prs": [r["number"] for r in records if not r["clean"]],
    "residual_names": git("diff", "--name-status", state, integration).stdout,
    "residual_stat": git("diff", "--stat", state, integration).stdout,
}, indent=2))
