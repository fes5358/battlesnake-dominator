"""
github_push.py — Minimal GitHub Git Data API client (stdlib only, no extra deps).

Used by auto_improve.py to commit auto-tuned constants straight to the
`battlesnake-dominator` repo without shelling out to git (git CLI commits are
not available in this environment).
"""

import json
import os
import urllib.error
import urllib.request

OWNER = "fes5358"
REPO = "battlesnake-dominator"
API = f"https://api.github.com/repos/{OWNER}/{REPO}"


def _headers() -> dict:
    token = os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_PERSONAL_ACCESS_TOKEN is not set")
    return {
        "Authorization": f"token {token}",
        "User-Agent": "battlesnake-dominator-auto-improve",
        "Accept": "application/vnd.github+json",
        "Content-Type": "application/json",
    }


def _req(method: str, url: str, data: dict = None):
    body = json.dumps(data).encode() if data is not None else None
    r = urllib.request.Request(url, data=body, headers=_headers(), method=method)
    try:
        with urllib.request.urlopen(r, timeout=20) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


def push_files(files: dict, commit_message: str, branch: str = "main") -> str:
    """
    Commit one or more files (path -> text content) directly to `branch` via
    the GitHub Git Data API. Returns the new commit SHA.
    """
    status, ref = _req("GET", f"{API}/git/refs/heads/{branch}")
    if status != 200:
        raise RuntimeError(f"Failed to get ref: {status} {ref}")
    base_sha = ref["object"]["sha"]

    status, base_commit = _req("GET", f"{API}/git/commits/{base_sha}")
    if status != 200:
        raise RuntimeError(f"Failed to get base commit: {status} {base_commit}")
    base_tree = base_commit["tree"]["sha"]

    tree_entries = []
    for path, content in files.items():
        status, blob = _req("POST", f"{API}/git/blobs", {"content": content, "encoding": "utf-8"})
        if status not in (200, 201):
            raise RuntimeError(f"Failed to create blob for {path}: {status} {blob}")
        tree_entries.append({"path": path, "mode": "100644", "type": "blob", "sha": blob["sha"]})

    status, tree = _req("POST", f"{API}/git/trees", {"base_tree": base_tree, "tree": tree_entries})
    if status not in (200, 201):
        raise RuntimeError(f"Failed to create tree: {status} {tree}")

    status, new_commit = _req("POST", f"{API}/git/commits", {
        "message": commit_message,
        "tree": tree["sha"],
        "parents": [base_sha],
    })
    if status not in (200, 201):
        raise RuntimeError(f"Failed to create commit: {status} {new_commit}")

    status, updated_ref = _req("PATCH", f"{API}/git/refs/heads/{branch}", {"sha": new_commit["sha"]})
    if status != 200:
        raise RuntimeError(f"Failed to update ref: {status} {updated_ref}")

    return new_commit["sha"]
