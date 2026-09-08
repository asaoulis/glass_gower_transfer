#!/usr/bin/env python3
"""PreToolUse BLIND guard: deny any tool access to the raw real-data posteriors.

Wire on `Read|Bash|Grep|Glob|Edit|Write|NotebookEdit` (see the settings snippet in
.claude/runs/eval-and-viz/unblinding-prep/artifacts/guard_blind_settings_snippet.json). Exit 2 = deny
(reason fed back to the agent), exit 0 = allow. This is a tripwire on the obvious channels, not a
sandbox: the agent is also bound by the protocol never to compute a posterior mean/std of an
observation outside src/blind/standardise.py.

DENIED whenever the tool input mentions:
  * the local BLIND STORE (default /data/alex/unblinding/blind_store; env UNBLIND_BLIND_ROOT),
  * any `_truthkey/` directory (sim identity of a mock-as-real observation),
  * the cluster blind dirs `unblinding_blind/` and `checkpoints/<exp>/external/<label>/` when reached
    through the gatekeeper's read verbs (`ssh hypatia-glass view|logs …`),
  * a raw observation-posterior filename `external_posterior_samples_*` outside the two entry points.

ALLOWED entry points (the only commands that may name the blind store):
  * `python … -m src.blind.standardise` / `scripts/plot_blind_posteriors.py` /
    `scripts/sample_observation.py` (they READ the store but emit standardised arrays only),
  * `run_remote.py fetch … --out_dir <blind store>` (moves raw files IN, never reads them),
  * `ls`/`find`/`du` of the store (filenames only; no content).
"""
from __future__ import annotations

import json
import os
import re
import sys

BLIND_ROOT = os.environ.get("UNBLIND_BLIND_ROOT", "/data/alex/unblinding/blind_store")
DENY_PATTERNS = [
    (re.escape(BLIND_ROOT), "the blind store"),
    (r"_truthkey/", "a truth-key sidecar"),
    (r"unblinding_blind/", "the cluster blind subdir"),
    (r"external_posterior_samples_", "a raw observation posterior file"),
    (r"pooled_posterior_samples_", "a raw POOLED observation posterior file"),
    (r"checkpoints/pooled/[^/\s]+/external/", "the cluster pooled-observation dir"),
]
ALLOWED_ENTRY = [
    r"-m\s+src\.blind\.standardise", r"scripts/plot_blind_posteriors\.py", r"scripts/sample_observation\.py",
    r"run_remote\.py\s+fetch\b", r"src/blind/standardise\.py\s*$",
    # Tier-1 report: opens the sidecar for its sim_id ONLY (never printed) to drop that cosmology from the null
    r"scripts/tier1_report\.py",
]
# read-only listing verbs are fine on the blind store (filenames only)
LISTING_ONLY = re.compile(r"^\s*(ls|find|du|tree|stat|wc\s+-l)\b")


def _text_of(data: dict) -> str:
    ti = data.get("tool_input", {}) or {}
    parts = []
    for k in ("command", "file_path", "path", "pattern", "notebook_path", "content", "new_string", "old_string"):
        v = ti.get(k)
        if isinstance(v, str):
            parts.append(v)
    return "\n".join(parts)


def main() -> int:
    data = json.load(sys.stdin)
    tool = data.get("tool_name", "")
    text = _text_of(data)
    if not text:
        return 0
    hits = [why for pat, why in DENY_PATTERNS if re.search(pat, text)]
    if not hits:
        return 0
    # the mock-as-real gate needs external_posterior_samples_ mentioned in scripts (source code) --
    # only deny when a PATH under the blind root / cluster blind dirs is actually referenced,
    # or the filename is used with a reader
    if tool == "Bash":
        cmd = text
        if any(re.search(p, cmd) for p in ALLOWED_ENTRY):
            return 0
        if LISTING_ONLY.match(cmd) and "external_posterior_samples_" not in cmd:
            return 0
        if "ssh" in cmd and not re.search(r"\b(view|logs)\b", cmd):
            # gatekeeper verbs other than view/logs cannot read content
            if not re.search(re.escape(BLIND_ROOT) + r"|_truthkey/", cmd):
                return 0
    if tool in ("Edit", "Write", "NotebookEdit"):
        # editing source that merely MENTIONS the patterns is fine; touching the store itself is not
        fp = (data.get("tool_input", {}) or {}).get("file_path", "") or (data.get("tool_input", {}) or {}).get("notebook_path", "")
        if not re.search(re.escape(BLIND_ROOT) + r"|_truthkey/", fp or ""):
            return 0
    if tool in ("Grep", "Glob"):
        path = (data.get("tool_input", {}) or {}).get("path", "") or ""
        if not re.search(re.escape(BLIND_ROOT) + r"|_truthkey/|unblinding_blind/", path):
            return 0
    print(f"BLOCKED by guard_blind: this would touch {', '.join(hits)}. Raw observation posteriors "
          f"are BLIND; use `python -m src.blind.standardise` / scripts/plot_blind_posteriors.py (standardised "
          f"outputs only). If you are the user and mean it, run the command yourself.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
