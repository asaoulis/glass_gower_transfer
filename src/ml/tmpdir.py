"""Redirect the process-wide temp directory onto the models filesystem.

Lightning stages every checkpoint through tempfile.mkstemp() on the compute node's /tmp before
moving it to the shared store. Setting tempfile.tempdir keeps concurrent jobs from filling /tmp.
Call it before the pl.Trainer runs in every entry point that checkpoints.
"""

import os
import tempfile


def redirect_tempdir(base_path: str) -> str:
    """Point this process's temp dir at {base_path}/.tmp and return that path.

    base_path is the models root, so the temp dir shares the checkpoints' filesystem but stays
    outside checkpoints/, which get_best_checkpoint walks.
    """
    d = os.path.join(base_path, ".tmp")
    os.makedirs(d, exist_ok=True)

    # Set tempfile.tempdir, not only $TMPDIR: gettempdir() caches its first answer.
    tempfile.tempdir = d
    os.environ["TMPDIR"] = d

    print(f"[tmpdir] staging temp files under {d} "
          f"(fsspec stages checkpoints here before moving them into place)")
    return d
