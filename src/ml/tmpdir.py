"""Redirect the process-wide temp directory onto the models filesystem.

Why this exists
---------------
Lightning's ``_atomic_save`` (``lightning_fabric/utilities/cloud_io.py:92``)
writes a checkpoint under fsspec's ``fs.transaction``, which opens the target
with ``autocommit=False``. That path calls ``tempfile.mkstemp()`` with **no**
``dir=`` argument (``fsspec/implementations/local.py:401``), so the *entire*
serialised checkpoint -- ~106 MB for the BGP hybrid rows -- is written to the
compute node's ``/tmp`` first and only then ``shutil.move``d onto the shared
store (``local.py:441``).

On hypatia the compute nodes' ``/tmp`` is small, so concurrent training jobs
each staging a ~106 MB checkpoint there exhaust it and die with
``OSError: [Errno 28] No space left on device`` raised *inside* fsspec's
``LocalFileOpener.write`` -- while the destination share has terabytes free.
That is why the failure looked like "gpu5 is full" when it never was: reads
worked everywhere, direct writes to gpu5 from a compute node worked (the
prebake wrote 800 files during the same window), and the first few epochs'
checkpoints landed fine before /tmp filled up.

Pointing the temp dir at ``{base_path}/.tmp`` fixes it, and as a bonus makes
fsspec's ``shutil.move`` a same-filesystem rename instead of a cross-device
copy.

Call this from every entry point that checkpoints, before the ``pl.Trainer``
runs: ``src/ml/models/utils.py:fit_model`` and the embeddings NDE trainer in
``src/ml/embeddings/embeddings_utils.py``.
"""

import os
import tempfile


def redirect_tempdir(base_path: str) -> str:
    """Point this process's temp dir at ``{base_path}/.tmp``; return that path.

    ``base_path`` is the models root (the same one the ``ModelCheckpoint``
    ``dirpath`` is built from), so the temp dir lands on the *same* filesystem
    as the checkpoints -- but deliberately NOT under ``checkpoints/``, which
    ``get_best_checkpoint`` and the cluster ``status`` ckpt-counter both walk.
    """
    d = os.path.join(base_path, ".tmp")
    os.makedirs(d, exist_ok=True)

    # `tempfile.tempdir` is the assignment that actually matters, not $TMPDIR:
    # `tempfile.gettempdir()` caches its answer into `tempfile.tempdir` on the
    # first call, and `mkstemp(dir=None)` reads that cache. By the time training
    # starts, wandb/torch/matplotlib imports have already called it -- so setting
    # only os.environ["TMPDIR"] here would be a SILENT NO-OP.
    tempfile.tempdir = d
    # ...and $TMPDIR for anything spawned as a subprocess afterwards.
    os.environ["TMPDIR"] = d

    print(f"[tmpdir] staging temp files under {d} "
          f"(fsspec stages checkpoints here before moving them into place)")
    return d
