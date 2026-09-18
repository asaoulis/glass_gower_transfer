"""Ingestion of EXTERNAL full-sky matter shells into the KiDS-Legacy forward model.

Non-protected: nothing here edits `src/cosmology/` — it subclasses and calls it.

Currently one source: the GowerSt2 / `Flamingo_big` box (a 1250 Mpc/h PKDGRAV3 run at the
FLAMINGO cosmology), shipped as two stacked HEALPix delta cubes -- a gravity-only one and a
FLAMINGO-baryonified one sharing the same phases.
"""

from .gowerst2 import (  # noqa: F401
    FLAMINGO_BIG,
    ShellCubeSimulator,
    load_external_shells,
    prepare_external_backend,
)

__all__ = [
    "FLAMINGO_BIG",
    "ShellCubeSimulator",
    "load_external_shells",
    "prepare_external_backend",
]
