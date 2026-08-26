"""Archived experiment definitions -- pre-BGP campaigns, kept only so old runs stay reproducible.

Nothing in here is part of the current BGP analysis. The live configs are:

    config/kids_legacy_bgp.py   the BGP campaign (the current production analysis)
    config/kids_legacy.py       shared factories/constants the BGP suite imports
    config/kids_legacy_novd.py  shared encoder kwargs the BGP suite imports
    config/default.py           the default ConfigDict every experiment is layered onto

These modules are still merged into the experiment registry by train.py / eval.py, so archived
rows remain launchable, but they should not be extended -- add new work to kids_legacy_bgp.py.
"""
