"""Fixed Gower test-set support (opt-in).

Lets all experiments share ONE fixed test set keyed on the Gower simulation suite id
(the ``sim_num`` written into ``output_<sim_num>_...h5`` filenames, == the Gower Serial
Number). The list of test sim_ids is locked to a small JSON file checked into the repo
(``config/fixed_test_sets/gower_test_ids.json``); the data pipeline reads it and forces
exactly those cosmologies into the test split (see ``data_selection.split_by_cosmology``).

This is EXTRA, opt-in behaviour: it is only active when ``config.fixed_test_sim_ids`` is
set (a path to a lock-file JSON or an explicit list of ints). When unset (``None``, the
default) the normal cosmology-based split is used unchanged.

The lock-file can be (re)generated against an on-disk dataset via the CLI:

    python -m src.ml.data.fixed_test_set \
        --out config/fixed_test_sets/gower_test_ids.json \
        --patterns "/share/.../gower_mocks/output_*.h5" --min-id 193
"""
import argparse
import json
import os
from typing import List, Optional, Sequence, Set, Union

from .data_selection import collect_paths, extract_cosmo_index


def load_fixed_test_ids(path: str) -> Set[int]:
    """Read a lock-file of Gower test sim_ids.

    Accepts either the canonical ``{"sim_ids": [...]}`` object or a bare JSON list.
    """
    with open(path, "r") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        ids = payload.get("sim_ids", [])
    else:
        ids = payload
    return set(int(i) for i in ids)


def resolve_fixed_test_ids(
    spec: Optional[Union[str, Sequence[int]]],
) -> Optional[Set[int]]:
    """Resolve the ``fixed_test_sim_ids`` config value to a set of ints (or None).

    - ``None`` => None (feature off).
    - ``str``  => path to a JSON lock-file (loaded via ``load_fixed_test_ids``).
    - list/tuple/set => used directly as the id set.
    """
    if spec is None:
        return None
    if isinstance(spec, str):
        return load_fixed_test_ids(spec)
    return set(int(i) for i in spec)


def write_fixed_test_ids(
    out_path: str,
    patterns: Union[str, Sequence[str]],
    min_id: int = 193,
    max_id: Optional[int] = None,
) -> List[int]:
    """Lock the fixed test sim_ids derived from an on-disk dataset to a JSON file.

    Globs ``patterns``, extracts each file's Gower sim_id via the SAME
    ``extract_cosmo_index`` the split uses, keeps the unique ids with
    ``min_id <= id`` (and ``id <= max_id`` when given), and writes a sorted JSON
    ``{"min_id", "max_id", "sim_ids"}`` to ``out_path``. Returns the sorted id list.
    """
    all_paths = collect_paths(patterns)
    ids: Set[int] = set()
    for p in all_paths:
        cidx = extract_cosmo_index(p)
        if cidx < min_id:
            continue
        if max_id is not None and cidx > max_id:
            continue
        ids.add(cidx)
    sim_ids = sorted(ids)
    derived_max = sim_ids[-1] if sim_ids else None
    payload = {
        "min_id": int(min_id),
        "max_id": int(max_id) if max_id is not None else derived_max,
        "sim_ids": sim_ids,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"[fixed_test_set] wrote {len(sim_ids)} sim_ids to {out_path}")
    return sim_ids


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Lock a fixed list of Gower test sim_ids to a JSON file."
    )
    parser.add_argument("--out", required=True, help="Output JSON lock-file path.")
    parser.add_argument(
        "--patterns", required=True, nargs="+", help="Glob pattern(s) for output_*.h5 mocks."
    )
    parser.add_argument(
        "--min-id", type=int, default=193, help="Inclusive lower bound on sim_id (default 193)."
    )
    parser.add_argument(
        "--max-id", type=int, default=None, help="Inclusive upper bound on sim_id (default: derived)."
    )
    args = parser.parse_args()
    write_fixed_test_ids(args.out, args.patterns, min_id=args.min_id, max_id=args.max_id)


if __name__ == "__main__":
    _main()
