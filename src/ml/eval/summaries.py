"""Summary-space extraction for the OOD / misspecification diagnostic (``eval.py --mode summaries``).

For ONE base experiment and each requested training repeat, this dumps the compressor's summary
vector ``z`` (the flow context — ``embedding_net`` output with ``only_return_mu=True``, exactly what
``compute_embeddings`` in the embeddings pipeline caches) for

* ``_train``   — a random subset of the model's own TRAIN split (the reference cloud),
* ``_idtest``  — the model's strictly held-out in-distribution TEST split (the null),
* each variate — its test set, built with the ORIGINAL training scalers injected
  (``misspec.build_variate_test_loader``; never refit on a variate),

then scores every variate against the train cloud + ID null with ``src/ml/eval/ood.py`` and writes
per-event scores/p-values next to the summaries. Everything is PER MODEL: for an eval-time
ensemble each member is extracted and scored separately (own encoder = own summary space); only the
calibrated outputs (p-values, AUROC, permutation p) may be aggregated across members/repeats.

Outputs under ``{base_path}/checkpoints/<exp>/summaries[_shared]/``:
    <variate>/summaries_<match>[_m<j>].npz   z, theta (scaled), test_files, sim_ids, aug_ids, params,
                                              + per-event OOD scores/p-values (``ood_*`` keys)
    <variate>/ood_<match>[_m<j>].json        dataset-level OOD statistics
    ood_summary_<match>.json                  all variates x members for the repeat
"""
import json
import os
import traceback
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from ..data.data_loaders import build_nested_keys_from_quantities
from ..data.data_selection import extract_cosmo_index
from ..utils import build_ensemble_model_from_checkpoints, prepare_data_parameters
from .misspec import (
    VARIATE_SETS,
    _load_experiment_config,
    _probe_variate_file,
    _wrap_paths_as_loader,
    build_variate_test_loader,
    derive_test_id_pool,
    resolve_shared_pool,
)
from .ood import OODReference
from .utils import _parse_aug_id, _resolve_test_paths, _to_json_compatible


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _member_encoders(model) -> List[torch.nn.Module]:
    """One encoder per model (ensemble members, or the single model)."""
    members = list(getattr(model, "members", [])) or [model]
    encs = []
    for m in members:
        core = getattr(m, "model", m)
        enc = getattr(core, "embedding_net", None)
        if enc is None:
            raise AttributeError("model has no embedding_net to extract summaries from")
        if hasattr(enc, "only_return_mu"):
            enc.only_return_mu = True
        enc.eval()
        encs.append(enc)
    return encs


@torch.no_grad()
def encode_loader(encoders: Sequence[torch.nn.Module], loader, max_items: Optional[int] = None):
    """Run every encoder over the loader ONCE (one read of the maps), returning per-encoder z + theta."""
    dev = _device()
    encs = [e.to(dev) for e in encoders]
    zs = [[] for _ in encs]
    thetas = []
    n = 0
    for batch in loader:
        data, theta = batch[0], batch[1]
        if isinstance(data, dict):
            data = {k: v.to(dev).float() for k, v in data.items()}
        else:
            data = data.to(dev).float()
        for i, e in enumerate(encs):
            z = e(data)
            if isinstance(z, (tuple, list)):
                z = z[0]
            zs[i].append(z.detach().float().cpu())
        thetas.append(theta.detach().float().cpu())
        n += theta.shape[0]
        if max_items is not None and n >= max_items:
            break
    z_out = [torch.cat(z, 0).numpy() for z in zs]
    th = torch.cat(thetas, 0).numpy()
    if max_items is not None:
        z_out = [z[:max_items] for z in z_out]
        th = th[:max_items]
    return z_out, th


def _ids_payload(test_paths):
    files = [os.path.basename(p) for p in test_paths]
    return {
        "test_files": np.array(files),
        "sim_ids": np.array([extract_cosmo_index(p) for p in test_paths], dtype=np.int64),
        "aug_ids": np.array([_parse_aug_id(f) for f in files], dtype=np.int64),
    }


def _save_summaries(path, z, theta, test_paths, param_names, extra=None):
    payload = {"z": z.astype(np.float32), "theta": theta.astype(np.float32), "params": np.array(list(param_names))}
    if test_paths is not None and len(test_paths) == z.shape[0]:
        payload.update(_ids_payload(test_paths))
    if extra:
        payload.update({k: np.asarray(v) for k, v in extra.items()})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **payload)


def _ood_line(tag: str, ds: Dict) -> str:
    keys = ["knn_auroc", "cond_knn_auroc", "mahalanobis_auroc", "knn_mean_p", "cond_knn_mean_p",
            "c2st_auroc_gbm", "cond_c2st_auroc_gbm", "mmd_pvalue", "cond_mmd_pvalue"]
    return f"[ood] {tag}: " + " ".join(f"{k}={ds[k]:.3f}" for k in keys if k in ds) + f" n={ds.get('n_query')}"


def run_summary_extraction(
    base_experiment: str,
    repeat_indices: Sequence[int] = (0,),
    variate_set: Optional[str] = None,
    variate_names: Optional[Sequence[str]] = None,
    variates: Optional[List[Dict]] = None,
    max_test_files: Optional[int] = None,
    test_id_source: str = "heldout",
    max_train_files: int = 20000,
    test_shape_noise_idx=(0, (0, 1)),
    data_patterns_override: Optional[str] = None,
    out_subdir: str = "summaries",
    ood_k: int = 10,
    ood_k_theta: int = 50,
    ood_n_perm: int = 200,
    run_ood: bool = True,
    seed: int = 0,
    base_path_override: Optional[str] = None,
    fixed_test_ids_override: Optional[str] = None,
    max_trainval_cosmos_override: Optional[int] = None,
):
    """``test_id_source``: 'heldout' (lock file or the model's own held-out split), 'shared' (ids
    common to the OOD variates), or 'all' (every on-disk cosmology of each variate — valid when the
    base model never trained on that suite, e.g. a GLASS foundation scored on Gower stores, and the
    natural mode for a real-data file). ``fixed_test_ids_override`` / ``max_trainval_cosmos_override``
    re-split the base experiment's store (e.g. give a GLASS-trained encoder the Gower chain's lock +
    300-cosmology train set so its train cloud / null are the NLE Stage-B's)."""
    from ..models.utils import apply_repeat_config
    from .utils import load_best_model_and_build_posterior

    if variates is None:
        variates = VARIATE_SETS[variate_set] if variate_set else []
    if variate_names:
        wanted = set(variate_names)
        missing = wanted - {v["name"] for v in variates}
        if missing:
            raise KeyError(f"variate name(s) {sorted(missing)} not in set '{variate_set}'")
        variates = [v for v in variates if v["name"] in wanted]
    repeat_indices = [int(r) for r in repeat_indices]
    if test_id_source == "shared" and out_subdir == "summaries":
        out_subdir = "summaries_shared"

    print(f"[summaries] setup: loading config for '{base_experiment}'", flush=True)
    cfg0 = _load_experiment_config(base_experiment)
    if base_path_override:
        cfg0.base_path = base_path_override
    param_names = list(cfg0.cosmo_param_names)
    eb_variant = getattr(cfg0, "eb_map_variant", None)
    nested_keys = build_nested_keys_from_quantities(list(cfg0.dataset_quantities), eb_variant)
    out_root = os.path.join(cfg0.base_path, "checkpoints", cfg0.experiment_name, out_subdir)
    print(f"[summaries] params={param_names} eb_variant={eb_variant} out_root={out_root} "
          f"variates={[v['name'] for v in variates]}", flush=True)

    shared_pool = resolve_shared_pool(variates) if test_id_source == "shared" else None
    if test_id_source not in ("heldout", "shared", "all"):
        raise ValueError(f"test_id_source must be 'heldout', 'shared' or 'all', got {test_id_source!r}")
    if test_id_source == "all" and out_subdir == "summaries":
        out_subdir = "summaries_all"
        out_root = os.path.join(cfg0.base_path, "checkpoints", cfg0.experiment_name, out_subdir)

    all_results: Dict[str, Dict] = {}
    for r in repeat_indices:
        cfg = _load_experiment_config(base_experiment)
        cfg.test_shape_noise_idx = list(test_shape_noise_idx)
        if base_path_override:
            cfg.base_path = base_path_override
        if data_patterns_override:
            cfg.data_patterns = data_patterns_override
        if fixed_test_ids_override:
            spec = fixed_test_ids_override
            if "/" not in spec and not os.path.exists(spec):
                spec = os.path.join("config", "fixed_test_sets", f"{spec}.json")   # bare lock name
            cfg.fixed_test_sim_ids = spec
        if max_trainval_cosmos_override is not None:
            cfg.max_trainval_cosmos = int(max_trainval_cosmos_override)
            cfg.match_num_cosmo = False   # the checkpoint dirs are named by the ORIGINAL config
        repeat_match, _ = apply_repeat_config(cfg, r)
        cfg.match_string = repeat_match
        print(f"[summaries] repeat={r} match_string={repeat_match} data_patterns={cfg.data_patterns}", flush=True)

        scalers, train_loader, _val_loader, id_test_loader = prepare_data_parameters(cfg)
        key_scalers, cosmo_scaler = scalers["data"], scalers["cosmo"]

        n_ens = int(getattr(cfg, "ensemble_repeats", 1) or 1)
        if n_ens > 1:
            model = build_ensemble_model_from_checkpoints(
                cfg, id_test_loader, match_string=cfg.match_string,
                member_test_loaders=[id_test_loader] * n_ens,
            )
        else:
            loaded = load_best_model_and_build_posterior(cfg, ds_string_match=cfg.match_string, data_parameters=id_test_loader)
            model = loaded[0] if loaded else None
        if model is None:
            print(f"[summaries] repeat {r}: no checkpoints for {cfg.match_string} — skipping.", flush=True)
            continue
        encoders = _member_encoders(model)
        tags = [f"{repeat_match}_m{j}" for j in range(len(encoders))] if len(encoders) > 1 else [repeat_match]
        print(f"[summaries] repeat {r}: {len(encoders)} encoder(s)", flush=True)

        # --- TRAIN cloud (no augmentation: plain read through the training scalers) -------------
        train_paths = list(_resolve_test_paths(train_loader) or [])
        rng = np.random.default_rng(seed + r)
        if max_train_files and len(train_paths) > max_train_files:
            train_paths = [train_paths[i] for i in sorted(rng.choice(len(train_paths), max_train_files, replace=False))]
        bs = min(64, int(getattr(cfg, "test_batch_size", None) or getattr(cfg, "batch_size", 64)))
        train_plain, _ = _wrap_paths_as_loader(
            train_paths, nested_keys, param_names, key_scalers, cosmo_scaler,
            batch_size=bs, num_workers=4, eb_noise_norm=getattr(cfg, "eb_noise_norm", None),
        )
        print(f"[summaries] repeat {r}: encoding TRAIN cloud ({len(train_paths)} files)", flush=True)
        z_train, th_train = encode_loader(encoders, train_plain)
        for j, tag in enumerate(tags):
            _save_summaries(os.path.join(out_root, "_train", f"summaries_{tag}.npz"), z_train[j], th_train, train_paths, param_names)

        # --- ID held-out test split (the null) -----------------------------------------------
        id_paths = list(_resolve_test_paths(id_test_loader) or [])
        print(f"[summaries] repeat {r}: encoding ID held-out test split ({len(id_paths)} files)", flush=True)
        z_id, th_id = encode_loader(encoders, id_test_loader)
        for j, tag in enumerate(tags):
            _save_summaries(os.path.join(out_root, "_idtest", f"summaries_{tag}.npz"), z_id[j], th_id, id_paths, param_names)

        refs = None
        if run_ood:
            refs = [OODReference.fit(z_train[j], z_id[j], theta_train=th_train, theta_id=th_id,
                                     k=ood_k, k_theta=ood_k_theta, seed=seed) for j in range(len(encoders))]
            # self-check: the ID split against itself must read as null (AUROC 0.5 by construction)

        if test_id_source == "all":
            test_id_pool = None   # build_variate_test_loader falls back to ALL on-disk cosmologies
        else:
            test_id_pool = shared_pool if shared_pool is not None else derive_test_id_pool(cfg, id_test_loader, r)

        for variate in variates:
            name = variate["name"]
            try:
                vkeys = nested_keys
                if variate.get("eb_variant") is not None:
                    vkeys = build_nested_keys_from_quantities(list(cfg.dataset_quantities), variate["eb_variant"])
                loader, meta = build_variate_test_loader(
                    variate["patterns"], vkeys, param_names, key_scalers, cosmo_scaler,
                    test_id_pool=test_id_pool, test_shape_noise_idx=test_shape_noise_idx,
                    batch_size=bs, max_test_files=max_test_files,
                    eb_noise_norm=getattr(cfg, "eb_noise_norm", None), inject=variate.get("inject"),
                )
                missing_keys, present_cosmo, probe_path = _probe_variate_file(meta["test_paths"], vkeys, param_names)
                if missing_keys:
                    raise RuntimeError(f"data keys missing from {probe_path}: {missing_keys}")
                missing_params = [p for p in param_names if p not in present_cosmo]
                exclude = list(variate.get("exclude_params", []))
                print(f"[summaries] {name}@r{r}: n_test={meta['n_test_files']} ({meta['n_test_cosmologies']} cosmologies, "
                      f"fixed_lock={meta['test_ids_from_fixed_lock']}) missing_params={missing_params} exclude={exclude}", flush=True)
                z_v, th_v = encode_loader(encoders, loader)
                theta_dims = [i for i, p in enumerate(param_names) if p not in missing_params and p not in exclude]
                for j, tag in enumerate(tags):
                    extra, ds = {}, None
                    if refs is not None:
                        res = refs[j].evaluate(z_v[j], th_v, theta_dims=theta_dims, n_perm=ood_n_perm, seed=seed)
                        extra = {f"ood_{k}": v for k, v in res["per_event"].items()}
                        ds = dict(res["dataset"])
                        ds.update({"variate": name, "repeat": r, "member": j, "match": tag,
                                   "n_test_cosmologies": meta["n_test_cosmologies"],
                                   "missing_params": missing_params, "excluded_params": exclude})
                        os.makedirs(os.path.join(out_root, name), exist_ok=True)
                        with open(os.path.join(out_root, name, f"ood_{tag}.json"), "w") as fh:
                            json.dump(_to_json_compatible(ds), fh, indent=2)
                        print(_ood_line(f"{name}@r{r} m{j}", ds), flush=True)
                        all_results[f"{name}@{tag}"] = ds
                    _save_summaries(os.path.join(out_root, name, f"summaries_{tag}.npz"), z_v[j], th_v, meta["test_paths"], param_names, extra=extra)
            except Exception as e:
                print(f"[summaries] {name}@r{r}: FAILED — {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
                all_results[f"{name}@{repeat_match}"] = {"error": f"{type(e).__name__}: {e}"}
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        os.makedirs(out_root, exist_ok=True)
        with open(os.path.join(out_root, f"ood_summary_{repeat_match}.json"), "w") as fh:
            json.dump(_to_json_compatible({k: v for k, v in all_results.items() if k.endswith(repeat_match) or f"@{repeat_match}_m" in k}), fh, indent=2)
        del model, encoders
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n[summaries] ============ OOD SUMMARY ============", flush=True)
    for key, ds in all_results.items():
        if "error" in ds:
            print(f"[ood] {key}: ERROR {ds['error']}", flush=True)
        else:
            print(_ood_line(key, ds), flush=True)
    return all_results
