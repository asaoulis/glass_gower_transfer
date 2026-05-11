
"""Convenience utilities for posterior sampling on held-out test data.

This module is intentionally small and pragmatic:
- Select random dataset indices from a test loader.
- Optionally align those indices across multiple loaders by underlying file path.
- Build an sbi-style conditioning object (tensor or dict of tensors).
- Sample a posterior (or ensemble posterior) via ``posterior.gen_samples``.
- Convert both truth and samples back to physical units using the provided scaler.

The main entrypoint for multi-model comparisons is ``sample_across_models``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import numpy as np
import torch


def _get_wrapped_base_dataset(ds: Any) -> Any:
	"""Best-effort unwrap of TransformingDataset-like wrappers."""
	base = getattr(ds, "base_ds", None)
	return base if base is not None else ds


def _resolve_cosmo_scaler(test_loader, param_scaler=None):
	"""Resolve the scaler that was actually used to scale theta in the loader.

	In this repo, ``TransformingDataset`` stores the cosmo scaler as
	``dataset.cosmo_scaler`` and applies it inside ``__getitem__``.

	If ``param_scaler`` is provided and differs from the dataset scaler, we
	prefer the dataset scaler (to keep truths consistent with the conditioning
	variables returned by the loader).
	"""
	ds = getattr(test_loader, "dataset", None)
	ds_scaler = getattr(ds, "cosmo_scaler", None)
	if ds_scaler is None:
		return param_scaler

	# If a scaler is explicitly provided and appears compatible, keep it.
	if param_scaler is None:
		return ds_scaler

	# Heuristic compatibility check for MinMaxScaler-like objects.
	ds_min = getattr(ds_scaler, "min", None)
	ds_max = getattr(ds_scaler, "max", None)
	p_min = getattr(param_scaler, "min", None)
	p_max = getattr(param_scaler, "max", None)
	if ds_min is not None and ds_max is not None and p_min is not None and p_max is not None:
		try:
			if np.allclose(np.asarray(ds_min), np.asarray(p_min)) and np.allclose(np.asarray(ds_max), np.asarray(p_max)):
				return param_scaler
		except Exception:
			pass

	# Fall back to dataset scaler (authoritative for the loader).
	return ds_scaler


def get_dataset_paths(loader) -> list[str] | None:
	"""Return dataset paths if available (used for alignment).

	Works with both H5CosmoDataset (``.paths``) and TransformingDataset wrappers
	(``.base_ds.paths``).
	"""
	ds = getattr(loader, "dataset", None)
	if ds is None:
		return None

	base = _get_wrapped_base_dataset(ds)
	paths = getattr(base, "paths", None)
	if paths is None:
		return None
	return list(paths)


def random_test_indices(test_loader, k: int, *, seed: int | None = None) -> list[int]:
	"""Choose ``k`` random indices from ``test_loader.dataset`` (no replacement)."""
	n = len(test_loader.dataset)
	if k <= 0:
		raise ValueError("k must be > 0")
	if k > n:
		raise ValueError(f"k={k} exceeds dataset length n={n}")
	rng = np.random.default_rng(seed)
	return rng.choice(n, size=k, replace=False).tolist()


def _split_item(item: Any) -> tuple[Any, Any]:
	if not isinstance(item, (tuple, list)) or len(item) != 2:
		raise ValueError("Dataset item must be a 2-tuple (data, theta) or (theta, data)")
	return item[0], item[1]


def _is_data_like(x: Any) -> bool:
	return isinstance(x, Mapping) or torch.is_tensor(x)


def _is_theta_like(x: Any) -> bool:
	return torch.is_tensor(x) and x.ndim in {1, 2}


def _ensure_data_theta_order(item: Any) -> tuple[Any, torch.Tensor]:
	"""Normalize dataset item to (data, theta)."""
	a, b = _split_item(item)
	# Common case: (data_dict, theta)
	if isinstance(a, Mapping) and torch.is_tensor(b):
		return a, b
	# Sometimes (theta, data_dict)
	if isinstance(b, Mapping) and torch.is_tensor(a):
		return b, a
	# Tensor data (rare), still try to disambiguate by shape
	if torch.is_tensor(a) and torch.is_tensor(b):
		# Heuristic: theta is 1D/2D small; data is often higher rank.
		if _is_theta_like(a) and not _is_theta_like(b):
			return b, a
		return a, b
	raise ValueError("Could not infer (data, theta) order from dataset item")


def _stack_data(items: Sequence[Any], *, inputs: Sequence[str] | None = None) -> Any:
	"""Stack a list of per-index data entries into a batch."""
	if not items:
		raise ValueError("No data items to stack")

	first = items[0]
	if isinstance(first, Mapping):
		keys = list(first.keys()) if inputs is None else list(inputs)
		out: dict[str, torch.Tensor] = {}
		for k in keys:
			out[k] = torch.stack([d[k] for d in items])
		return out

	if torch.is_tensor(first):
		return torch.stack(list(items))

	raise ValueError("Data items must be dict-like or torch.Tensor")


def _move_to_device_float32(x: Any, device: torch.device | str) -> Any:
	if torch.is_tensor(x):
		return x.to(device=device, dtype=torch.float32)
	if isinstance(x, Mapping):
		return {k: _move_to_device_float32(v, device) for k, v in x.items()}
	return x


@dataclass(frozen=True)
class PreparedBatch:
	"""Batch prepared from dataset indices."""

	indices: list[int]
	paths: list[str] | None
	theta_scaled: torch.Tensor  # [B, D]
	theta_phys: np.ndarray  # [B, D]
	x: Any  # conditioning batch: dict[str, Tensor] or Tensor


def prepare_test_batch(
	test_loader,
	param_scaler,
	*,
	inputs: Sequence[str] | None = None,
	indices: Sequence[int],
	verbose: bool = False,
) -> PreparedBatch:
	"""Build a conditioning batch and truth theta from dataset indices.

	Notes
	-----
	The underlying datasets in this repo return ``(data, cosmo)``.
	``cosmo`` is typically already scaled by the loader wrapper.

	Returns
	-------
	PreparedBatch
		- ``theta_scaled``: tensor shaped [B, D]
		- ``theta_phys``: numpy array shaped [B, D]
		- ``x``: conditioning batch (dict or tensor) shaped [B, ...]
	"""
	if indices is None:
		raise ValueError("indices must be provided")
	indices = list(indices)
	if len(indices) == 0:
		raise ValueError("indices must be non-empty")

	# NOTE: We do not iterate the DataLoader at all here.
	# Indexing the dataset directly avoids any accidental shuffling or
	# collation/batch-size quirks.
	dataset = test_loader.dataset
	base_ds = getattr(dataset, "base_ds", None)
	paths = get_dataset_paths(test_loader)

	# Scaler actually used by the dataset wrapper.
	cosmo_scaler = _resolve_cosmo_scaler(test_loader, param_scaler)
	if verbose and getattr(dataset, "cosmo_scaler", None) is not None and cosmo_scaler is not param_scaler:
		print("[prepare_test_batch] Using dataset.cosmo_scaler (authoritative for loader)")

	if verbose:
		print(f"Test loader batches: {len(test_loader)}")
		if paths is not None:
			print("Selected dataset paths:")
			for i in indices:
				if 0 <= i < len(paths):
					print(f"  Index {i}: {paths[i]}")

	# Build conditioning (x) from the *same* transform pipeline used in training.
	# Build truth (theta_phys) from base_ds (unscaled physical parameters).
	x_items: list[Any] = []
	theta_phys_items: list[torch.Tensor] = []
	theta_scaled_items: list[torch.Tensor] = []

	for i in indices:
		# Conditioning data: use dataset output directly.
		# For TransformingDataset this is already scaled exactly once.
		data_i, _ = _ensure_data_theta_order(dataset[i])

		x_items.append(data_i)

		# Truth theta: always from base_ds if available.
		if base_ds is not None:
			_, theta_phys_i = _ensure_data_theta_order(base_ds[i])
		else:
			# Fallback: best effort from the dataset itself.
			_, theta_phys_i = _ensure_data_theta_order(dataset[i])
			# If dataset theta is scaled, try to invert.
			if cosmo_scaler is not None:
				try:
					theta_phys_i = cosmo_scaler.inverse_transform(theta_phys_i)
				except Exception:
					pass

		theta_phys_items.append(theta_phys_i)

		# Also provide scaled theta when possible (useful for sanity checks).
		if cosmo_scaler is not None:
			theta_scaled_items.append(cosmo_scaler.transform(theta_phys_i))
		else:
			theta_scaled_items.append(theta_phys_i)

	theta_scaled = torch.stack(theta_scaled_items, dim=0)
	theta_phys = torch.stack(theta_phys_items, dim=0).detach().cpu().numpy()

	x = _stack_data(x_items, inputs=inputs)
	return PreparedBatch(
		indices=indices,
		paths=paths,
		theta_scaled=theta_scaled,
		theta_phys=theta_phys,
		x=x,
	)


def _build_posterior(model, *, prior=None):
	"""Build a posterior object from a LightningModule-like model."""
	if not hasattr(model, "build_posterior_object"):
		raise ValueError("Model does not implement build_posterior_object")
	try:
		return model.build_posterior_object(prior=prior)
	except TypeError:
		# Some models don't accept prior kwarg.
		return model.build_posterior_object()


def sample_posterior(
	posterior,
	x,
	*,
	num_points: int | None = None,
	param_scaler=None,
	n_samples: int = 10_000,
	device: str | torch.device | None = None,
	use_latent: bool = False,
	**mcmc_kwargs,
) -> np.ndarray:
	"""Sample posterior and return samples in physical units.

	Parameters
	----------
	posterior:
		Object exposing ``gen_samples(num_samples, x=..., use_latent=..., **kwargs)``.
	x:
		Conditioning batch: dict[str, Tensor] or Tensor with leading batch dim.
	num_points:
		If given, only use the first ``num_points`` items along the batch axis.
	param_scaler:
		Scaler with ``inverse_transform`` mapping scaled theta -> physical.
	n_samples:
		Number of posterior samples.
	device:
		Device used for sampling.
	use_latent:
		Passed through to ``gen_samples``.
	mcmc_kwargs:
		Passed through to ``gen_samples`` (e.g. ``num_chains``, ``warmup_steps``).

	Returns
	-------
	np.ndarray
		Samples in physical units with shape [n_samples, B, D].
	"""
	if device is None:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	device = torch.device(device) if not isinstance(device, torch.device) else device

	if isinstance(x, Mapping):
		x_batch = {k: v for k, v in x.items()}
		if num_points is not None:
			x_batch = {k: v[:num_points] for k, v in x_batch.items()}
	else:
		x_batch = x[:num_points] if num_points is not None else x

	x_batch = _move_to_device_float32(x_batch, device)

	samples = posterior.gen_samples(
		n_samples,
		x=x_batch,
		use_latent=use_latent,
		**mcmc_kwargs,
	)
	if isinstance(samples, tuple):
		samples = samples[0]

	# Keep stable shape: [n_samples, B, D]
	if samples.ndim == 2:
		samples = samples.unsqueeze(1)

	samples = samples.detach().cpu()
	if param_scaler is None:
		return samples.numpy()

	samples_phys_t = param_scaler.inverse_transform(samples)
	return samples_phys_t.detach().cpu().numpy()


def _paths_to_index(loader) -> dict[str, int] | None:
	paths = get_dataset_paths(loader)
	if paths is None:
		return None
	return {p: i for i, p in enumerate(paths)}


def align_indices_by_path(
	loaders: Mapping[str, Any],
	indices: Sequence[int],
	*,
	base_name: str | None = None,
) -> dict[str, list[int]]:
	"""Map a set of base indices onto each loader using underlying file paths."""
	if base_name is None:
		base_name = next(iter(loaders.keys()))
	base_loader = loaders[base_name]
	base_paths = get_dataset_paths(base_loader)
	if base_paths is None:
		# Cannot align; return same indices everywhere.
		return {name: list(indices) for name in loaders}

	selected_paths = [base_paths[i] for i in indices]
	out: dict[str, list[int]] = {}
	for name, loader in loaders.items():
		mapping = _paths_to_index(loader)
		if mapping is None:
			out[name] = list(indices)
			continue
		missing = [p for p in selected_paths if p not in mapping]
		if missing:
			raise ValueError(
				f"Loader '{name}' is missing {len(missing)} selected paths; "
				"test sets are not aligned."
			)
		out[name] = [mapping[p] for p in selected_paths]
	return out


def sample_across_models(
	model_configs: Mapping[str, Mapping[str, Any]],
	*,
	indices: Sequence[int] | None = None,
	n_indices: int | None = None,
	seed: int | None = None,
	n_samples: int = 10_000,
	num_points: int | None = None,
	device: str | torch.device | None = None,
	prior=None,
	use_latent: bool = False,
	align_by_path: bool = True,
	truth_atol: float = 1e-5,
	truth_rtol: float = 1e-5,
	verbose: bool = False,
	**mcmc_kwargs,
) -> dict[str, Any]:
	"""Sample posteriors for multiple models on the same underlying test cases.

	Parameters
	----------
	model_configs:
		Mapping model_name -> dict with keys:
		  - "model": lightning module
		  - "test_loader": DataLoader
		  - "scalers": dict containing "cosmo" scaler
		  - "inputs": optional list of data keys to use for conditioning
	indices / n_indices / seed:
		If ``indices`` is None, sample ``n_indices`` random indices from the
		first model's test loader.
	align_by_path:
		If True and datasets expose ``.paths``, align selected points across
		models by file path. This prevents mismatches when different splits
		reorder test sets.

	Returns
	-------
	dict with:
	  - "indices": selected indices in the *base* loader
	  - "paths": selected file paths (if available)
	  - "truth": physical truth theta [B, D]
	  - "samples": dict model_name -> samples_phys [n_samples, B', D]
	"""
	if not model_configs:
		raise ValueError("model_configs must be non-empty")

	# Resolve base selection.
	first_name = next(iter(model_configs.keys()))
	base_loader = model_configs[first_name]["test_loader"]
	if indices is None:
		if n_indices is None:
			raise ValueError("Provide either indices or n_indices")
		indices = random_test_indices(base_loader, n_indices, seed=seed)
	indices = list(indices)

	# Optionally align across loaders.
	loaders = {name: cfg["test_loader"] for name, cfg in model_configs.items()}
	aligned = (
		align_indices_by_path(loaders, indices, base_name=first_name)
		if align_by_path
		else {name: list(indices) for name in loaders}
	)

	# Prepare batches and check truth alignment.
	prepared: dict[str, PreparedBatch] = {}
	truth_ref: np.ndarray | None = None
	paths_ref: list[str] | None = None

	for name, cfg in model_configs.items():
		test_loader = cfg["test_loader"]
		param_scaler = cfg.get("scalers", {}).get("cosmo")
		param_scaler = _resolve_cosmo_scaler(test_loader, param_scaler)
		inputs = cfg.get("inputs")
		pb = prepare_test_batch(
			test_loader,
			param_scaler,
			inputs=inputs,
			indices=aligned[name],
			verbose=verbose,
		)
		prepared[name] = pb

		if truth_ref is None:
			truth_ref = pb.theta_phys
			paths_ref = pb.paths
		else:
			if not np.allclose(pb.theta_phys, truth_ref, rtol=truth_rtol, atol=truth_atol):
				raise ValueError(
					"Truth mismatch across models. This typically means the test loaders "
					"are not aligned (different paths/order), or the wrong scaler was used. "
					"Set align_by_path=True (recommended) or inspect loader paths."
				)

	# Sample each model.
	out_samples: dict[str, np.ndarray] = {}
	for name, cfg in model_configs.items():
		model = cfg["model"]
		test_loader = cfg["test_loader"]
		param_scaler = cfg.get("scalers", {}).get("cosmo")
		param_scaler = _resolve_cosmo_scaler(test_loader, param_scaler)
		pb = prepared[name]

		posterior = _build_posterior(model, prior=prior)

		# Default: sample all selected points.
		k = num_points if num_points is not None else len(pb.indices)
		if verbose:
			print(f"Sampling '{name}' with B={k}, n_samples={n_samples}")

		samples_phys = sample_posterior(
			posterior,
			pb.x,
			num_points=k,
			param_scaler=param_scaler,
			n_samples=n_samples,
			device=device,
			use_latent=use_latent,
			**mcmc_kwargs,
		)
		out_samples[name] = samples_phys

	base_paths = None
	if paths_ref is not None:
		base_paths = [paths_ref[i] for i in indices] if len(paths_ref) > 0 else None

	return {
		"indices": indices,
		"paths": base_paths,
		"truth": truth_ref,
		"samples": out_samples,
	}


def compute_s8(samples: np.ndarray, omega_m_idx: int = 0, sigma8_idx: int = 1) -> np.ndarray:
	"""Compute S8 from posterior samples in physical units.

	Expects samples shaped [..., D] where Omega_m and sigma8 indices are given.
	"""
	omega_m = samples[..., omega_m_idx]
	sigma8 = samples[..., sigma8_idx]
	return sigma8 * np.sqrt(omega_m / 0.3)


def augment_with_s8(samples: np.ndarray, omega_m_idx: int = 0, sigma8_idx: int = 1) -> np.ndarray:
	"""Insert S8 as a new column after Omega_m.

	Input:  (N, D) or (N, B, D)
	Output: (N, D+1) or (N, B, D+1)
	"""
	s8 = compute_s8(samples, omega_m_idx=omega_m_idx, sigma8_idx=sigma8_idx)
	# Promote s8 to have a trailing singleton dim for concatenation.
	s8 = np.expand_dims(s8, axis=-1)
	return np.concatenate([samples[..., :1], s8, samples[..., 1:]], axis=-1)
