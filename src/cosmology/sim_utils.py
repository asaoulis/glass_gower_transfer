from pathlib import Path

import h5py
import numpy as np

from src.ml.data.data_loading import load_cosmo_params


def resolve_systematics_model(args) -> str:
	"""Resolve the systematics model name from CLI args.

	- If `--systematics-model auto`, uses `nla` iff `--kids-systematics`.
	- Otherwise returns the provided model string.
	"""
	model = getattr(args, "systematics_model", "auto")
	if model == "auto":
		return "nla" if getattr(args, "kids_systematics", False) else "none"
	return model


def build_systematics(model: str, *, systematics_cls, cosmo, ia_params: dict, shear_bias: dict, sigma_e_base: np.ndarray):
	"""Build systematics object + derived inputs for the simulator.

	Returns:
		(systematics_or_none, sigma_e_sim, shift_nz)
	"""
	if model == "none":
		return None, np.zeros_like(sigma_e_base), False
	if model == "nla":
		systematics = systematics_cls(
			shear_bias=shear_bias,
			nla=ia_params,
			cosmo=cosmo,
		)
		return systematics, sigma_e_base.copy(), True
	raise ValueError(f"Unknown systematics model: {model!r}")


def prepare_glass_backend(
	sim_num: int,
	*,
	rng: np.random.Generator,
	output_dir: Path,
	cosmo_prior,
	prior_ranges: dict,
	sim_grid: dict,
	los_grid: dict,
	camb_limits: dict,
	cache: dict,
):
	"""Prepare GLASS matter+shells+cosmology for a given `sim_num`.

	Behavior:
	- Fixes cosmology+nuisance params per `sim_num`.
	- If an output file already exists (e.g. resuming after a crash), loads
	  the saved parameter dict from disk via `load_root_param_dict`.
	- Recomputes CAMB->GLASS spectra for the chosen parameters and caches the
	  expensive products (shells, glass_cls) per `sim_num`.
	"""
	if sim_num not in cache:
		param_dict = load_root_param_dict(output_dir, sim_num, cosmo_params=None)
		if param_dict is None:
			sampled_cosmo_params = cosmo_prior.draw_param_dict_sample(rng=rng)
			nuisance_params = {
				"a_ia": float(rng.uniform(*prior_ranges["a_ia"])),
				"b_ia": float(rng.uniform(*prior_ranges["b_ia"])),
			}
			param_dict = {
				**sampled_cosmo_params,
				**nuisance_params,
			}

		from src.cosmology.mpi_camb import (
			compute_camb_glass_in_child_npz_subproc,
			load_camb_child_pickle,
		)

		npz_out_path = compute_camb_glass_in_child_npz_subproc(
			param_dict,
			sim_grid["lmax"],
			los_grid["zmin"],
			los_grid["zmax"],
			los_grid["dx"],
			mem_limit_gb=camb_limits["mem_limit_gb"],
			timeout_s=camb_limits["timeout_s"],
			sim_tag=f"sim{sim_num}",
		)
		shells, glass_cls = load_camb_child_pickle(npz_out_path, remove_after_load=True)
		cache[sim_num] = {
			"param_dict": param_dict,
			"shells": shells,
			"glass_cls": glass_cls,
		}

	param_dict = cache[sim_num]["param_dict"]
	shells = cache[sim_num]["shells"]
	glass_cls = cache[sim_num]["glass_cls"]

	from cosmology import Cosmology
	import glass
	from src.cosmology import parameters

	cosmo, pars = parameters.build_cosmology(param_dict)
	glass_cls_discretized = glass.discretized_cls(
		glass_cls,
		nside=sim_grid["nside"],
		lmax=sim_grid["lmax"],
		ncorr=1,
	)
	fields = glass.lognormal_fields(shells)
	gls = glass.solve_gaussian_spectra(fields, glass_cls_discretized)
	matter = glass.generate(fields, gls, sim_grid["nside"], ncorr=1, rng=rng)
	cosmo = Cosmology.from_camb(pars)
	return {
		"param_dict": param_dict,
		"shells": shells,
		"matter": matter,
		"cosmo": cosmo,
	}


def prepare_gower_backend(
	sim_num: int,
	*,
	rng: np.random.Generator,
	loader,
	prior_ranges: dict,
	sim_grid: dict,
):
	"""Prepare GowerStreet matter+shells+cosmology for a given `sim_num`.

	Matches legacy behavior: nuisance params are re-sampled each call.
	"""
	nuisance_params = {
		"a_ia": float(rng.uniform(*prior_ranges["a_ia"])),
		"b_ia": float(rng.uniform(*prior_ranges["b_ia"])),
	}
	param_dict = loader.get_params_from_sim_id(sim_num, extra_params=nuisance_params)
	shells, matter, cosmo = loader.load_shells_matter_and_cosmology(sim_num, nside=sim_grid["nside"])
	_, pars, _ = loader.get_simulation_cosmology(sim_num, nuisance_params)

	from cosmology import Cosmology
	cosmo = Cosmology.from_camb(pars)
	return {
		"param_dict": param_dict,
		"shells": shells,
		"matter": matter,
		"cosmo": cosmo,
	}


def load_root_param_dict(
	output_dir: Path,
	sim_num: int,
	cosmo_params=None,
	map_root: str = "output",
):
	"""
	Look for the 'root' simulation file:
		{map_root}_{sim_num}_out0_rot0_0.h5
	If it exists, load the cosmology parameters via load_cosmo_params and
	return them as a plain Python dict (param_name -> value).

	Returns:
		dict of parameters if file exists, otherwise None.
	"""
	root_file = output_dir / f"{map_root}_{sim_num}_out0_rot0_0.h5"
	if not root_file.exists():
		return None

	vals, names = load_cosmo_params(
		str(root_file),
		cosmo_params=cosmo_params,
		as_torch=False,
		dtype=np.float64,
	)

	return {name: float(val) for name, val in zip(names, vals)}


def save_results_h5(filename, cat_idx, cls_results, pixelised_results, cosmo_dict):
	filename = Path(filename)
	if filename.suffix == "":
		outname = filename.with_name(f"{filename.stem}_{cat_idx}")
	else:
		outname = filename.with_name(f"{filename.stem}_{cat_idx}{filename.suffix}")

	outdir = outname.parent
	outdir.mkdir(parents=True, exist_ok=True)

	def _save_dict(h5group, dictionary):
		for key, value in dictionary.items():
			if isinstance(value, dict):
				subgroup = h5group.create_group(str(key))
				_save_dict(subgroup, value)
			elif isinstance(value, str):
				dt = h5py.string_dtype(encoding="utf-8")
				h5group.create_dataset(str(key), data=value, dtype=dt)
			else:
				arr = np.asarray(value)

				if arr.dtype == object:
					try:
						arr = arr.astype(np.float64)
					except Exception as exc:
						raise TypeError(
							f"Cannot cast key '{key}' to float64: {exc}\nValue={value}"
						)

				h5group.create_dataset(str(key), data=arr)

	with h5py.File(outname, "w") as f:
		_save_dict(f.create_group("cls_results"), cls_results)
		_save_dict(f.create_group("pixelised_results"), pixelised_results)
		_save_dict(f.create_group("cosmo_dict"), cosmo_dict)

	print(f"Results saved to {outname}")
