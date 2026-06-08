from pathlib import Path

import h5py
import numpy as np

from src.ml.data.data_loading import load_cosmo_params


def resolve_systematics_model(args) -> str:
	"""Resolve the systematics model name from CLI args.

	- `--variable-depth` -> `nla_vd` (variable depth implies the full NLA + shear-bias
	  systematics, matching the reference VD driver's `do_systematics/do_shear_bias=True`).
	- Else if `--systematics-model auto`, uses `nla` iff `--kids-systematics`.
	- Otherwise returns the provided model string.
	"""
	model = getattr(args, "systematics_model", "auto")
	if model == "auto":
		model = "nla" if getattr(args, "kids_systematics", False) else "none"
	if getattr(args, "variable_depth", False):
		return "nla_vd"
	return model


def model_shift_nz(model: str) -> bool:
	"""Whether the tomographic n(z) is dz-shifted for a given systematics model.

	Mirrors the `shift_nz` returned by `build_systematics`; exposed separately so the master can
	compute `tomo_nz` (needed to build the variable-depth objects) BEFORE `build_systematics`.
	"""
	return model in ("nla", "nla_vd")


def build_systematics(model: str, *, systematics_cls, cosmo, ia_params: dict, shear_bias: dict, sigma_e_base: np.ndarray, vd: dict | None = None):
	"""Build systematics object + derived inputs for the simulator.

	`vd` (only for model `nla_vd`) bundles the pre-built variable-depth objects + per-sim realised
	PSF / m-bias-vd inputs.

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
	if model == "nla_vd":
		if vd is None:
			raise ValueError("build_systematics: model 'nla_vd' requires vd=... inputs")
		from .systematics import VariableDepthSystematics
		systematics = VariableDepthSystematics(
			shear_bias=shear_bias,
			nla=ia_params,
			cosmo=cosmo,
			var_depth_mask=vd["var_depth_mask"],
			vd_shapes=vd["vd_shapes"],
			vd_map=vd["vd_map"],
			vd_trace_edges=vd["vd_trace_edges"],
			n_vardepth_bins=vd["n_vardepth_bins"],
			nside=vd["nside"],
			m_bias_vd_realised=vd["m_bias_vd_realised"],
			alpha_1_realised=vd["alpha_1_realised"],
			alpha_2_realised=vd["alpha_2_realised"],
			psf_bias_map_1=vd["psf_bias_map_1"],
			psf_bias_map_2=vd["psf_bias_map_2"],
		)
		return systematics, sigma_e_base.copy(), True
	raise ValueError(f"Unknown systematics model: {model!r}")


def build_variable_depth(data_dir, *, mask, tomo_nz, los_z_integration, zb_tuple, nside, sigma_e, dndz_scale=1.0):
	"""Construct the variable-depth (VD) look-up objects.

	``dndz_scale`` multiplies the per-VD-bin n(z) normalisation so it shares the SAME overall
	galaxy-density scale as the ``tomo_nz`` passed in (the LOS depth fraction is the ratio
	``dndz_vardepth / tomo_nz``, which must be scale-consistent). It is 1.0 in production (then
	`dndz_vd` is built exactly as the reference); the local smoke scales `tomo_nz` down by
	`n_eff_scale`, so it must pass the same factor here.

	Ports the reference VD driver (kids_legacy_sim_vd_looped.py lines ~334-458) VERBATIM, only
	parameterised by the simulator's survey geometry:

	- load per-tomo VD tracer maps (`vd_map`);
	- build the count-contrast functions `n_contrast_vd` (mask-normalised so <contrast>_mask = 1,
	  guaranteeing total galaxy counts match the no-VD case per tomo bin);
	- build the per-pixel sigma_eps model `sigma_eps_var` (clipped cubic, rescaled so
	  <sigma_eps>_mask = sigma_e[i]);
	- build the per-VD-bin n(z) `dndz_vd` from the `comb_1` ascii;
	- assemble `AngularLosVariableDepthMask` + `VariableDepthShapeDispersion`.

	Returns:
		(var_depth_mask, vd_shapes, vd_map)
	"""
	from scipy.interpolate import interp1d

	from .variable_depth import (
		AngularLosVariableDepthMask,
		VariableDepthShapeDispersion,
	)
	from src.KiDS.tomo import nbins, ztomo, ztomo_label, n_arcmin2
	from src.KiDS.variable_depth_config import (
		a_se,
		b_se,
		c_se,
		d_se,
		load_vd_maps,
		n_eff_table,
		n_vardepth_bins,
		vd_trace_eff_centre,
	)

	vd_map = load_vd_maps(data_dir, nside)  # (nbins, npix) at run nside

	# n_eff interpolation -> count-contrast, mask-normalised so <contrast>_mask = 1.
	n_eff_interp = [
		interp1d(
			vd_trace_eff_centre[i], n_eff_table[:, i],
			kind="linear", bounds_error=False,
			fill_value=(n_eff_table[0, i], n_eff_table[-1, i]),
		)
		for i in range(nbins)
	]
	_disc_corr = np.array([
		1.0 / np.average(n_eff_interp[i](vd_map[i]), weights=mask)
		for i in range(nbins)
	])
	n_contrast_vd = [
		lambda x, i=i: _disc_corr[i] * n_eff_interp[i](x)
		for i in range(nbins)
	]

	# sigma_eps VD model: clipped cubic in the tracer, rescaled so <sigma_eps>_mask = sigma_e[i].
	_se_raw = [
		lambda x, i=i: np.clip(
			a_se[i]*x**3 + b_se[i]*x**2 + c_se[i]*x + d_se[i], 0.25, 0.34,
		)
		for i in range(nbins)
	]
	_se_corr = np.array([
		sigma_e[i] / np.average(_se_raw[i](vd_map[i]), weights=mask)
		for i in range(nbins)
	])
	sigma_eps_var = np.array([
		lambda x, i=i: _se_corr[i] * _se_raw[i](x)
		for i in range(nbins)
	])

	# Per-VD-bin n(z): all read the SAME comb_1 ascii, scaled by n_contrast_vd at the VD-bin centre.
	dndz_vd = np.zeros((nbins, n_vardepth_bins, len(los_z_integration)))
	for i in range(nbins):
		z1a, z1b = ztomo_label[i][0].split(".")
		z2a, z2b = ztomo_label[i][1].split(".")
		filename = (
			f"{data_dir}/nofzs/nz_tgweights/BLINDSHAPES_KIDS_Legacy_NS_shear_noSG_noWeiCut_newCut_blindABC_A1_rmcol_filt_PSF_RAD_calc_filt_filt_comb_1_"
			f"ZB{z1a}p{z1b}t{z2a}p{z2b}_calib_goldwt_Nz_recalibrated.ascii"
		)
		hdu = np.loadtxt(filename).T
		z = hdu[0]
		zmid = z[:-1] + 0.5 * (z[1:] - z[:-1])
		for j in range(n_vardepth_bins):
			dndz_interpolated = np.interp(
				los_z_integration,
				zmid,
				dndz_scale * n_arcmin2[i] * n_contrast_vd[i](vd_trace_eff_centre[i][j]) * hdu[1][:-1] / np.trapezoid(hdu[1][:-1], zmid),
			)
			dndz_vd[i][j] = np.clip(dndz_interpolated, 0, None)

	var_depth_mask = AngularLosVariableDepthMask(
		vd_map,
		n_bins=nbins,
		zbins=zb_tuple,
		ztomo=ztomo,
		dndz=tomo_nz,
		z=los_z_integration,
		dndz_vardepth=dndz_vd,
		vardepth_values=vd_trace_eff_centre,
		vardepth_los_tracer=None,
		vardepth_tomo_functions=n_contrast_vd,
	)
	vd_shapes = VariableDepthShapeDispersion(sigma_eps_var, vd_map, nside)

	return var_depth_mask, vd_shapes, vd_map


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
