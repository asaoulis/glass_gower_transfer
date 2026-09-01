import torch
from torch.distributions import MultivariateNormal, Normal, Uniform

try:
    from sbi.utils import MultipleIndependent, RestrictedPrior
except Exception:  # pragma: no cover
    MultipleIndependent = None
    RestrictedPrior = None

from .distributions import (
    NFlowDistribution,
    TruncatedNormal1D,
    PermutedDistribution,
    ScaledDistribution,
    ScaledJointDistribution,
    ScaledMVNDistribution,
    S8OmegaCH2OmegaBH2HPriorToModelParams,
    TorchKDE1D,
    build_gower_paper_known_priors,
)


def _accept_within_unit_hypercube(theta):
    """Module-level accept/reject fn for RestrictedPrior (picklable by reference)."""
    return torch.all((theta >= 0) & (theta <= 1), dim=-1)


if RestrictedPrior is not None:
    class DeviceMovableRestrictedPrior(RestrictedPrior):
        """sbi RestrictedPrior + a real ``to(device)`` method.

        Must be a module-level SUBCLASS, not an instance-bound ``types.MethodType``
        monkey-patch: bound methods stored on the instance pickle as
        ``getattr(obj, func.__name__)`` and explode on unpickle inside joblib/loky
        MCMC workers (AttributeError: no attribute 'to_device')."""

        def to(self, device):
            self._prior.to(device)
            self._device = torch.device(device)
            return self
else:  # pragma: no cover
    DeviceMovableRestrictedPrior = None


mu_AIA = 5.74
mu_beta = 0.44
sigma_AIA = 0.29
sigma_beta = 0.03
rho = -0.59

cov = [
    [sigma_AIA**2, rho * sigma_AIA * sigma_beta],
    [rho * sigma_AIA * sigma_beta, sigma_beta**2],
]
mean = [mu_AIA, mu_beta]


# --- NLA-family IA priors (Wright et al. 2025) ---------------------------------------------
# These complement the NLA-M bivariate Gaussian above. The IA model is disambiguated by which
# *companion* parameter accompanies `a_ia`: b_ia -> NLA-M, b_z -> NLA-z, b_src -> TATT/NLA-k,
# none -> plain NLA.
AIA_NLA_RANGE = (-6.0, 6.0)     # wide top-hat on A_IA for NLA / NLA-z / TATT
BZ_MEAN, BZ_STD = -3.7, 4.3     # NLA-z redshift-slope B_IA Gaussian prior
BSRC_RANGE = (-0.5, 1.5)        # TATT / NLA-k density-weighting bias prior

IA_COMPANION_PARAMS = ("b_ia", "b_z", "b_src")
IA_PARAMS = ("a_ia",) + IA_COMPANION_PARAMS

# Galaxy-bias marginalisation (BGP campaign). The per-tomo-bin b_g values are DRAWN per mock, so
# when they are inferred (`b_g_bin1..6` in cosmo_param_names) their prior is analytic and known:
# the Flamingo KiDS-Legacy O3-diag calibration at kappa=1, truncated at +-3 sigma.
#
# ⚠️ SOURCE OF TRUTH is `src/KiDS/simulation_config.py` (GALAXY_BIAS_PRIOR_MEANS / _SIGMAS), which
# is what the SIMULATOR actually draws from. These are duplicated here rather than imported because
# that module pulls in healpy and the whole survey-geometry stack, which has no business loading in
# a GPU eval job — the same reason the IA constants above are duplicated. If the simulator's preset
# ever changes, THIS MUST BE UPDATED IN LOCKSTEP or every shrinkage number silently references the
# wrong prior. GALAXY_BIAS_CLIP binds only once the prior is WIDENED: at kappa=1 it lies far outside
# +-3 sigma for every bin and never applies, but at kappa=2 it clips the low-z bins (up to ~2.18 % of
# draws), so it must be carried here rather than assumed inert.
GALAXY_BIAS_PRIOR_MEANS = [1.0181, 1.0698, 1.1302, 1.2427, 1.3739, 1.4805]
GALAXY_BIAS_PRIOR_SIGMAS = [0.1801, 0.1491, 0.1252, 0.0951, 0.0960, 0.0985]
GALAXY_BIAS_PRIOR_NSIGMA = 3.0
GALAXY_BIAS_CLIP = (0.3, 2.2)          # src/KiDS/simulation_config.py — keep in lockstep
GALAXY_BIAS_PARAMS = tuple("b_g_bin%d" % i for i in range(1, len(GALAXY_BIAS_PRIOR_MEANS) + 1))


def galaxy_bias_marginal_priors(params, kappa=1.0, clip=GALAXY_BIAS_CLIP):
    """Per-parameter 1D galaxy-bias priors for any `b_g_bin{i}` present in `params`.

    Returns {} when none are present, so this is a no-op for every pre-BGP parameter set.

    Each is a TruncatedNormal1D(mean_i, sigma_i) on [mean_i - 3 sigma_i, mean_i + 3 sigma_i],
    matching the simulator's per-(sim, outer, rot) draw. Supplying these as ANALYTIC priors is what
    keeps them out of `build_gower_prior`'s empirical-flow branch — the Gower Street CSV has no
    b_g columns, so a b_g param reaching the flow is a hard failure, not a degraded prior.
    """
    present = [p for p in params if p in GALAXY_BIAS_PARAMS]
    if not present:
        return {}
    out = {}
    for name in present:
        i = GALAXY_BIAS_PARAMS.index(name)
        loc = GALAXY_BIAS_PRIOR_MEANS[i]
        # kappa RESCALES the width: the simulator draws b_i ~ N(mean_i, kappa*sigma_i) truncated at
        # +-3*kappa*sigma_i and clipped to GALAXY_BIAS_CLIP (src/KiDS/simulation_config.py). Scaling
        # only the BOUNDS and not the SIGMA would still mis-model the prior, so both scale here.
        scale = kappa * GALAXY_BIAS_PRIOR_SIGMAS[i]
        half = GALAXY_BIAS_PRIOR_NSIGMA * scale
        low, high = loc - half, loc + half
        if clip is not None:
            low, high = max(low, clip[0]), min(high, clip[1])
        out[name] = TruncatedNormal1D(loc=loc, scale=scale, low=low, high=high)
    return out


def _infer_galaxy_bias_kappa(preset_overrides):
    """Recover the b_g prior-width multiplier `kappa` from the scaled parameter boxes.

    The boxes an experiment supplies as `scaler_options['cosmo']['preset_overrides']` are exactly
    `(max(mean - 3*kappa*sigma, CLIP_LO), min(mean + 3*kappa*sigma, CLIP_HI))`, so kappa is
    recoverable from any box that the clip did not touch. Clipping only ever SHRINKS a box, so the
    WIDEST bin gives the true kappa; at kappa=2 the clip reaches at most 2.18 % of draws and leaves
    several bins untouched, and at kappa=1 it is inert for every bin.

    Returns 1.0 when no b_g boxes are supplied, which is the pre-existing behaviour.
    """
    if not preset_overrides:
        return 1.0
    ratios = []
    for name, box in preset_overrides.items():
        if name not in GALAXY_BIAS_PARAMS:
            continue
        try:
            lo, hi = float(box[0]), float(box[1])
        except (TypeError, IndexError, ValueError):
            continue
        sigma = GALAXY_BIAS_PRIOR_SIGMAS[GALAXY_BIAS_PARAMS.index(name)]
        ratios.append((hi - lo) / (2.0 * GALAXY_BIAS_PRIOR_NSIGMA * sigma))
    if not ratios:
        return 1.0
    return max(ratios)


def ia_marginal_priors(params):
    """Per-parameter 1D IA priors for the IA params present in `params`.

    The IA model is inferred from the companion parameter so a single `a_ia` name can carry the
    right prior per model:
      - {a_ia, b_ia}  -> NLA-M  (a_ia~U[4.48,7], b_ia~U[0.28,0.6] marginals)
      - {a_ia, b_z}   -> NLA-z  (a_ia~U[-6,6], b_z~N(-3.7,4.3))
      - {a_ia, b_src} -> TATT   (a_ia~U[-6,6], b_src~U[-0.5,1.5])
      - {a_ia}        -> NLA    (a_ia~U[-6,6])
    (For NLA-M the joint Gaussian is used as a block by `build_analytic_prior`; these are the
    independent marginals used by the empirical `build_gower_prior` path.)
    """
    present = [p for p in params if p in IA_PARAMS]
    if not present:
        return {}
    companions = [p for p in present if p in IA_COMPANION_PARAMS]
    if len(companions) > 1:
        raise ValueError(
            f"ia_marginal_priors: at most one IA companion param allowed, got {companions}"
        )
    if "a_ia" not in present:
        raise ValueError(f"ia_marginal_priors: IA companion {companions} present without a_ia")

    out = {}
    if "b_ia" in present:  # NLA-M marginals
        out["a_ia"] = Uniform(torch.tensor([4.48]), torch.tensor([7.0]))
        out["b_ia"] = Uniform(torch.tensor([0.28]), torch.tensor([0.6]))
    else:  # NLA family
        out["a_ia"] = Uniform(
            torch.tensor([AIA_NLA_RANGE[0]]), torch.tensor([AIA_NLA_RANGE[1]])
        )
        if "b_z" in present:
            out["b_z"] = Normal(torch.tensor([BZ_MEAN]), torch.tensor([BZ_STD]))
        if "b_src" in present:
            out["b_src"] = Uniform(
                torch.tensor([BSRC_RANGE[0]]), torch.tensor([BSRC_RANGE[1]])
            )
    return out


def build_scaled_joint_gaussian(names, mean, cov, scaler):
    name_to_idx = {n: i for i, n in enumerate(scaler.parameter_names)}

    mins = []
    maxs = []

    for name in names:
        idx = name_to_idx[name]
        mins.append(scaler.min[idx])
        maxs.append(scaler.max[idx])

    base = MultivariateNormal(
        torch.tensor(mean, dtype=torch.float32),
        torch.tensor(cov, dtype=torch.float32),
    )

    return ScaledMVNDistribution(base, mins, maxs)


def build_s8_box_known_priors(
    *,
    pivot_omega_m: float = 0.3,
    cosmo_parameter_order=("sigma_8", "omega_m", "ombh2", "h"),
):
    """Analytic priors in physical units with flat boxes in (S8, ωc, ωb) and h."""

    cosmo_parameter_order = tuple(cosmo_parameter_order)
    expected = {"sigma_8", "omega_m", "ombh2", "h"}
    if set(cosmo_parameter_order) != expected or len(cosmo_parameter_order) != 4:
        raise ValueError(
            "build_s8_box_known_priors: cosmo_parameter_order must be a permutation "
            f"of {sorted(list(expected))}; got {cosmo_parameter_order!r}"
        )

    ia_base = MultivariateNormal(
        torch.tensor(mean, dtype=torch.float32),
        torch.tensor(cov, dtype=torch.float32),
    )

    cosmo_joint = S8OmegaCH2OmegaBH2HPriorToModelParams(
        s8_low=0.5,
        s8_high=1.0,
        oc_low=0.051,
        oc_high=0.18,
        ob_low=0.022,
        ob_high=0.0228,
        h_low=0.64,
        h_high=0.78,
        pivot_omega_m=float(pivot_omega_m),
        parameter_order=cosmo_parameter_order,
    )

    return {
        cosmo_parameter_order: cosmo_joint,
        "ns": Uniform(torch.tensor([0.948]), torch.tensor([0.984])),
        "w0": Uniform(torch.tensor([-1.0]), torch.tensor([-1.0 / 3.0])),
        "mnu": Uniform(torch.tensor([0.06]), torch.tensor([0.14])),
        ("a_ia", "b_ia"): ia_base,
    }


def _combine_independent(dists):
    """Join independent priors into one distribution.

    `sbi`'s MultipleIndependent asserts len(dists) > 1, so a SINGLE distribution must be returned
    as-is rather than wrapped. That case arises whenever the leading distribution already spans
    every inferred parameter and no extra analytic priors are appended -- e.g. the 2-param
    {omega_m, sigma_8} runs, where the Gower flow covers both. With 2+ distributions the behaviour
    is unchanged.
    """
    if len(dists) == 1:
        return dists[0]
    return MultipleIndependent(dists)


def build_kde_prior_from_df(
    df,
    columns,
    scaler,
    extra_priors=None,
):
    if MultipleIndependent is None:
        raise ModuleNotFoundError("build_kde_prior_from_df requires 'sbi' (MultipleIndependent).")

    dists = []
    name_to_idx = {n: i for i, n in enumerate(scaler.parameter_names)}

    for column in columns:
        idx = name_to_idx[column]
        dists.append(TorchKDE1D(df[column].values, scaler.min[idx], scaler.max[idx]))

    if extra_priors is not None:
        for name, dist in extra_priors.items():
            idx = name_to_idx[name]
            dists.append(ScaledDistribution(dist, scaler.min[idx], scaler.max[idx]))

    return _combine_independent(dists)


def build_gower_st_prior(
    variables,
    scaler,
    csv_path,
    drop_first=192,
    n_samples=5000,
    extra_priors=None,
):
    from src.cosmology.gower_street import GowerStPrior

    gower_prior_builder = GowerStPrior.from_csv(
        csv_path,
        drop_first=drop_first,
    )

    res = gower_prior_builder.sample(n_samples)
    return build_kde_prior_from_df(
        res,
        columns=variables,
        scaler=scaler,
        extra_priors=extra_priors,
    )


def build_flow_with_extras_prior(
    flow,
    columns,
    scaler,
    extra_priors=None,
    return_restricted=False,
):
    if MultipleIndependent is None:
        raise ModuleNotFoundError("build_flow_with_extras_prior requires 'sbi' (MultipleIndependent).")
    if return_restricted and RestrictedPrior is None:
        raise ModuleNotFoundError("build_flow_with_extras_prior(return_restricted=True) requires 'sbi' (RestrictedPrior).")

    name_to_idx = {n: i for i, n in enumerate(scaler.parameter_names)}
    flow_dist = NFlowDistribution(flow=flow, dims=len(columns))
    dists = [flow_dist]

    if extra_priors is not None:
        for name, dist in extra_priors.items():
            idx = name_to_idx[name]
            dists.append(ScaledDistribution(dist, min_val=scaler.min[idx], max_val=scaler.max[idx]))

    joint_prior = _combine_independent(dists)

    if not return_restricted:
        return joint_prior

    return DeviceMovableRestrictedPrior(
        prior=joint_prior,
        accept_reject_fn=_accept_within_unit_hypercube,
        sample_with="rejection",
    )


def build_analytic_prior(
    params,
    scaler,
    *,
    pivot_omega_m: float = 0.3,
    return_restricted: bool = False,
):
    """Build an analytic prior in scaled space ([0,1]^D)."""

    if MultipleIndependent is None:
        raise ModuleNotFoundError("build_analytic_prior requires 'sbi' (MultipleIndependent).")

    params = list(params)
    name_to_idx = {n: i for i, n in enumerate(scaler.parameter_names)}

    cosmo_set = {"omega_m", "sigma_8", "ombh2", "h"}
    cosmo_block = [p for p in params if p in cosmo_set]
    if set(cosmo_block) != cosmo_set:
        missing = sorted(list(cosmo_set - set(cosmo_block)))
        raise ValueError(f"build_analytic_prior: missing required cosmo params {missing} in params={params}")

    phys_priors = build_s8_box_known_priors(
        pivot_omega_m=float(pivot_omega_m),
        cosmo_parameter_order=tuple(cosmo_block),
    )

    phys_cosmo = phys_priors[tuple(cosmo_block)]
    cosmo_mins = [float(scaler.min[name_to_idx[p]]) for p in cosmo_block]
    cosmo_maxs = [float(scaler.max[name_to_idx[p]]) for p in cosmo_block]
    scaled_cosmo = ScaledJointDistribution(phys_cosmo, cosmo_mins, cosmo_maxs)

    one_d_priors = {
        "ns": phys_priors["ns"],
        "w0": phys_priors["w0"],
        "mnu": phys_priors["mnu"],
    }

    ia_names = [p for p in params if p in IA_PARAMS]
    ia_companions = [p for p in ia_names if p in IA_COMPANION_PARAMS]
    if len(ia_companions) > 1:
        raise ValueError(
            "build_analytic_prior: at most one IA companion param (b_ia/b_z/b_src) allowed. "
            f"Got {ia_companions!r} in params={params!r}"
        )
    if ia_names and "a_ia" not in ia_names:
        raise ValueError(
            f"build_analytic_prior: IA companion {ia_companions!r} present without 'a_ia' "
            f"in params={params!r}"
        )
    # NLA-M uses the (a_ia, b_ia) joint Gaussian as a block; NLA / NLA-z / TATT use independent
    # 1D IA marginals (a_ia ~ U[-6,6] plus b_z ~ N / b_src ~ U).
    nla_m_block = (
        tuple(p for p in params if p in {"a_ia", "b_ia"}) if "b_ia" in ia_names else None
    )

    dists = [scaled_cosmo]
    internal_order = list(cosmo_block)
    used = set(cosmo_block)
    used.update(ia_names)

    for p in params:
        if p in used:
            continue
        if p not in one_d_priors:
            raise ValueError(
                f"build_analytic_prior: no analytic prior specified for parameter {p!r}. "
                f"Handled: {sorted(list(cosmo_set | set(one_d_priors) | set(IA_PARAMS)))}"
            )
        idx = name_to_idx[p]
        dists.append(ScaledDistribution(one_d_priors[p], scaler.min[idx], scaler.max[idx]))
        internal_order.append(p)

    if nla_m_block is not None:
        ia_base_ab = phys_priors[("a_ia", "b_ia")]
        if nla_m_block == ("a_ia", "b_ia"):
            ia_base = ia_base_ab
        elif nla_m_block == ("b_ia", "a_ia"):
            perm = torch.tensor([1, 0], dtype=torch.long)
            mean_perm = ia_base_ab.mean[perm]
            cov_perm = ia_base_ab.covariance_matrix[perm][:, perm]
            ia_base = MultivariateNormal(mean_perm, cov_perm)
        else:
            raise ValueError(f"build_analytic_prior: unexpected IA ordering {nla_m_block!r}")

        ia_mins = [float(scaler.min[name_to_idx[p]]) for p in nla_m_block]
        ia_maxs = [float(scaler.max[name_to_idx[p]]) for p in nla_m_block]
        dists.append(ScaledMVNDistribution(ia_base, ia_mins, ia_maxs))
        internal_order.extend(list(nla_m_block))
    elif ia_names:
        # NLA family: independent 1D IA priors (a_ia plus optional b_z / b_src), in params order.
        ia_1d = ia_marginal_priors(params)
        for p in params:
            if p in ia_1d:
                idx = name_to_idx[p]
                dists.append(ScaledDistribution(ia_1d[p], scaler.min[idx], scaler.max[idx]))
                internal_order.append(p)

    joint_prior = _combine_independent(dists)
    base_to_wrap = joint_prior

    if return_restricted:
        if RestrictedPrior is None:
            raise ModuleNotFoundError(
                "build_analytic_prior(return_restricted=True) requires 'sbi' (RestrictedPrior)."
            )

        base_to_wrap = DeviceMovableRestrictedPrior(
            prior=joint_prior,
            accept_reject_fn=_accept_within_unit_hypercube,
            sample_with="rejection",
        )

    return PermutedDistribution(
        base_to_wrap,
        base_order=internal_order,
        target_order=params,
        enforce_unit_hypercube=True,
    )
