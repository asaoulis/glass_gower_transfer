import torch
from torch.distributions import MultivariateNormal, Uniform

try:
    from sbi.utils import MultipleIndependent, RestrictedPrior
except Exception:  # pragma: no cover
    MultipleIndependent = None
    RestrictedPrior = None

from .distributions import (
    NFlowDistribution,
    PermutedDistribution,
    ScaledDistribution,
    ScaledJointDistribution,
    ScaledMVNDistribution,
    S8OmegaCH2OmegaBH2HPriorToModelParams,
    TorchKDE1D,
    build_gower_paper_known_priors,
)


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
        s8_high=0.9,
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

    return MultipleIndependent(dists)


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

    joint_prior = MultipleIndependent(dists)

    if not return_restricted:
        return joint_prior

    def is_within_unit_hypercube(theta):
        return torch.all((theta >= 0) & (theta <= 1), dim=-1)

    restricted_prior = RestrictedPrior(
        prior=joint_prior,
        accept_reject_fn=is_within_unit_hypercube,
        sample_with="rejection",
    )

    import types

    def to_device(self, device):
        self._prior.to(device)
        self._device = torch.device(device)
        return self

    restricted_prior.to = types.MethodType(to_device, restricted_prior)
    return restricted_prior


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

    ia_names = [p for p in params if p in {"a_ia", "b_ia"}]
    if len(ia_names) == 1:
        raise ValueError(
            "build_analytic_prior: if using IA params, both 'a_ia' and 'b_ia' must be present. "
            f"Got ia_names={ia_names!r} in params={params!r}"
        )
    ia_block = tuple(ia_names) if len(ia_names) == 2 else None

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
                f"Handled: {sorted(list(cosmo_set | set(one_d_priors) | {'a_ia', 'b_ia'}))}"
            )
        idx = name_to_idx[p]
        dists.append(ScaledDistribution(one_d_priors[p], scaler.min[idx], scaler.max[idx]))
        internal_order.append(p)

    if ia_block is not None:
        ia_base_ab = phys_priors[("a_ia", "b_ia")]
        if ia_block == ("a_ia", "b_ia"):
            ia_base = ia_base_ab
        elif ia_block == ("b_ia", "a_ia"):
            perm = torch.tensor([1, 0], dtype=torch.long)
            mean_perm = ia_base_ab.mean[perm]
            cov_perm = ia_base_ab.covariance_matrix[perm][:, perm]
            ia_base = MultivariateNormal(mean_perm, cov_perm)
        else:
            raise ValueError(f"build_analytic_prior: unexpected IA ordering {ia_block!r}")

        ia_mins = [float(scaler.min[name_to_idx[p]]) for p in ia_block]
        ia_maxs = [float(scaler.max[name_to_idx[p]]) for p in ia_block]
        dists.append(ScaledMVNDistribution(ia_base, ia_mins, ia_maxs))
        internal_order.extend(list(ia_block))

    joint_prior = MultipleIndependent(dists)
    base_to_wrap = joint_prior

    if return_restricted:
        if RestrictedPrior is None:
            raise ModuleNotFoundError(
                "build_analytic_prior(return_restricted=True) requires 'sbi' (RestrictedPrior)."
            )

        def is_within_unit_hypercube(theta):
            return torch.all((theta >= 0) & (theta <= 1), dim=-1)

        restricted_prior = RestrictedPrior(
            prior=joint_prior,
            accept_reject_fn=is_within_unit_hypercube,
            sample_with="rejection",
        )

        import types

        def to_device(self, device):
            self._prior.to(device)
            self._device = torch.device(device)
            return self

        restricted_prior.to = types.MethodType(to_device, restricted_prior)
        base_to_wrap = restricted_prior

    return PermutedDistribution(
        base_to_wrap,
        base_order=internal_order,
        target_order=params,
        enforce_unit_hypercube=True,
    )
