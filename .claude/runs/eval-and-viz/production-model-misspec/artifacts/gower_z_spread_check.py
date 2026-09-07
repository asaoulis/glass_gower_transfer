import numpy as np

R = "ml-checkpoints/gower_npe_finetune_nla_m_bgp_z8_ens1/misspec"
M = [f"ncosmo300_{i}" for i in range(5)]

print("Is kappa=2's near-ID calibration because there is NO bias, or because biases CANCEL?")
print("If they cancel, the MEAN z stays ~0 while the SPREAD of z widens.")
print("A perfectly calibrated posterior gives mean 0, sd 1.\n")
print(f"{'suite':<18}{'mean z(om)':>12}{'sd z(om)':>11}{'mean z(s8)':>13}{'sd z(s8)':>11}")
print("-" * 65)
for v in ["gower_bgp_nla_m", "gower_gbk2", "gower_vd", "gower_gb1p0", "gower_gb1p3"]:
    mo, so, ms, ss = [], [], [], []
    for m in M:
        with np.load(f"{R}/{v}/misspec_posterior_moments_{m}.npz", allow_pickle=True) as f:
            z = f["z"]
        mo.append(np.nanmean(z[:, 0])); so.append(np.nanstd(z[:, 0]))
        ms.append(np.nanmean(z[:, 1])); ss.append(np.nanstd(z[:, 1]))
    print(f"{v:<18}{np.mean(mo):>+12.3f}{np.mean(so):>11.3f}"
          f"{np.mean(ms):>+13.3f}{np.mean(ss):>11.3f}")
