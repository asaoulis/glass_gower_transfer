# PRESET KiDS Gower Street mins and maxes for cosmology parameterrs
COSMO_PARAM_PRESET_MINMAX = {
    'omega_m': (0.1, 0.5),
    'sigma_8': (0.6, 1.0),
    'h0': (0.6, 0.8),
    'n_s': (0.9, 1.0),
    'omega_b': (0.04, 0.06)
}

COSMO_PARAM_PRESET_MINMAX = {
     'omega_m': (0.1232301212, 0.4967720335),
      'sigma_8': (0.4197701053, 1.28618999), 
      'w0': (-1.00973808, -0.3391160133), 
      'ombh2': (0.02190540964, 0.02280284524), 
      'h': (0.6087511211, 0.7799672924), 
      'ns': (0.9467205001, 0.9844127952),
       'mnu': (0.06, 0.1398179454), 
       'Omega_b': (0.03670663082, 0.06051677202),
       'a_ia': (4.48, 7.0),
       'b_ia': (0.28, 0.6),
       # NLA-family IA params (Wright et al. 2025). NOTE: a_ia's box above is the NLA-M range;
       # NLA / NLA-z / TATT datasets use a_ia ~ U[-6, 6] and so need an a_ia scaler box of
       # (-6.0, 6.0) — override per dataset/config (the global preset cannot hold two a_ia boxes).
       'b_z': (-25.2, 17.8),    # ~5 sigma around N(-3.7, 4.3) for NLA-z
       'b_src': (-0.5, 1.5),    # TATT / NLA-k density-weighting bias prior range
}