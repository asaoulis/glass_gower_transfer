
from src.cosmology.simulators import generate_rotation_specs
base_rotation_specs = generate_rotation_specs(
    delta_deg=0.0,
    recenter_deg=0.0,
    flips=(False,),
    base_angles=(0,),
    backend="identity"
)
rotation_specs_15 = generate_rotation_specs(
    delta_deg=0.0,
    recenter_deg=15.0,
    flips=(False, True),
    backend="alm",
    base_angles=(0,),
)

rotation_specs_25 = generate_rotation_specs(
    delta_deg=0.0,
    recenter_deg=25.0,
    flips=(False, True),
    backend="alm",
    base_angles=(0,),
)
KiDS_PATCH_GOWER_ROTATIONS = base_rotation_specs + rotation_specs_15 + rotation_specs_25

KiDS_PATCH_GLASS_ROTATIONS = base_rotation_specs