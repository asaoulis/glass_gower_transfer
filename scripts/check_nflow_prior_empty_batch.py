"""Pin the empty-batch guard on NFlowDistribution.log_prob (distributions.py:512).

Covers the three cases: all rows valid, a mix, and NONE valid (the crash that killed
VD `_hf` r0). Also checks the finite values are unchanged by the guard.
"""
import sys
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms import (
    CompositeTransform,
    PiecewiseRationalQuadraticCouplingTransform,
)
from nflows.nn.nets import ResidualNet
from nflows.utils import create_alternating_binary_mask

from src.ml.data.priors.distributions import NFlowDistribution

D = 2


def make_flow():
    torch.manual_seed(0)
    t = CompositeTransform([
        PiecewiseRationalQuadraticCouplingTransform(
            mask=create_alternating_binary_mask(D, even=True),
            transform_net_create_fn=lambda i, o: ResidualNet(
                i, o, hidden_features=8, num_blocks=1
            ),
            num_bins=4,
            tails="linear",
            tail_bound=3.0,
        )
    ])
    return Flow(t, StandardNormal([D]))


def main():
    dist = NFlowDistribution(make_flow(), dims=D)

    all_valid = torch.rand(5, D)
    mixed = torch.cat([torch.rand(3, D), torch.full((2, D), 5.0)], dim=0)
    none_valid = torch.full((7, D), 5.0)      # every row outside [0,1]^2
    empty = torch.zeros(0, D)

    lp_all = dist.log_prob(all_valid)
    assert torch.isfinite(lp_all).all(), "valid rows must be finite"
    print(f"all-valid   : shape={tuple(lp_all.shape)} finite={int(torch.isfinite(lp_all).sum())}/5")

    lp_mixed = dist.log_prob(mixed)
    assert torch.isfinite(lp_mixed[:3]).all() and torch.isinf(lp_mixed[3:]).all()
    print(f"mixed       : shape={tuple(lp_mixed.shape)} finite={int(torch.isfinite(lp_mixed).sum())}/5")

    # the regression: this used to raise RuntimeError from nflows
    lp_none = dist.log_prob(none_valid)
    assert lp_none.shape == (7,) and torch.isinf(lp_none).all() and (lp_none < 0).all()
    print(f"NONE-valid  : shape={tuple(lp_none.shape)} all -inf -> OK (was RuntimeError)")

    lp_empty = dist.log_prob(empty)
    assert lp_empty.shape == (0,)
    print(f"empty input : shape={tuple(lp_empty.shape)} -> OK")

    # the guard must not perturb finite values: same input, same answer
    again = dist.log_prob(all_valid)
    assert torch.allclose(lp_all, again)
    assert torch.allclose(dist.log_prob(mixed[:3]), lp_mixed[:3]), "mixed path unchanged"
    print("values unchanged on every non-empty path -> OK")
    print("\nALL CHECKS PASS")


if __name__ == "__main__":
    main()
