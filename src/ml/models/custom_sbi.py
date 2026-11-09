# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from functools import partial
from typing import List, Literal, Optional, Sequence, Tuple, Union

import torch
import zuko
from pyknos.nflows import distributions as distributions_
from pyknos.nflows import flows, transforms
from pyknos.nflows.nn import nets
from pyknos.nflows.transforms.splines import (
    rational_quadratic,  # pyright: ignore[reportAttributeAccessIssue]
)
from torch import Tensor, nn, relu, tanh, tensor, uint8
from torch.distributions import Distribution
from zuko.lazy import Flow, LazyDistribution

from sbi.neural_nets.estimators import NFlowsFlow, ZukoFlow, ZukoUnconditionalFlow
from sbi.utils.nn_utils import MADEMoGWrapper, get_numel
from sbi.utils.sbiutils import (
    biject_transform_zuko,
    mcmc_transform,
    standardizing_net,
    standardizing_transform,
    standardizing_transform_zuko,
    z_score_parser,
)
from sbi.utils.torchutils import create_alternating_binary_mask
from sbi.utils.user_input_checks import check_data_device

nflow_specific_kwargs = ["num_bins", "num_components", "tail_bound"]

import os
import pickle
import torch
import numpy as np

from nflows.flows.autoregressive import MaskedAutoregressiveFlow
import torch
from torch import nn
import torch.nn.functional as F
from pyknos.nflows import distributions as distributions_
from pyknos.nflows import flows, transforms
from pyknos.nflows.nn import nets
from pyknos.nflows.transforms.splines import (
    rational_quadratic,  # pyright: ignore[reportAttributeAccessIssue]
)
from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.coupling import PiecewiseRationalQuadraticCouplingTransform
from nflows.utils import create_alternating_binary_mask
from nflows.nn import nets

class NeuralSplineFlow(Flow):
    def __init__(
        self,
        features: int,
        hidden_features: int = 64,
        num_layers: int = 5,
        num_blocks_per_layer: int = 2,
        num_bins: int = 10,
        tail_bound: float = 3.0,
        dropout_probability: float = 0.0,
        use_batch_norm: bool = False,
    ):
        """Builds an Unconditional Neural Spline Flow."""
        super().__init__(transform=None, distribution=None) # Initialize later below

        distribution = StandardNormal(shape=[features])

        transform_list = []
        for i in range(num_layers):
            mask = create_alternating_binary_mask(
                features=features, even=(i % 2 == 0)
            )

            # context_features=None makes the conditioner unconditional.
            def create_resnet(in_features, out_features):
                return nets.ResidualNet(
                    in_features=in_features,
                    out_features=out_features,
                    hidden_features=hidden_features,
                    context_features=None, 
                    num_blocks=num_blocks_per_layer,
                    activation=F.relu,
                    dropout_probability=dropout_probability,
                    use_batch_norm=use_batch_norm,
                )

            coupling_transform = transforms.PiecewiseRationalQuadraticCouplingTransform(
                mask=mask,
                transform_net_create_fn=create_resnet,
                num_bins=num_bins,
                tails="linear",
                tail_bound=tail_bound,
                apply_unconditional_transform=False,
            )
            transform_list.append(coupling_transform)

            # LU linear mixing, only for more than one dimension.
            if features > 1:
                transform_list.append(
                    transforms.LULinear(features, identity_init=True)
                )

        transform = CompositeTransform(transform_list)

        super().__init__(transform=transform, distribution=distribution)

class _DeviceSafeMADEMoGWrapper(MADEMoGWrapper):
    """sbi's MADEMoGWrapper with its dummy input created on the inputs' device and dtype.

    The sbi version creates it on the CPU and crashes on CUDA."""

    def _log_prob(self, inputs, context=None):
        import numpy as np
        import torch.nn.functional as F
        dummy_input = torch.zeros(
            inputs.shape[:-1] + (1,), device=inputs.device, dtype=inputs.dtype
        )
        concat_inputs = torch.cat((dummy_input, inputs), dim=-1)

        outputs = self._made.forward(concat_inputs, context=context)
        outputs = outputs.reshape(
            *concat_inputs.shape, self._made.num_mixture_components, 3
        )

        logits, means, unconstrained_stds = (
            outputs[..., 0],
            outputs[..., 1],
            outputs[..., 2],
        )
        logits = logits[..., 1:, :]
        means = means[..., 1:, :]
        unconstrained_stds = unconstrained_stds[..., 1:, :]

        log_mixture_coefficients = torch.log_softmax(logits, dim=-1)
        stds = F.softplus(unconstrained_stds) + self._made.epsilon

        log_prob = torch.sum(
            torch.logsumexp(
                log_mixture_coefficients
                - 0.5
                * (
                    np.log(2 * np.pi)
                    + 2 * torch.log(stds)
                    + ((inputs[..., None] - means) / stds) ** 2
                ),
                dim=-1,
            ),
            dim=-1,
        )
        return log_prob



def build_made(
    batch_x: Tensor,
    batch_y: Tensor,
    z_score_x: Literal[
        "none", "independent", "structured", "transform_to_unconstrained"
    ] = "independent",
    z_score_y: Literal[
        "none", "independent", "structured", "transform_to_unconstrained"
    ] = "independent",
    hidden_features: int = 50,
    num_mixture_components: int = 10,
    embedding_net: nn.Module = nn.Identity(),
    **kwargs,
) -> NFlowsFlow:
    """Builds a MADE mixture-of-Gaussians density estimator p(x|y) (flow_type 'mdn').

    batch_x and batch_y set the dimensions and the optional z-scoring, which z_score_x and
    z_score_y choose ('none', 'independent', 'structured' or 'transform_to_unconstrained').
    embedding_net embeds y. Other kwargs are ignored. Returns an sbi NFlowsFlow.
    """
    check_data_device(batch_x, batch_y)
    x_numel = get_numel(batch_x, embedding_net=None)
    y_numel = get_numel(batch_y, embedding_net=embedding_net)

    transform = transforms.IdentityTransform()

    z_score_x_bool, structured_x = z_score_parser(z_score_x)
    if z_score_x_bool:
        transform_zx = standardizing_transform(batch_x, structured_x)
        transform = transforms.CompositeTransform([transform_zx, transform])

    embedding_net = _prepare_y_embedding(z_score_y, batch_y, embedding_net)

    distribution = _DeviceSafeMADEMoGWrapper(
        features=x_numel,
        hidden_features=hidden_features,
        context_features=y_numel,
        num_blocks=5,
        num_mixture_components=num_mixture_components,
        use_residual_blocks=True,
        random_mask=False,
        activation=relu,
        dropout_probability=0.0,
        use_batch_norm=False,
        custom_initialization=True,
    )

    neural_net = flows.Flow(transform, distribution, embedding_net)
    flow = NFlowsFlow(
        neural_net, input_shape=batch_x[0].shape, condition_shape=batch_y[0].shape
    )

    return flow


def build_maf_rqs(
    batch_x: Tensor,
    batch_y: Tensor,
    z_score_x: Literal[
        "none", "independent", "structured", "transform_to_unconstrained"
    ] = "independent",
    z_score_y: Literal[
        "none", "independent", "structured", "transform_to_unconstrained"
    ] = "independent",
    hidden_features: int = 50,
    num_transforms: int = 5,
    embedding_net: nn.Module = nn.Identity(),
    num_blocks: int = 2,
    num_bins: int = 10,
    tails: Optional[str] = "linear",
    tail_bound: float = 3.0,
    dropout_probability: float = 0.0,
    use_batch_norm: bool = False,
    min_bin_width: float = rational_quadratic.DEFAULT_MIN_BIN_WIDTH,
    min_bin_height: float = rational_quadratic.DEFAULT_MIN_BIN_HEIGHT,
    min_derivative: float = rational_quadratic.DEFAULT_MIN_DERIVATIVE,
    **kwargs,
) -> NFlowsFlow:
    """Builds MAF p(x|y) with rational-quadratic spline (RQS) transforms (flow_type 'rqs').

    batch_x and batch_y set the dimensions and the optional z-scoring (z_score_x, z_score_y).
    tails=None gives a constrained RQS. 'linear' applies it on [-tail_bound, tail_bound] with the
    identity outside. embedding_net embeds y. Other kwargs are ignored. Returns an sbi NFlowsFlow.
    """
    check_data_device(batch_x, batch_y)
    x_numel = get_numel(
        batch_x,
        embedding_net=None,
        warn_on_1d=True,  # warn if output space is 1D.
    )
    y_numel = get_numel(batch_y, embedding_net=embedding_net)

    transform_list = []
    for _ in range(num_transforms):
        block = [
            transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=x_numel,
                hidden_features=hidden_features,
                context_features=y_numel,
                num_bins=num_bins,
                tails=tails,
                tail_bound=tail_bound,
                num_blocks=num_blocks,
                use_residual_blocks=False,
                random_mask=False,
                activation=tanh,
                dropout_probability=dropout_probability,
                use_batch_norm=use_batch_norm,
                min_bin_width=min_bin_width,
                min_bin_height=min_bin_height,
                min_derivative=min_derivative,
            ),
            transforms.RandomPermutation(features=x_numel),
        ]
        transform_list += block

    z_score_x_bool, structured_x = z_score_parser(z_score_x)
    if z_score_x_bool:
        transform_list = [
            standardizing_transform(batch_x, structured_x)
        ] + transform_list

    embedding_net = _prepare_y_embedding(z_score_y, batch_y, embedding_net)

    # Combine transforms.
    transform = transforms.CompositeTransform(transform_list)

    distribution = get_base_dist(x_numel, **kwargs)
    neural_net = flows.Flow(transform, distribution, embedding_net)
    flow = NFlowsFlow(
        neural_net, input_shape=batch_x[0].shape, condition_shape=batch_y[0].shape
    )

    return flow

from src.ml.models.patched_rqs import DtypePatchedPiecewiseRationalQuadraticCouplingTransform

def build_nsf(
    batch_x: Tensor,
    batch_y: Tensor,
    z_score_x: Literal[
        "none", "independent", "structured", "transform_to_unconstrained"
    ] = "independent",
    z_score_y: Literal[
        "none", "independent", "structured", "transform_to_unconstrained"
    ] = "independent",
    hidden_features: int = 50,
    num_transforms: int = 5,
    num_bins: int = 10,
    embedding_net: nn.Module = nn.Identity(),
    tail_bound: float = 3.0,
    hidden_layers_spline_context: int = 1,
    num_blocks: int = 2,
    dropout_probability: float = 0.0,
    use_batch_norm: bool = False,
    **kwargs,
) -> NFlowsFlow:
    """Builds NSF p(x|y) from dtype-patched RQS coupling layers and LU mixing (flow_type 'nsf').

    batch_x and batch_y set the dimensions and the optional z-scoring (z_score_x, z_score_y).
    A 1D x uses ContextSplineMap (hidden_layers_spline_context) as the conditioner, otherwise a
    residual net of num_blocks. embedding_net embeds y. Other kwargs are ignored. Returns an sbi
    NFlowsFlow.
    """
    check_data_device(batch_x, batch_y)
    x_numel = get_numel(batch_x, embedding_net=None)
    y_numel = get_numel(batch_y, embedding_net=embedding_net)

    # Define mask function to alternate between predicted x-dimensions.
    def mask_in_layer(i):
        return create_alternating_binary_mask(features=x_numel, even=(i % 2 == 0))

    # If x is just a scalar then use a dummy mask and learn spline parameters using the
    # conditioning variables only.
    if x_numel == 1:
        # Conditioner ignores the data and uses the conditioning variables only.
        conditioner = partial(
            ContextSplineMap,
            hidden_features=hidden_features,
            context_features=y_numel,
            hidden_layers=hidden_layers_spline_context,
        )
    else:
        # Use conditional resnet as spline conditioner.
        conditioner = partial(
            nets.ResidualNet,
            hidden_features=hidden_features,
            context_features=y_numel,
            num_blocks=num_blocks,
            activation=relu,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
        )

    # Stack spline transforms.
    transform_list = []
    for i in range(num_transforms):
        block: List[transforms.Transform] = [
            DtypePatchedPiecewiseRationalQuadraticCouplingTransform(
                mask=mask_in_layer(i) if x_numel > 1 else tensor([1], dtype=uint8),
                transform_net_create_fn=conditioner,
                num_bins=num_bins,
                tails="linear",
                tail_bound=tail_bound,
                apply_unconditional_transform=False,
            )
        ]
        # Add LU transform only for high D x. Permutation makes sense only for more than
        # one feature.
        if x_numel > 1:
            block.append(
                transforms.LULinear(x_numel, identity_init=True),
            )
        transform_list += block

    z_score_x_bool, structured_x = z_score_parser(z_score_x)
    if z_score_x_bool:
        # Prepend standardizing transform to nsf transforms.
        transform_list = [
            standardizing_transform(batch_x, structured_x)
        ] + transform_list

    embedding_net = _prepare_y_embedding(z_score_y, batch_y, embedding_net)

    distribution = get_base_dist(x_numel, **kwargs)

    # Combine transforms.
    transform = transforms.CompositeTransform(transform_list)
    neural_net = flows.Flow(transform, distribution, embedding_net)
    flow = NFlowsFlow(
        neural_net, input_shape=batch_x[0].shape, condition_shape=batch_y[0].shape
    )

    return flow


def _prepare_y_embedding(
    z_score_y: Literal[
        "none", "independent", "structured", "transform_to_unconstrained"
    ],
    batch_y: Tensor,
    embedding_net: nn.Module,
) -> nn.Module:
    """
    Prepend the embedding network for y, adding z-scoring if needed.

    Args:
        z_score_y: Type of y preprocessing.
        batch_y: Batch of y data.
        embedding_net: Original embedding network.

    Returns:
        Modified embedding network.
    """
    z_score_y_bool, structured_y = z_score_parser(z_score_y)
    if z_score_y_bool:
        return nn.Sequential(standardizing_net(batch_y, structured_y), embedding_net)
    return embedding_net


class ContextSplineMap(nn.Module):
    """Neural network from `context` to the spline parameters, for a 1D x with no other dimension."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int,
        context_features: int,
        hidden_layers: int,
    ):
        """
        Initialize neural network that learns to predict spline parameters.

        Args:
            in_features: Unused since there is no `conditioner` in 1D.
            out_features: Number of spline parameters.
            hidden_features: Number of hidden units.
            context_features: Number of context features.
        """
        super().__init__()
        # `self.hidden_features` is only defined such that nflows can infer
        # a scaling factor for initializations.
        self.hidden_features = hidden_features

        # Use a non-linearity because otherwise, there will be a linear
        # mapping from context features onto distribution parameters.

        # Initialize with input layer.
        layer_list = [nn.Linear(context_features, hidden_features), nn.ReLU()]
        # Add hidden layers.
        layer_list += [
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
        ] * hidden_layers
        # Add output layer.
        layer_list += [nn.Linear(hidden_features, out_features)]
        self.spline_predictor = nn.Sequential(*layer_list)

    def __call__(self, inputs: Tensor, context: Tensor, *args, **kwargs) -> Tensor:
        """
        Return parameters of the spline given the context.

        Args:
            inputs: Unused. It would usually be the other dimensions, but in
                1D, there are no other dimensions.
            context: Context features.

        Returns:
            Spline parameters.
        """
        return self.spline_predictor(context)


def get_base_dist(
    num_dims: int, dtype: torch.dtype = torch.float32, **kwargs
) -> distributions_.Distribution:
    """Returns the base distribution for a flow with given float dtype."""

    base = distributions_.StandardNormal((num_dims,))
    base._log_z = base._log_z.to(dtype)
    return base