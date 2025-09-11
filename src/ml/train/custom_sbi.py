from sbi.inference.snpe import PosteriorEstimator
# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from typing import Any, Callable, Dict, Optional, Union
from warnings import warn

import torch
from torch import Tensor, nn
from sbi.samplers.rejection import rejection


from sbi import utils as utils

from sbi.utils import (

    within_support
)

from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch



def reshape_to_batch_event(theta_or_x, event_shape):
    """Return theta or x s.t. its shape is `(batch_dim, *event_shape)`.

    Args:
        theta_or_x: The tensor to be reshaped. Can have any of the following shapes:
            - (event)
            - (batch, event)
        event_shape: The shape of a single datapoint (without batch dimension or sample
            dimension).

    Returns:
        A tensor of shape `(batch, event)`.
    """
    # `2` for image data, `3` for video data, ...
    event_shape_dim = len(event_shape)

    trailing_theta_or_x_shape = theta_or_x.shape[-event_shape_dim:]
    leading_theta_or_x_shape = theta_or_x.shape[:-event_shape_dim]
    assert trailing_theta_or_x_shape == event_shape, (
        "The trailing dimensions of `theta_or_x` do not match the `event_shape`."
    )

    if len(leading_theta_or_x_shape) == 0:
        # A single datapoint is passed. Add batch artificially.
        return theta_or_x.unsqueeze(0)
    elif len(leading_theta_or_x_shape) == 1:
        # A batch dimension was passed.
        return theta_or_x
    else:
        raise ValueError(
            f"`len(leading_theta_or_x_shape) = {leading_theta_or_x_shape} > 1`. "
            f"It is unclear how the additional entries should be interpreted"
        )
def sample_batched(
    self,
    sample_shape,
    x: Tensor,
    condition_shape,
    max_sampling_batch_size: int = 10_000,
    show_progress_bars: bool = True,
) -> Tensor:
    r"""Given a batch of observations [x_1, ..., x_B] this function samples from
    posteriors $p(\theta|x_1)$, ... ,$p(\theta|x_B)$, in a batched (i.e. vectorized)
    manner.

    Args:
        sample_shape: Desired shape of samples that are drawn from the posterior
            given every observation.
        x: A batch of observations, of shape `(batch_dim, event_shape_x)`.
            `batch_dim` corresponds to the number of observations to be drawn.
        max_sampling_batch_size: Maximum batch size for rejection sampling.
        show_progress_bars: Whether to show sampling progress monitor.

    Returns:
        Samples from the posteriors of shape (*sample_shape, B, *input_shape)
    """
    num_samples = torch.Size(sample_shape).numel()
    x = reshape_to_batch_event(x, event_shape=condition_shape)
    num_xos = x.shape[0]

    # throw warning if num_x * num_samples is too large
    if num_xos * num_samples > 2**21:  # 2 million-ish
        warn(
            "Note that for batched sampling, the direct posterior sampling "
            "generates {num_xos} * {num_samples} = {num_xos * num_samples} "
            "samples. This can be slow and memory-intensive. Consider "
            "reducing the number of samples or batch size.",
            stacklevel=2,
        )

    max_sampling_batch_size = (
        self.max_sampling_batch_size
        if max_sampling_batch_size is None
        else max_sampling_batch_size
    )

    samples = rejection.accept_reject_sample(
        proposal=self.posterior_estimator,
        accept_reject_fn=lambda theta: within_support(self.prior, theta),
        num_samples=num_samples,
        show_progress_bars=show_progress_bars,
        max_sampling_batch_size=max_sampling_batch_size,
        proposal_sampling_kwargs={"x": x},
        alternative_method="build_posterior(..., sample_with='mcmc')",
    )[0]

    return samples


from functools import partial
from typing import Optional
from warnings import warn

from pyknos.nflows import distributions as distributions_
from pyknos.nflows import flows, transforms
from pyknos.nflows.nn import nets
from torch import Tensor, nn, relu, tanh, tensor, uint8

from sbi.utils.sbiutils import (
    standardizing_net,
    standardizing_transform,
    z_score_parser,
)
from sbi.utils.torchutils import create_alternating_binary_mask
from sbi.utils.user_input_checks import check_data_device, check_embedding_net_device
class ContextSplineMap(nn.Module):
    """
    Neural network from `context` to the spline parameters.

    We cannot use the resnet as conditioner to learn each dimension conditioned
    on the other dimensions (because there is only one). Instead, we learn the
    spline parameters directly. In the case of conditinal density estimation,
    we make the spline parameters conditional on the context. This is
    implemented in this class.
    """

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

def build_nsf(
    batch_x: Tensor,
    batch_y: Tensor,
    z_score_x: Optional[str] = "independent",
    z_score_y: Optional[str] = "independent",
    hidden_features: int = 50,
    num_transforms: int = 5,
    num_bins: int = 10,
    embedding_net: nn.Module = nn.Identity(),
    tail_bound: float = 3.0,
    hidden_layers_spline_context: int = 1,
    num_blocks: int = 2,
    dropout_probability: float = 0.0,
    use_batch_norm: bool = False,
    conditional_dim = None,
    **kwargs,
) -> nn.Module:
    """Builds NSF p(x|y).

    Args:
        batch_x: Batch of xs, used to infer dimensionality and (optional) z-scoring.
        batch_y: Batch of ys, used to infer dimensionality and (optional) z-scoring.
        z_score_x: Whether to z-score xs passing into the network, can be one of:
            - `none`, or None: do not z-score.
            - `independent`: z-score each dimension independently.
            - `structured`: treat dimensions as related, therefore compute mean and std
            over the entire batch, instead of per-dimension. Should be used when each
            sample is, for example, a time series or an image.
        z_score_y: Whether to z-score ys passing into the network, same options as
            z_score_x.
        hidden_features: Number of hidden features.
        num_transforms: Number of transforms.
        num_bins: Number of bins used for the splines.
        embedding_net: Optional embedding network for y.
        tail_bound: tail bound for each spline.
        hidden_layers_spline_context: number of hidden layers of the spline context net
            for one-dimensional x.
        num_blocks: number of blocks used for residual net for context embedding.
        dropout_probability: dropout probability for regularization in residual net.
        use_batch_norm: whether to use batch norm in residual net.
        kwargs: Additional arguments that are passed by the build function but are not
            relevant for maf and are therefore ignored.

    Returns:
        Neural network.
    """
    check_data_device(batch_x, batch_y)
    x_numel = batch_x[0].numel()
    if conditional_dim:
        y_numel = conditional_dim
    else:
        y_numel = x_numel
    print(y_numel, x_numel)

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
        block = [
            transforms.PiecewiseRationalQuadraticCouplingTransform(
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

    z_score_y_bool, structured_y = z_score_parser(z_score_y)
    if z_score_y_bool:
        # Prepend standardizing transform to y-embedding.
        embedding_net = nn.Sequential(
            standardizing_net(batch_y, structured_y), embedding_net
        )

    # Combine transforms.
    transform = transforms.CompositeTransform(transform_list)

    distribution = distributions_.StandardNormal((x_numel,))
    neural_net = flows.Flow(transform, distribution, embedding_net)

    return neural_net
from nflows.utils import torchutils
from torch.nn import functional as F

import torch
import torch.nn as nn
from nflows import transforms
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.utils import torchutils


class ResidualMaskedAutoregressiveTransform(MaskedAffineAutoregressiveTransform):
    """A MADE-based autoregressive transform that starts as an identity function,
    while also handling a predefined permutation of inputs."""

    def __init__(self, perm_indices=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # stor
        # self.perm_indices = perm_indices  # Store permutation indices
        # store perm_indices as weights with no grad to allow for checkpointing
        self.register_buffer("perm_indices", torch.tensor(perm_indices, dtype=torch.long))

    def _elementwise_forward(self, inputs, autoregressive_params):
        """Ensure near-identity initialization and handle input permutation."""
        # if self.perm_indices is not None:
        #     inputs = inputs[:, torch.argsort(self.perm_indices)]  # Apply permutation

        unconstrained_scale, shift = self._unconstrained_scale_and_shift(
            autoregressive_params
        )
        scale = 1 + unconstrained_scale + self._epsilon
        log_scale = torch.log(scale)
        outputs = scale * inputs + shift
        logabsdet = torchutils.sum_except_batch(log_scale, num_batch_dims=1)

        if self.perm_indices is not None:
            outputs = outputs[:, torch.argsort(self.perm_indices)]  # Reverse permutation

        return outputs, logabsdet

    def _elementwise_inverse(self, inputs, autoregressive_params):
        """Ensure near-identity inverse transform and handle input permutation."""
        if self.perm_indices is not None:
            inputs = inputs[:, self.perm_indices]  # Apply permutation
        unconstrained_scale, shift = self._unconstrained_scale_and_shift(
            autoregressive_params
        )
        scale = 1 + unconstrained_scale + self._epsilon
        log_scale = torch.log(scale)
        outputs = (inputs - shift) / scale
        logabsdet = -torchutils.sum_except_batch(log_scale, num_batch_dims=1)

        # if self.perm_indices is not None:
        #     outputs = outputs[:, torch.argsort(self.perm_indices)]  # Reverse permutation

        return outputs, logabsdet


def build_maf(
    batch_x: Tensor,
    batch_y: Tensor,
    z_score_x: Optional[str] = "independent",
    z_score_y: Optional[str] = "independent",
    hidden_features: int = 50,
    num_transforms: int = 5,
    embedding_net: nn.Module = nn.Identity(),
    num_blocks: int = 2,
    dropout_probability: float = 0.0,
    use_batch_norm: bool = False,
    use_residual_blocks: bool = False,
    use_identity: bool = False,
    random_permutation: bool = True,
    **kwargs,
) -> nn.Module:
    """Builds MAF p(x|y).

    Args:
        batch_x: Batch of xs, used to infer dimensionality and (optional) z-scoring.
        batch_y: Batch of ys, used to infer dimensionality and (optional) z-scoring.
        z_score_x: Whether to z-score xs passing into the network, can be one of:
            - `none`, or None: do not z-score.
            - `independent`: z-score each dimension independently.
            - `structured`: treat dimensions as related, therefore compute mean and std
            over the entire batch, instead of per-dimension. Should be used when each
            sample is, for example, a time series or an image.
        z_score_y: Whether to z-score ys passing into the network, same options as
            z_score_x.
        hidden_features: Number of hidden features.
        num_transforms: Number of transforms.
        embedding_net: Optional embedding network for y.
        num_blocks: number of blocks used for residual net for context embedding.
        dropout_probability: dropout probability for regularization in residual net.
        use_batch_norm: whether to use batch norm in residual net.
        kwargs: Additional arguments that are passed by the build function but are not
            relevant for maf and are therefore ignored.

    Returns:
        Neural network.
    """
    x_numel = batch_x[0].numel()
    # Infer the output dimensionality of the embedding_net by making a forward pass.
    # check_data_device(batch_x, batch_y)
    # check_embedding_net_device(embedding_net=embedding_net, datum=batch_y)
    y_numel = embedding_net(batch_y[:1]).numel()
    made_constructor = MaskedAffineAutoregressiveTransform if  not use_identity else ResidualMaskedAutoregressiveTransform

    if x_numel == 1:
        warn("In one-dimensional output space, this flow is limited to Gaussians")

    transform_list = []
    for _ in range(num_transforms):
            perm_indices = None  # Default: no permutation

            if random_permutation:
                perm_transform = transforms.RandomPermutation(features=x_numel)
                perm_indices = perm_transform._permutation  # Extract permutation order

            block = [
                made_constructor(
                    features=x_numel,
                    hidden_features=hidden_features,
                    context_features=y_numel,
                    num_blocks=num_blocks,
                    use_residual_blocks=use_residual_blocks,
                    random_mask=False,
                    activation=torch.tanh,
                    dropout_probability=dropout_probability,
                    use_batch_norm=use_batch_norm,
                    **dict(perm_indices=perm_indices) if use_identity else {},  # Pass permutation indices
                )
            ]

            if random_permutation:
                block.append(perm_transform)  # Add the actual permutation layer

            transform_list += block

    z_score_x_bool, structured_x = z_score_parser(z_score_x)
    if z_score_x_bool:
        transform_list = [
            standardizing_transform(batch_x, structured_x)
        ] + transform_list

    z_score_y_bool, structured_y = z_score_parser(z_score_y)
    if z_score_y_bool:
        embedding_net = nn.Sequential(
            standardizing_net(batch_y, structured_y), embedding_net
        )

    # Combine transforms.
    transform = transforms.CompositeTransform(transform_list)

    distribution = distributions_.StandardNormal((x_numel,))
    neural_net = flows.Flow(transform, distribution, embedding_net)

    return neural_net


from pyknos.nflows.transforms.splines import (
    rational_quadratic,  # pyright: ignore[reportAttributeAccessIssue]
)
def build_maf_rqs(
    batch_x: Tensor,
    batch_y: Tensor,
    z_score_x: Optional[str] = "independent",
    z_score_y: Optional[str] = "independent",
    hidden_features: int = 50,
    num_transforms: int = 5,
    embedding_net: nn.Module = nn.Identity(),
    conditional_dim = None,
    num_blocks: int = 2,
    num_bins: int = 10,
    tails: Optional[str] = "linear",
    tail_bound: float = 3.0,
    resblocks: bool = False,
    dropout_probability: float = 0.0,
    use_batch_norm: bool = False,
    min_bin_width: float = rational_quadratic.DEFAULT_MIN_BIN_WIDTH,
    min_bin_height: float = rational_quadratic.DEFAULT_MIN_BIN_HEIGHT,
    min_derivative: float = rational_quadratic.DEFAULT_MIN_DERIVATIVE,
    **kwargs,
):
    """Builds MAF p(x|y), where the diffeomorphisms are rational-quadratic
    splines (RQS).

    Args:
        batch_x: Batch of xs, used to infer dimensionality and (optional) z-scoring.
        batch_y: Batch of ys, used to infer dimensionality and (optional) z-scoring.
        z_score_x: Whether to z-score xs passing into the network, can be one of:
            - `none`, or None: do not z-score.
            - `independent`: z-score each dimension independently.
            - `structured`: treat dimensions as related, therefore compute mean and std
            over the entire batch, instead of per-dimension. Should be used when each
            sample is, for example, a time series or an image.
        z_score_y: Whether to z-score ys passing into the network, same options as
            z_score_x.
        hidden_features: Number of hidden features.
        num_transforms: Number of transforms.
        embedding_net: Optional embedding network for y.
        num_blocks: number of blocks used for residual net for context embedding.
        num_bins: Number of bins of the RQS.
        tails: Whether to use constrained or unconstrained RQS, can be one of:
            - None: constrained RQS.
            - 'linear': unconstrained RQS (RQS transformation is only
            applied on domain [-B, B], with `linear` tails, outside [-B, B],
            identity transformation is returned).
        tail_bound: RQS transformation is applied on domain [-B, B],
            `tail_bound` is equal to B.
        dropout_probability: dropout probability for regularization in residual net.
        use_batch_norm: whether to use batch norm in residual net.
        min_bin_width: Minimum bin width.
        min_bin_height: Minimum bin height.
        min_derivative: Minimum derivative at knot values of bins.
        kwargs: Additional arguments that are passed by the build function but are not
            relevant for maf and are therefore ignored.

    Returns:
        Neural network.
    """
    check_data_device(batch_x, batch_y)
    x_numel = batch_x[0].numel()
    if conditional_dim:
        y_numel = conditional_dim
    else:
        y_numel = x_numel

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
                use_residual_blocks=resblocks,
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

    z_score_y_bool, structured_y = z_score_parser(z_score_y)
    if z_score_y_bool:
        embedding_net = nn.Sequential(
            standardizing_net(batch_y, structured_y), embedding_net
        )

    # Combine transforms.
    transform = transforms.CompositeTransform(transform_list)

    distribution = distributions_.StandardNormal((x_numel,))
    neural_net = flows.Flow(transform, distribution, embedding_net)

    return neural_net