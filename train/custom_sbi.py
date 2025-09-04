from sbi.inference.snpe import PosteriorEstimator
# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.
import time
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Union
from warnings import warn

import torch
from torch import Tensor, nn, ones, optim
from torch.distributions import Distribution
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils import data
from torch.utils.tensorboard.writer import SummaryWriter
from sbi.samplers.rejection import rejection


from sbi import utils as utils
from sbi.inference import NeuralInference, check_if_proposal_has_default_x
from sbi.inference.posteriors import (
    DirectPosterior,
    MCMCPosterior,
    RejectionPosterior,
    VIPosterior,
)
from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.inference.potentials import posterior_estimator_based_potential
from sbi.utils import (
    RestrictedPrior,
    check_estimator_arg,
    handle_invalid_x,
    nle_nre_apt_msg_on_invalid_x,
    npe_msg_on_invalid_x,
    test_posterior_net_for_multi_d_x,
    validate_theta_and_x,
    warn_if_zscoring_changes_data,
    x_shape_from_simulation,
    within_support
)
from sbi.utils.sbiutils import ImproperEmpirical, mask_sims_from_prior

from sbi.inference.snpe import PosteriorEstimator
from sbi.inference.snpe.snpe_c import SNPE_C
from sbi.inference.snle.snle_base import LikelihoodEstimator
from sbi.inference.snle.snle_a import SNLE_A
from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch

from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler

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


class CustomSBITrain:
    def train(
            self,
            training_batch_size: int = 50,
            learning_rate: float = 5e-4,
            validation_fraction: float = 0.1,
            stop_after_epochs: int = 20,
            max_num_epochs: int = 2**31 - 1,
            clip_max_norm: Optional[float] = 5.0,
            calibration_kernel: Optional[Callable] = None,
            resume_training: bool = False,
            force_first_round_loss: bool = False,
            discard_prior_samples: bool = False,
            retrain_from_scratch: bool = False,
            show_train_summary: bool = False,
            dataloader_kwargs: Optional[dict] = None,
            network=None,
            optimizer=None,
            scheduler=None,
            weight_decay=0,
            test_dataloader=None,
            logger=None,
        ) -> nn.Module:
            # Load data from most recent round.
            self._round = max(self._data_round_index)

            if self._round == 0 and self._neural_net is not None:
                assert force_first_round_loss, (
                    "You have already trained this neural network. After you had trained "
                    "the network, you again appended simulations with `append_simulations"
                    "(theta, x)`, but you did not provide a proposal. If the new "
                    "simulations are sampled from the prior, you can set "
                    "`.train(..., force_first_round_loss=True`). However, if the new "
                    "simulations were not sampled from the prior, you should pass the "
                    "proposal, i.e. `append_simulations(theta, x, proposal)`. If "
                    "your samples are not sampled from the prior and you do not pass a "
                    "proposal and you set `force_first_round_loss=True`, the result of "
                    "SNPE will not be the true posterior. Instead, it will be the proposal "
                    "posterior, which (usually) is more narrow than the true posterior."
                )

            # Calibration kernels proposed in Lueckmann, Gonçalves et al., 2017.
            if calibration_kernel is None:
                calibration_kernel = lambda x: ones([len(x)], device=self._device)

            # Starting index for the training set (1 = discard round-0 samples).
            start_idx = int(discard_prior_samples and self._round > 0)

            if self.use_non_atomic_loss or hasattr(self, "_ran_final_round"):
                start_idx = self._round


            proposal = self._proposal_roundwise[-1]

            train_loader, val_loader = self.get_dataloaders(
                start_idx,
                training_batch_size,
                validation_fraction,
                resume_training,
                dataloader_kwargs=dataloader_kwargs,
            )
            if network is not None:
                self._neural_net = network
            elif self._neural_net is None or retrain_from_scratch:

                # Get theta,x to initialize NN
                theta, x, _ = self.get_simulations(starting_round=start_idx)
                # Use only training data for building the neural net (z-scoring transforms)
                self._neural_net = self._build_neural_net(
                    theta[self.train_indices].to("cpu"),
                    x[self.train_indices].to("cpu"),
                )
                self._x_shape = x_shape_from_simulation(x.to("cpu"))

                test_posterior_net_for_multi_d_x(
                    self._neural_net,
                    theta.to("cpu"),
                    x.to("cpu"),
                )

                del theta, x

            # Move entire net to device for training.
            self._neural_net.to(self._device)

            if not resume_training:
                if optimizer is not None:
                    self.optimizer = optimizer
                else:
                    self.optimizer = optim.Adam(
                        list(self._neural_net.parameters()), lr=learning_rate, weight_decay=weight_decay
                    )
                self.epoch, self._val_log_prob = 0, float("-Inf")
            best_val_log_prob = float('-inf')
            while self.epoch <= max_num_epochs and not self._converged(
                self.epoch, stop_after_epochs
            ):

                # Train for a single epoch.
                self._neural_net.train()
                train_log_probs_sum = 0
                epoch_start_time = time.time()
                for batch in train_loader:
                    # zero grad over all optimizers
                    for optimizer in self.optimizer:
                        optimizer.zero_grad()
                    # Get batches on current device.
                    theta_batch, x_batch, masks_batch = (
                        batch[0].to(self._device),
                        batch[1].to(self._device),
                        batch[2].to(self._device),
                    )

                    train_losses = self._loss(
                        theta_batch,
                        x_batch,
                        masks_batch,
                        proposal,
                        calibration_kernel,
                        force_first_round_loss=force_first_round_loss,
                    )
                    train_loss = torch.mean(train_losses)
                    train_log_probs_sum -= train_losses.sum().item()

                    train_loss.backward()
                    if clip_max_norm is not None:
                        clip_grad_norm_(
                            self._neural_net.parameters(), max_norm=clip_max_norm
                        )
                    for optimizer in self.optimizer:
                        optimizer.step()

                self.epoch += 1
                if scheduler:
                    scheduler.step()

                train_log_prob_average = train_log_probs_sum / (
                    len(train_loader) * train_loader.batch_size  # type: ignore
                )
                self._summary["training_log_probs"].append(train_log_prob_average)

                # Calculate validation performance.
                self._neural_net.eval()
                val_log_prob_sum = 0

                with torch.no_grad():
                    for batch in val_loader:
                        theta_batch, x_batch, masks_batch = (
                            batch[0].to(self._device),
                            batch[1].to(self._device),
                            batch[2].to(self._device),
                        )
                        # Take negative loss here to get validation log_prob.
                        val_losses = self._loss(
                            theta_batch,
                            x_batch,
                            masks_batch,
                            proposal,
                            calibration_kernel,
                            force_first_round_loss=force_first_round_loss,
                        )
                        val_log_prob_sum -= val_losses.sum().item()

                # Take mean over all validation samples.
                self._val_log_prob = val_log_prob_sum / (
                    len(val_loader) * val_loader.batch_size  # type: ignore
                )
                # Log validation log prob for every epoch.
                self._summary["validation_log_probs"].append(self._val_log_prob)
                self._summary["epoch_durations_sec"].append(time.time() - epoch_start_time)
                if self._val_log_prob > best_val_log_prob:
                    best_val_log_prob = self._val_log_prob
                    best_model_weights = deepcopy(self._neural_net.state_dict())
                if test_dataloader:
                    test_log_prob_sum = 0
                    with torch.no_grad():
                        for batch in test_dataloader:
                            theta_batch, x_batch = (
                                batch[0].to(self._device),
                                batch[1].to(self._device),
                            )
                            test_losses = self._loss(
                                theta_batch,
                                x_batch,
                                masks_batch,
                                proposal,
                                calibration_kernel,
                                force_first_round_loss=True,
                            )
                            test_log_prob_sum -= test_losses.sum().item()
                    if logger:
                        logger.log({"test_log_prob": test_log_prob_sum  / len(test_dataloader.dataset)})

                self._maybe_show_progress(self._show_progress_bars, self.epoch)

            self._report_convergence_at_end(self.epoch, stop_after_epochs, max_num_epochs)

            # Update summary.
            self._summary["epochs_trained"].append(self.epoch)
            self._summary["best_validation_log_prob"].append(self._best_val_log_prob)

            # Update tensorboard and summary dict.
            self._summarize(round_=self._round)

            # Update description for progress bar.
            if show_train_summary:
                print(self._describe_round(self._round, self._summary))

            self._neural_net.load_state_dict(best_model_weights)
            self._neural_net.zero_grad(set_to_none=True)

            return deepcopy(self._neural_net)
# CustomPosteriorEstimator: Subclass PosteriorEstimator
class CustomPosteriorEstimator(PosteriorEstimator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Calls PosteriorEstimator's __init__


class CustomSNPE_C(CustomSBITrain, SNPE_C, CustomPosteriorEstimator):
    def __init__(self, *args, **kwargs):
        SNPE_C.__init__(self, *args, **kwargs)  # Resolves MRO and calls the appropriate __init__
    
# CustomSNPE_C: Combine SNPE_C and CustomPosteriorEstimator
class CustomSNLE_A(CustomSBITrain, SNLE_A, LikelihoodEstimator):
    def __init__(self, *args, **kwargs):
        self.use_non_atomic_loss = False
        
        SNLE_A.__init__(self, *args, **kwargs)  # Resolves MRO and calls the appropriate __init__
        self._proposal_roundwise = [self._prior]

    def _loss(self, theta: Tensor, x: Tensor, *args, **kwargs) -> Tensor:
        r"""Return loss for SNLE, which is the likelihood of $-\log q(x_i | \theta_i)$.

        Returns:
            Negative log prob.
        """
        return -self._neural_net.log_prob(x, context=theta)

    
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

class PairedDataset(Dataset):
    """A dataset that pairs two datasets by index."""
    def __init__(self, dataset1: Dataset, dataset2: Dataset, dataset3: Dataset):
        assert len(dataset1) == len(dataset2), "Datasets must have the same length."
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.dataset3 = dataset3
        self.d3_length = len(dataset3)
        self.expensive_cheap_ratio = len(dataset2) / self.d3_length


    def __len__(self):
        return len(self.dataset1)

    def __getitem__(self, index):
        d3_index = torch.randint(0, self.d3_length, (10,), dtype=torch.int64)
        return self.dataset1[index], self.dataset2[index], self.dataset3[d3_index]
    
import wandb
from torch.nn.utils import clip_grad_norm_

class Log_SNPE(SNPE_C, CustomPosteriorEstimator):

    def train(self, *args, **kwargs):
        test_dataloader = kwargs.pop("test_dataloader")



class Joint_SNPE(SNPE_C, CustomPosteriorEstimator):
    
    def _compute_loss(self, batch, proposal, calibration_kernel, epistemic_loss, expensive_cheap_ratio, logger = None, log_type="train_"):
        """
        Compute the combined loss for a batch.
        """
        theta1, x1, mask1 = batch[0][0].to(self._device), batch[0][1].to(self._device), batch[0][2].to(self._device)
        theta2, x2, mask2 = batch[1][0].to(self._device), batch[1][1].to(self._device), batch[1][2].to(self._device)
        theta3, x3, mask3 = batch[2][0].to(self._device), batch[2][1].to(self._device), batch[2][2].to(self._device)
        theta3, x3, mask3 = theta3.flatten(0, 1), x3.flatten(0, 1), mask3.flatten(0, 1)
        # Calculate losses
        loss1 = self._neural_net.shared_loss(theta1, x1, theta2, x2, expensive_cheap_ratio)
        loss3 = self._loss(
            theta3, x3, mask3, proposal, calibration_kernel, head="surrogate"
        )
        combined_loss = torch.mean(loss1) + expensive_cheap_ratio * torch.mean(loss3)

        if epistemic_loss:
            noise_base = self._neural_net.transform_to_noise(theta1, x1, head="base")
            noise_surrogate = self._neural_net.transform_to_noise(theta2, x2, head="surrogate")
            # epistemic_contribution = torch.mean((noise_base - noise_surrogate)**2) * epistemic_loss

            epistemic_contribution = (torch.mean((noise_base - noise_surrogate)**2) + torch.mean(torch.max(torch.zeros_like(noise_surrogate), 2- torch.linalg.norm(noise_base - torch.flipud(noise_surrogate))**2)) )* epistemic_loss
            combined_loss += epistemic_contribution
            if logger:
                logger.log({f"{log_type}epistemic": epistemic_contribution})

        return combined_loss
    def train(
        self,
        training_batch_size: int = 50,
        learning_rate: float = 5e-4,
        validation_fraction: float = 0.1,
        stop_after_epochs: int = 20,
        max_num_epochs: int = 2**31 - 1,
        clip_max_norm: Optional[float] = 5.0,
        calibration_kernel: Optional[Callable] = None,
        resume_training: bool = False,
        force_first_round_loss: bool = False,
        discard_prior_samples: bool = False,
        retrain_from_scratch: bool = False,
        show_train_summary: bool = False,
        dataloader_kwargs: Optional[dict] = None,
        network=None,
        optimizer=None,
        weight_decay=0,
        epistemic_loss = False,
        loss_ratio = 1.0,
        logger=None,
    ) -> nn.Module:
        self._round = max(self._data_round_index)
        proposal = self._proposal_roundwise[-1]
        train_loader, val_loader, test_loader = self.get_dataloaders(
            0,
            training_batch_size,
            validation_fraction,
            resume_training,
            dataloader_kwargs=dataloader_kwargs,
        )

        if network is not None:
            self._neural_net = network
        elif self._neural_net is None or retrain_from_scratch:
            theta, x, _ = self.get_simulations(starting_round=self._round)
            self._neural_net = self._build_neural_net(
                theta[self.train_indices].to("cpu"),
                x[self.train_indices].to("cpu"),
            )
            self._x_shape = x_shape_from_simulation(x.to("cpu"))
            del theta, x
        best_val_log_prob = float('-inf')
        self._neural_net.to(self._device)

        if not resume_training:
            if optimizer is not None:
                self.optimizer = optimizer
            else:
                self.optimizer = optim.Adam(
                    list(self._neural_net.parameters()), lr=learning_rate, weight_decay=weight_decay
                )
            self.epoch, self._val_log_prob = 0, float("-Inf")

        while self.epoch <= max_num_epochs and not self._converged(self.epoch, stop_after_epochs):
            self._neural_net.train()
            train_log_probs_sum = 0
            epoch_start_time = time.time()

            for batch in train_loader:
                for optimizer in self.optimizer:
                    optimizer.zero_grad()
                combined_loss = self._compute_loss(
                    batch, proposal, calibration_kernel, epistemic_loss, loss_ratio, logger
                )
                combined_loss.backward()

                if clip_max_norm is not None:
                    clip_grad_norm_(self._neural_net.parameters(), max_norm=clip_max_norm)
                for optimizer in self.optimizer:
                    optimizer.step()

                train_log_probs_sum -= (combined_loss.sum().item())

            train_log_prob_average = train_log_probs_sum / (
                len(train_loader) * train_loader.batch_size  # type: ignore
            )
            self._summary["training_log_probs"].append(train_log_prob_average)
            if logger:
                logger.log({"train_log_prob": train_log_prob_average, "epoch": self.epoch})

            # Validation loss
            self._neural_net.eval()
            val_log_prob_sum = 0

            with torch.no_grad():
                for batch in val_loader:
                    theta1, x1, mask1 = batch[0][0].to(self._device), batch[0][1].to(self._device), batch[0][2].to(self._device)
                    loss1 = self._loss(
                        theta1, x1, mask1, proposal, calibration_kernel, head="base"
                    )
                    combined_loss = torch.mean(loss1)
                    val_log_prob_sum -= (combined_loss.sum().item())

            self._val_log_prob = val_log_prob_sum / (
                len(val_loader) * val_loader.batch_size  # type: ignore
            )
            self._summary["validation_log_probs"].append(self._val_log_prob)
            self._summary["epoch_durations_sec"].append(time.time() - epoch_start_time)

            if self._val_log_prob > best_val_log_prob:
                best_val_log_prob = self._val_log_prob
                best_model_weights = deepcopy(self._neural_net.state_dict())

            if logger:
                logger.log({"val_log_prob": self._val_log_prob})
            test_log_prob_sum = 0
            with torch.no_grad():
                for batch in test_loader:
                    theta1, x1, mask1 = batch[0].to(self._device), batch[1].to(self._device), batch[2].to(self._device)
                    test_loss = self._loss(theta1, x1, mask1, proposal, calibration_kernel, force_first_round_loss=True)
                    test_log_prob_sum -=(test_loss.sum().item() / len(test_loader.dataset))
            if logger:
                logger.log({"test_log_prob": test_log_prob_sum})
            self._maybe_show_progress(self._show_progress_bars, self.epoch)
            self.epoch += 1

        self._report_convergence_at_end(self.epoch, stop_after_epochs, max_num_epochs)
        self._summary["epochs_trained"].append(self.epoch)
        self._summary["best_validation_log_prob"].append(self._best_val_log_prob)
        self._summarize(round_=self._round)

        if show_train_summary:
            print(self._describe_round(self._round, self._summary))

        self._neural_net.load_state_dict(best_model_weights)
        self._neural_net.zero_grad(set_to_none=True)
        return deepcopy(self._neural_net)






    def get_dataloaders(
        self,
        starting_round: int = 0,
        training_batch_size: int = 50,
        validation_fraction: float = 0.1,
        resume_training: bool = False,
        dataloader_kwargs: Optional[dict] = None,
    ) -> Tuple[data.DataLoader, data.DataLoader]:
        """Return dataloaders for training and validation.

        Args:
            dataset: holding all theta and x, optionally masks.
            training_batch_size: training arg of inference methods.
            resume_training: Whether the current call is resuming training so that no
                new training and validation indices into the dataset have to be created.
            dataloader_kwargs: Additional or updated kwargs to be passed to the training
                and validation dataloaders (like, e.g., a collate_fn).

        Returns:
            Tuple of dataloaders for training and validation.

        """

        #
        theta, x, prior_masks = self.get_simulations(starting_round)
        theta_surr, x_surr, prior_masks_surr = self.get_simulations(starting_round + 1)
        extra_theta_surr, extra_x_surr, extra_prior_masks_surr = self.get_simulations(starting_round + 2)

        test_theta, test_x, test_masks = self.get_simulations(starting_round + 3)


        dataset = data.TensorDataset(theta, x, prior_masks)
        dataset_surr = data.TensorDataset(theta_surr, x_surr, prior_masks_surr)
        dataset_extra_surr = data.TensorDataset(extra_theta_surr, extra_x_surr, extra_prior_masks_surr)

        test_dataset = data.TensorDataset(test_theta, test_x, test_masks)

        paired_dataset = PairedDataset(dataset, dataset_surr, dataset_extra_surr)
        # Get total number of training examples.
        num_examples = theta.size(0)
        # Select random train and validation splits from (theta, x) pairs.
        num_training_examples = int((1 - validation_fraction) * num_examples)
        num_validation_examples = num_examples - num_training_examples

        if not resume_training:
            # Seperate indicies for training and validation
            permuted_indices = torch.randperm(num_examples)
            self.train_indices, self.val_indices = (
                permuted_indices[:num_training_examples],
                permuted_indices[num_training_examples:],
            )

        # Create training and validation loaders using a subset sampler.
        # Intentionally use dicts to define the default dataloader args
        # Then, use dataloader_kwargs to override (or add to) any of these defaults
        # https://stackoverflow.com/questions/44784577/in-method-call-args-how-to-override-keyword-argument-of-unpacked-dict
        train_loader_kwargs = {
            "batch_size": min(training_batch_size, num_training_examples),
            "drop_last": True,
            "sampler": SubsetRandomSampler(self.train_indices.tolist()),
        }
        val_loader_kwargs = {
            "batch_size": min(training_batch_size, num_validation_examples),
            "shuffle": False,
            "drop_last": True,
            "sampler": SubsetRandomSampler(self.val_indices.tolist()),
        }
        if dataloader_kwargs is not None:
            train_loader_kwargs = dict(train_loader_kwargs, **dataloader_kwargs)
            val_loader_kwargs = dict(val_loader_kwargs, **dataloader_kwargs)

        train_loader = data.DataLoader(paired_dataset, **train_loader_kwargs)
        val_loader = data.DataLoader(paired_dataset, **val_loader_kwargs)
        test_loader = data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

        return train_loader, val_loader,test_loader

    def get_simulations(self, round: int):
        """Return theta, x, and masks for a given round."""
        theta = self._theta_roundwise[round]
        x = self._x_roundwise[round]
        masks = self._prior_masks[round]
        return theta, x, masks

    def _loss(
        self,
        theta: Tensor,
        x: Tensor,
        masks: Tensor,
        proposal: Optional[Any],
        calibration_kernel: Callable,
        force_first_round_loss: bool = False,
        **kwargs
    ) -> Tensor:
        """Return loss with proposal correction (`round_>0`) or without it (`round_=0`).

        The loss is the negative log prob. Irrespective of the round or SNPE method
        (A, B, or C), it can be weighted with a calibration kernel.

        Returns:
            Calibration kernel-weighted negative log prob.
            force_first_round_loss: If `True`, train with maximum likelihood,
                i.e., potentially ignoring the correction for using a proposal
                distribution different from the prior.
        """
        log_prob = self._neural_net._log_prob(theta, x, **kwargs)
        return -log_prob
    
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