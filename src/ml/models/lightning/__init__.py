"""LightningModule implementations and helpers.

This package contains the (formerly monolithic) contents of
`src.ml.models.lightning_modules`, split into smaller modules.

Public API is re-exported here for convenient imports:

    from src.ml.models.lightning import NDELightningModule

The legacy import path is kept working via a thin shim in
`src/ml/models/lightning_modules.py`.
"""

from .utils import (
    load_partial_weights,
    ConditionDict,
    _move_nested_to_device,
    _BatchableTransform,
)
from .flows import _CondEmbeddingFlow, MultipleFlow
from .estimators import PatchedConditionalDensityEstimator, PatchedLikelihoodEstimator
from .base import BaseLightningModule, RegressionLightningModule, GaussianLightningModule
from .npe import NDELightningModule
from .kl import KLDRegularisedNDELightningModule
from .nle import LikelihoodNDELightningModule
from .ensemble_npe import EnsembleNDELightningModule
from .ensemble_nle import EnsembleLikelihoodNDELightningModule
from .joint import JointVMIMNLELightningModule

__all__ = [
    # utils
    "load_partial_weights",
    "ConditionDict",
    "_move_nested_to_device",
    "_BatchableTransform",
    # flow wrappers
    "_CondEmbeddingFlow",
    "MultipleFlow",
    # sbi estimator wrappers
    "PatchedConditionalDensityEstimator",
    "PatchedLikelihoodEstimator",
    # lightning modules
    "BaseLightningModule",
    "RegressionLightningModule",
    "GaussianLightningModule",
    "NDELightningModule",
    "KLDRegularisedNDELightningModule",
    "LikelihoodNDELightningModule",
    "EnsembleNDELightningModule",
    "EnsembleLikelihoodNDELightningModule",
    "JointVMIMNLELightningModule",
]
