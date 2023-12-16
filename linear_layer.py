from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Sequence, Set, Tuple, Union, cast

import numpy as np
import pandas as pd
from pydantic import BaseModel
from tqdm import tqdm

from heaan_sdk.core import Block, Context
from heaan_sdk.core.base import DefaultHasContextMixin, Storable
from heaan_sdk.core.bootstrappable import bootstrap_if
from heaan_sdk.core.mod import heaan
from heaan_sdk.exceptions import SDKDeviceError, SDKHEError
from heaan_sdk.matrix import HEMatrix, HESubMatrix
from heaan_sdk.matrix.ops import mat_ops as mop
from heaan_sdk.ml.linear_model import initializers, utils
from heaan_sdk.ml.linear_model.base import LinearEstimator, LinearPredictor
from heaan_sdk.ml.linear_model.optimizers.lr_scheduler import (
    LRScheduler,
    LRSchedulerFactory,
)
from heaan_sdk.ml.linear_model.optimizers.optimizer import (
    SGD,
    Optimizer,
    OptimizerFactory,
)
from heaan_sdk.ml.linear_model.optimizers.regularizer import (
    Regularizer,
    RegularizerFactory,
)
from heaan_sdk.ml.linear_model.preprocessor.scaler import Scaler
from heaan_sdk.ml.metrics import is_higher_better, log_loss, r2_score
from heaan_sdk.util import as_shape, get_random_state
from heaan_sdk.util.check import check_and_set_unit_shape, check_pow_of_two
from heaan_sdk.util.io import (
    attr_path,
    load_attr,
    load_metadata,
    load_rns,
    save_attr,
    save_metadata,
    save_rns,
)


class Linear:
    def __init__(
        self,
        context: Context,
        unit_shape: Tuple[int, ...],
        input_dim: int,
        output_dim: int,
        bias: bool,
        param_initializer: str = "kaiming_he",
        random_state: Optional[int] = None,
        activation: str = "auto",
    ) -> None:
        self.context = context
        self.unit_shape = unit_shape
        self.input_dim = input_dim
        self.output_dim = output_dim
        self._bias = bias
        if self._bias:
            # TODO
            self.bias = None

        self.initializer = param_initializer
        self.theta = self._init_theta(self.initializer)

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.output_dim, self.input_dim + 1)

    def _init_theta(self, initializer: str = "kaiming_he") -> HESubMatrix:
        return cast(
            HESubMatrix,
            getattr(initializers, initializer)(
                self.context, self.shape, self.unit_shape, self.random_state
            ),
        )

    def forward(self, X) -> None:  # -> HESubMatrix ? or HEMatrix?
        # TODO
        return None

    def backward(self) -> None:  # -> HESubMatrix ? or HEMatrix?
        # TODO
        return None

    @property
    def encrypted(self) -> bool:
        return self.theta.encrypted
