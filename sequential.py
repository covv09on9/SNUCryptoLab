import time
import warnings
from math import ceil
from pathlib import Path
from tqdm import tqdm
import numpy as np

from collections import OrderedDict
from itertools import chain, islice
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Set, Tuple, Union, cast
from typing_extensions import Self

from .linear_layer import Linear
from .activation_layer import *
from .modules import HEModules

from heaan_sdk.core import Block, Context
from heaan_sdk.exceptions import SDKDeviceError, SDKHEError
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
from heaan_sdk.ml.linear_model import initializers, utils

if TYPE_CHECKING:
    from heaan_sdk.ml.linear_model.datasets import DataSet


class Sequential(HEModules):
    _modules: Dict[str, HEModules]
    _params: {}

    @overload
    def __init__(self, *args: Linear) -> None:
        ...

    @overload
    def __init__(self, arg: "OrderedDict[str, Linear]") -> None:
        ...

    def __init__(
        self,
        context: Context,
        batch_size: int = 128,
        num_epoch: int = 10,
        lr: float = 0.01,
        lr_scheduler: str = "constant",
        regularizer: str = "none",
        regularize_coeff: float = 1.0,
        activation: str = "auto",
        verbose: int = 1,
        optimizer: str = "nag",
        *args,
    ):
        i = 1
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
                if isinstance(module, Linear):
                    _params[f"W{i}"] = module.theta
                    _params[f"b{i}"] = module.bias
                    i += 1

        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
                if isinstance(module, Linear):
                    _params[f"W{i}"] = module.theta
                    _params[f"b{i}"] = module.bias
                    i += 1

        self.epoch_state: int = 0
        self.step_state: int = 1
        self.initializer = initializer
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.grads = OrderedDict()
        self._init_optimizer(optimizer, **kwargs)
        self._init_regularizer(regularizer, regularize_coeff, **kwargs)
        self._init_lr_scheduler(lr, lr_scheduler, **kwargs)
        self._init_activation(activation)
        self._init_verbose(verbose)

    def _init_optimizer(self, optimizer: str, **kwargs: Any) -> None:
        self.optimizer = OptimizerFactory.create_optimizer(
            optimizer, self.context, **kwargs
        )

    def _init_regularizer(
        self, regularizer: str, regularize_coeff: float, **kwargs: Any
    ) -> None:
        if regularizer != "none":
            self.regularizer = RegularizerFactory.create_regularizer(
                regularizer, regularize_coeff, **kwargs
            )
        else:
            self.regularizer = None
        self.regularize_coeff = regularize_coeff

    def _init_lr_scheduler(self, lr: float, lr_scheduler: str, **kwargs: Any) -> None:
        self.lr = lr
        self.lr_scheduler = LRSchedulerFactory.create_lr_scheduler(
            lr_scheduler, lr, **kwargs
        )

    def _init_activation(self, activation: str) -> None:
        if hasattr(self, "is_multi_class"):
            if activation == "auto":
                if self.is_multi_class():
                    self.activation = "softmax_wide"
                else:
                    self.activation = "sigmoid_wide"
            else:
                if activation not in [
                    "sigmoid",
                    "sigmoid_wide",
                    "softmax",
                    "softmax_wide",
                    "none",
                ]:
                    raise ValueError(f"Unsupported activation: {activation}")
                self.activation = activation

    def _init_verbose(self, verbose: int) -> None:
        if verbose not in (0, 1, 2):
            raise Exception("Invalid verbose option. Must be 0, 1 or 2.")
        self.verbose = verbose

    def append(self, module: Linear) -> "Sequential":
        r"""Appends a given module to the end.

        Args:
            module (nn.Module): module to append
        """
        self.add_module(str(len(self)), module)
        return self

    def insert(self, index: int, module: Linear) -> "Sequential":
        if not isinstance(module, Linear):
            raise AssertionError(f"module should be of type: {Linear}")
        n = len(self._modules)
        if not (-n <= index <= n):
            raise IndexError(f"Index out of range: {index}")
        if index < 0:
            index += n
        for i in range(n, index, -1):
            self._modules[str(i)] = self._modules[str(i - 1)]
        self._modules[str(index)] = module
        return self

    def extend(self, sequential) -> "Sequential":
        for layer in sequential:
            self.append(layer)
        return self

    def fit(
        self,
        train: DataSet,
        *,
        val: Optional[DataSet] = None,
        eval_epoch: int = 4,
        eval_metric: Optional[str] = None,
        num_epoch: Optional[int] = None,
        batch_size: Optional[int] = None,
        optimizer: Optional[str] = None,
        lr: Optional[float] = None,
        lr_scheduler: Optional[str] = None,
        activation: Optional[str] = None,
        verbose: Optional[int] = None,
        regularizer: Optional[str] = None,
        regularize_coeff: Optional[float] = None,
        **kwargs: Any,
    ) -> Self:
        """Fit the model according to the given `train` data set.

        Args:
            train (DataSet): data set to train the model.
        """
        encrypted = self.encrypted or train.encrypted
        if encrypted and not self.context.is_bootstrappable:
            raise SDKHEError(
                f"Need a bootstrappable context for training. Maybe the parameter used ({self.context.preset_name}) "
                "is not bootstrappable, or bootstrapper is not generated."
            )

        assert self.optimizer is not None
        assert self.lr_scheduler is not None
        if self.on_gpu and not encrypted:
            raise SDKDeviceError(
                "Both model and dataset are plaintext, and cannot be trained on GPU."
            )

        self.timestamp_fit_start = time.perf_counter()

        # Check encrypted
        encrypted = train.encrypted or self.encrypted
        if encrypted and not self.encrypted:
            self.encrypt()

        if train.scaler:
            self.scaler = train.scaler.deepcopy()

        if val:
            self.eval_epoch = eval_epoch
            self.eval_metric = self._check_eval_metric(eval_metric)
            if val.scaler:
                X_val = val.scaler.scale(val.feature)
            else:
                X_val = val.feature

            X_val = utils.add_bias_col(X_val)
            y_val = val.target.deepcopy()

            import numpy as np

            y_true = np.argmax(val.target.decrypt(False).to_ndarray(), axis=1)

            # TODO: this may create #gpu copies of validation set
            if self.on_gpu:
                X_val = X_val.to(self.device)
                y_val = y_val.to(self.device)

        # Override attributes
        if num_epoch is not None:
            self.num_epoch = num_epoch
        if batch_size is not None:
            self.batch_size = batch_size
        if optimizer is not None:
            self._init_optimizer(optimizer, **kwargs)
        if (lr_scheduler or lr) is not None:
            if lr is None:
                lr = self.lr
            if lr_scheduler is None:
                lr_scheduler = self.lr_scheduler.name
            self._init_lr_scheduler(lr, lr_scheduler, **kwargs)
        if (regularizer or regularize_coeff) is not None:
            if regularize_coeff is None:
                regularize_coeff = self.regularize_coeff
            if regularizer is None:
                if self.regularizer is None:
                    raise ValueError(
                        "The name of the regularizer must be provided to update the coefficient"
                    )
                regularizer = self.regularizer.name
            self._init_regularizer(regularizer, regularize_coeff, **kwargs)
        if activation is not None:
            self._init_activation(activation)
        if verbose is not None:
            self._init_verbose(verbose)

        total_list = list(range(len(train)))
        num_batch = min(int(ceil(self.batch_size / train.unit_shape[0])), len(train))
        if train.unit_shape[0] > self.batch_size:
            warnings.warn(
                f"The actual batch size will be {train.unit_shape[0]} instead of {self.batch_size}, "
                f"since the unit size ({train.unit_shape[0]}) is larger than the batch size."
            )

        if val is not None:
            self.eval_score = self.context.zeros(
                encrypted=self.encrypted or train.encrypted
            )

        FIRST_MASK = Block(
            self.context,
            data=[1],
            encrypted=self.encrypted or (val is not None and val.encrypted),
        )

        FIRST_MASK.to(self.device)
        best_score = None  # type: Optional[Block]
        import numpy as np
        from sklearn.metrics import accuracy_score

        for epoch_idx in range(self.num_epoch):
            self.random_state.shuffle(total_list)
            batch_set = [
                total_list[idx : idx + num_batch]
                for idx in range(0, len(train), num_batch)
            ]
            if self.verbose == 0:
                pass
            elif self.verbose == 1:
                batch_set = tqdm(batch_set, desc=f"Epoch {epoch_idx + 1}")
            elif self.verbose == 2:
                print(f"Epoch {epoch_idx + 1}/{self.num_epoch}")
            else:
                raise Exception("Invalid value for verbose.")
            lr = self.lr_scheduler.get_lr(epoch=epoch_idx)
            for batch_list in batch_set:
                batch_list = self._reorder_batch(train, batch_list)
                gradient = self.get_gradient(train, batch_list)
                self.optimizer.update(
                    self.theta,
                    gradient,
                    lr,
                    t=self.step_state,
                    regularizer=self.regularizer,
                )
                self.step_state += 1
            self.epoch_state += 1

            # Model evaluation
            theta_best: HESubMatrix
            if val and (
                self.epoch_state % self.eval_epoch == 0
                or self.epoch_state == self.num_epoch
            ):
                y_pred = self.evaluate(X_val, last_activation=True, add_bias=False)
                if self.eval_metric == "log_loss":
                    score = log_loss(y_val, y_pred)
                elif self.eval_metric == "r2":
                    score = r2_score(y_val, y_pred)
                offset = (self.epoch_state - 1) // self.eval_epoch
                self.eval_score += (score * FIRST_MASK) >> offset
                self.evaluated = True

                output_arr = y_pred.decrypt_decode()
                preds = output_arr.argmax(axis=1)
                acc_ = accuracy_score(y_true, preds)
                print(
                    self.batch_size,
                    self.epoch_state,
                    acc_,
                    score.deepcopy().to_host().decrypt(False).to_ndarray()[0],
                )

                # Keep theta with best score
                if best_score:
                    e = score.compare_wide(best_score, log_range=4).bootstrap_if(
                        cost_level=3
                    )
                    bootstrap_if(score, best_score, cost_level=1, extended=True)
                    self.theta.bootstrap_if(1, extended=True)
                    assert theta_best
                    if theta_best.encrypted:
                        theta_best.bootstrap_extended(cost_level=1)
                    if not is_higher_better(self.eval_metric):
                        e = 1 - e
                    e *= FIRST_MASK
                    e.rotate_sum(inplace=True)
                    theta_best = self.theta * e + theta_best * (1 - e)
                    best_score = score * e + best_score * (1 - e)
                else:
                    theta_best = self.theta.deepcopy()
                    best_score = score

        self.theta_final = self.theta
        if val:
            self.theta = theta_best

        self.timestamp_fit_end = time.perf_counter()

        return self
