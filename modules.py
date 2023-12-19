from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union, Tuple

from typing_extensions import Self

# from heaan_sdk.core.context import Context
from heaan_sdk.core import Block, Context
from heaan_sdk.core.mod import heaan
from heaan_sdk.matrix import HEMatrix, HESubMatrix


class HEModules(ABC):
    """Common module for deep learning layer"""

    def __init__(self, context: Context, unit_shape: Tuple[int, ...]) -> None:
        self.context = context
        self.unit_shape = unit_shape

    @abstractmethod
    def forward(self, X: HEMatrix) -> HEMatrix:
        pass

    @abstractmethod
    def backward(self, X: HEMatrix) -> HEMatrix:
        pass
