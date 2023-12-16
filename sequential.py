from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    overload,
    Tuple,
    TypeVar,
    Union,
)
from collections import OrderedDict
from itertools import chain, islice

from .linear_layer import Linear


class Sequential:
    _modules: Dict[str, Linear]

    @overload
    def __init__(self, *args: Linear) -> None:
        ...

    @overload
    def __init__(self, arg: "OrderedDict[str, Linear]") -> None:
        ...

    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

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

    def forward(self, input):
        for module in self:
            input = module.forward(input)
        return input

    def backward(self):
        for module in reversed(self):
            module.backward()
