import pickle

from collections import defaultdict
from typing import Any, BinaryIO, Dict, Optional, Tuple, Union

from .stat import Stat


class KStat(Stat):
    def __init__(
        self,
        result: Optional[Dict[str, Dict[Tuple[Tuple[int, ...], ...], float]]] = None,
        *args,
        **kwargs,
    ) -> "KStat":
        super().__init__(*args, **kwargs)
        if result is None:
            self._stat: Dict[
                str, Dict[Tuple[Tuple[int, ...], ...], float]
            ] = defaultdict(dict)
        else:
            self._stat: Dict[str, Dict[Tuple[Tuple[int, ...], ...], float]] = result

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(\n{self._stat}\n)"

    def contains(self, key: Union[str, Tuple[Any, ...]]) -> bool:
        if isinstance(key, str):
            return key in self._stat
        elif isinstance(key, tuple):
            kernel, axes = key
            assert isinstance(kernel, str)
            assert all(isinstance(axis, tuple) for axis in axes)
            return kernel in self._stat and axes in self._stat[kernel]
        else:
            raise TypeError(
                f"{self.__class__.__name__} received unexpected key type {type(key)}."
            )

    def get(self, key: Union[str, Tuple[Any, ...]], default: Any = None) -> Any:
        if isinstance(key, str):
            return self._stat.get(key, default)
        elif isinstance(key, tuple):
            kernel, axes = key
            assert isinstance(kernel, str)
            assert all(isinstance(axis, tuple) for axis in axes)
            return self._stat.get(kernel, {}).get(axes, default)
        else:
            raise TypeError(
                f"{self.__class__.__name__} received unexpected key type {type(key)}."
            )

    def set(
        self,
        key: Union[str, Tuple[Tuple[int, ...], ...]],
        value: Union[float, Dict[Tuple[Tuple[int, ...], ...], float]],
    ) -> None:
        if isinstance(key, str) and isinstance(value, dict):
            self._stat[key] = value
        elif isinstance(key, tuple) and isinstance(value, float):
            kernel, axes = key
            assert isinstance(kernel, str)
            assert all(isinstance(axis, tuple) for axis in axes)
            assert all(isinstance(dim, int) for axis in axes for dim in axis)
            self._stat[kernel][axes] = value
        else:
            raise TypeError(
                f"{self.__class__.__name__} received unexpected key type {type(key)} and value type {type(value)}."
            )

    @property
    def result(self) -> Dict[str, Dict[Tuple[Tuple[int, ...], ...], float]]:
        return self._stat

    def dump(self, f: BinaryIO) -> None:
        pickle.dump(self._stat, f)
