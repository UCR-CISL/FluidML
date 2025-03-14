import json

from collections import defaultdict
from typing import Any, Dict, Optional, TextIO, Tuple, Union


class KStat(object):
    def __init__(
        self,
        result: Optional[Dict[str, Dict[Tuple[Tuple[int, ...], ...], float]]] = None,
        *args,
        **kwargs,
    ) -> "KStat":
        super().__init__(*args, **kwargs)
        if result is None:
            self._kstat: Dict[
                str, Dict[Tuple[Tuple[int, ...], ...], float]
            ] = defaultdict(dict)
        else:
            self._kstat: Dict[str, Dict[Tuple[Tuple[int, ...], ...], float]] = result

    def __contains__(self, key: Union[str, Tuple[Any, ...]]) -> bool:
        return self.contains(key)

    def __getitem__(
        self, key: Union[str, Tuple[Tuple[int, ...], ...]]
    ) -> Union[Dict[Tuple[Tuple[int, ...], ...], float], float]:
        if isinstance(key, str):
            return self._kstat[key]
        elif isinstance(key, tuple):
            kernel, axes = key
            assert isinstance(kernel, str)
            assert all(isinstance(axis, tuple) for axis in axes)
            assert all(isinstance(dim, int) for axis in axes for dim in axis)
            return self._kstat[kernel][axes]
        else:
            raise TypeError(
                f"{self.__class__.__name__} received unexpected key type {type(key)}."
            )

    def __setitem__(
        self,
        key: Union[str, Tuple[Tuple[int, ...], ...]],
        value: Union[float, Dict[Tuple[Tuple[int, ...], ...], float]],
    ) -> None:
        if isinstance(key, str) and isinstance(value, dict):
            self._kstat[key] = value
        elif isinstance(key, tuple) and isinstance(value, float):
            kernel, axes = key
            assert isinstance(kernel, str)
            assert all(isinstance(axis, tuple) for axis in axes)
            assert all(isinstance(dim, int) for axis in axes for dim in axis)
            self._kstat[kernel][axes] = value
        else:
            raise TypeError(
                f"{self.__class__.__name__} received unexpected key type {type(key)} and value type {type(value)}."
            )

    def __str__(self) -> str:
        return str(self._kstat)

    def contains(self, key: Union[str, Tuple[Any, ...]]) -> bool:
        if isinstance(key, str):
            return key in self._kstat
        elif isinstance(key, tuple):
            kernel, axes = key
            assert isinstance(kernel, str)
            assert all(isinstance(axis, tuple) for axis in axes)
            return kernel in self._kstat and axes in self._kstat[kernel]
        else:
            raise TypeError(
                f"{self.__class__.__name__} received unexpected key type {type(key)}."
            )

    @property
    def result(self) -> Dict[str, Dict[Tuple[Tuple[int, ...], ...], float]]:
        return self._kstat

    @classmethod
    def build(cls, f: TextIO) -> "KStat":
        kstat: Dict[str, Dict[str, float]] = json.load(f)
        kstat: Dict[str, Dict[Tuple[Tuple[int, ...], ...], float]] = {
            k0: {eval(k1): v1 for k1, v1 in v0.items()} for k0, v0 in kstat.items()
        }
        return cls(kstat)

    def dump(self, f: TextIO) -> None:
        result: Dict[str, Dict[Tuple[Tuple[int, ...], ...], float]] = {
            k0: {str(k1): v1 for k1, v1 in v0.items()} for k0, v0 in self._kstat.items()
        }
        json.dump(result, f)
