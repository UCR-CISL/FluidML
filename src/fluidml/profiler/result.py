import json

from collections import defaultdict
from typing import Any, Dict, Optional, TextIO, Tuple, Union


class ProfileResult(object):
    def __init__(
        self,
        result: Optional[Dict[str, Dict[Tuple[Tuple[int, ...], ...], float]]] = None,
        *args,
        **kwargs,
    ) -> "ProfileResult":
        super().__init__(*args, **kwargs)
        if result is None:
            self._result: Dict[
                str, Dict[Tuple[Tuple[int, ...], ...], float]
            ] = defaultdict(dict)
        else:
            self._result: Dict[str, Dict[Tuple[Tuple[int, ...], ...], float]] = result

    def __getitem__(
        self, key: Union[str, Tuple[Any, ...]]
    ) -> Dict[Tuple[int, ...], float]:
        if isinstance(key, str):
            return self._result[key]
        elif isinstance(key, tuple):
            kernel = key[0]
            assert isinstance(kernel, str)
            axes = key[1:]
            assert all(isinstance(axis, tuple) for axis in axes)
            return self._result[kernel][axes]
        else:
            raise TypeError(f"received unexpected key type {type(key)}")

    def __setitem__(
        self,
        key: Union[str, Tuple[Any, ...]],
        value: Union[float, Dict[Tuple[int, ...], float]],
    ) -> None:
        if isinstance(key, str) and isinstance(value, dict):
            self._result[key] = value
        elif isinstance(key, tuple) and isinstance(value, float):
            kernel = key[0]
            assert isinstance(kernel, str)
            axes = key[1:]
            assert all(isinstance(axis, tuple) for axis in axes)
            self._result[kernel][axes] = value
        else:
            raise TypeError(
                f"received unexpected key type {type(key)} and value type {type(value)}"
            )

    def __str__(self) -> str:
        return str(self._result)

    @property
    def result(self) -> Dict[str, Dict[Tuple[Tuple[int, ...], ...], float]]:
        return self._result

    def dump(self, f: TextIO) -> None:
        result: Dict[str, Dict[Tuple[Tuple[int, ...], ...], float]] = {
            k0: {str(k1): v1 for k1, v1 in v0.items()}
            for k0, v0 in self._result.items()
        }
        json.dump(result, f)
