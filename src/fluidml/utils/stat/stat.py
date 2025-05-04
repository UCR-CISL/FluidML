from __future__ import annotations

from abc import abstractmethod
from typing import Any, BinaryIO, Dict, Iterator, Tuple, Union


class Stat(object):
    def __init__(self, *args, **kwargs) -> Stat:
        super().__init__(*args, **kwargs)

    def __contains__(self, key: Any) -> bool:
        return self.contains(key)

    def __getitem__(
        self, key: Union[str, Iterator[Tuple[int, ...]]]
    ) -> Union[Dict[Tuple[Tuple[int, ...], ...], float], float]:
        return self.get(key)

    def __setitem__(
        self,
        key: Union[str, Tuple[Tuple[int, ...], ...]],
        value: Union[float, Dict[Tuple[Tuple[int, ...], ...], float]],
    ) -> None:
        self.set(key, value)

    @abstractmethod
    def contains(self, key: Any) -> bool:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement the contains method."
        )

    @abstractmethod
    def dump(self, f: BinaryIO) -> None:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement the dump method."
        )

    @abstractmethod
    def get(
        self, key: Any, default: Any = None
    ) -> Union[Dict[Tuple[Tuple[int, ...], ...], float], float]:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement the get method."
        )

    @abstractmethod
    def set(
        self,
        key: Any,
        value: Any,
    ) -> None:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement the set method."
        )
