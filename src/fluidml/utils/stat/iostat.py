import pickle

from typing import Dict, Optional

from .stat import Stat


class IOStat(Stat):
    def __init__(
        self, result: Optional[Dict[str, float]] = None, *args, **kwargs
    ) -> "IOStat":
        super().__init__(*args, **kwargs)
        if result is None:
            self._stat: Dict[str, float] = {}
        else:
            self._stat: Dict[str, float] = result

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(\n{self._stat}\n)"

    def contains(self, key: str) -> bool:
        return key in self._stat

    def get(self, key: str, default: Optional[float] = None) -> Optional[float]:
        return self._stat.get(key, default)

    def set(self, key: str, value: float) -> None:
        self._stat[key] = value

    def dump(self, f) -> None:
        pickle.dump(self._stat, f)
