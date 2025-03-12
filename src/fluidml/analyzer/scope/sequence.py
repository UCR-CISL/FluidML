from typing import Iterator, List

from ..wrapper import OpWrapper
from .scope import Scope


class Sequence(Scope):
    def __init__(
        self, wrappers: Iterator[OpWrapper] = [], *args, **kwargs
    ) -> "Sequence":
        super().__init__(*args, **kwargs)
        self._wrappers: List[OpWrapper] = [*wrappers]

    def append(self, wrapper: OpWrapper) -> "Sequence":
        self._wrappers += [OpWrapper(wrapper._op, self)]
        return self

    def iter(self) -> Iterator[OpWrapper]:
        for wrapper in self._wrappers:
            yield wrapper

    def prepend(self, wrapper: OpWrapper) -> "Sequence":
        self._wrappers = [OpWrapper(wrapper._op, self)] + self._wrappers
        return self

    def put(self, wrapper: OpWrapper) -> "Sequence":
        return self.append(wrapper)
