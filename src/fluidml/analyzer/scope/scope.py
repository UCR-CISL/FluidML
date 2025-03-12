import iree.compiler.ir

from abc import abstractmethod
from typing import Iterator, List, Union

from ..wrapper import OpWrapper


class Scope(object):
    def __init__(self, *args, **kwargs) -> "Scope":
        super().__init__(*args, **kwargs)

    def __contains__(
        self, op: Union[OpWrapper, iree.compiler.ir.Operation, iree.compiler.ir.OpView]
    ) -> bool:
        return self.contains(op)

    def __iadd__(self, op: OpWrapper) -> "Scope":
        return self.put(op)

    def __iter__(self) -> Iterator[OpWrapper]:
        return self.iter()

    def contains(
        self,
        op: Union[OpWrapper, iree.compiler.ir.Operation, iree.compiler.ir.OpView],
    ) -> bool:
        if isinstance(op, iree.compiler.ir.Operation) or isinstance(
            op, iree.compiler.ir.OpView
        ):
            return any(
                map(
                    lambda wrapper: wrapper._op == op.opview,
                    self,
                )
            )
        elif isinstance(op, iree.compiler.ir.OpView):
            return any(
                map(
                    lambda wrapper: wrapper._op == op,
                    self,
                )
            )
        elif isinstance(op, OpWrapper):
            return any(
                map(
                    lambda wrapper: wrapper == op,
                    self,
                )
            )
        else:
            raise TypeError(
                f"Unsupported type {type(op)} for `{__class__.__name__}.contains.`"
            )

    def get(
        self,
        op: Union[iree.compiler.ir.Operation, iree.compiler.ir.OpView],
    ) -> OpWrapper:
        assert op in self, f"Op {op._op} is not in the graph."
        if isinstance(op, iree.compiler.ir.Operation):
            [wrapper] = list(
                filter(
                    lambda wrapper: wrapper._op == op.opview,
                    self,
                )
            )
        elif isinstance(op, iree.compiler.ir.OpView):
            [wrapper] = list(
                filter(
                    lambda wrapper: wrapper._op == op,
                    self,
                )
            )
        elif isinstance(op, OpWrapper):
            [wrapper] = list(
                filter(
                    lambda wrapper: wrapper == op,
                    self,
                )
            )
        else:
            raise TypeError(f"Unsupported type {type(op)} for Graph.get")
        return wrapper

    def get_inputs(self, op: OpWrapper) -> List[OpWrapper]:
        return [
            self.get(input.owner.opview)
            for input in op.inputs
            if input.owner.opview in self
        ]

    def get_outputs(self, op: OpWrapper) -> List[OpWrapper]:
        return [
            self.get(use.owner)
            for output in op.outputs
            for use in output.uses
            if use.owner in self
        ]

    @abstractmethod
    def iter(self) -> Iterator[OpWrapper]:
        raise NotImplementedError(
            f"Method `iter` is not implemented for {self.__class__.__name__}."
        )

    @abstractmethod
    def put(self, op: OpWrapper) -> "Scope":
        raise NotImplementedError(
            f"Method `put` is not implemented for {self.__class__.__name__}."
        )
