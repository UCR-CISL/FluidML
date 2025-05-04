from __future__ import annotations

import iree.compiler.ir

from abc import abstractmethod
from typing import Iterator, List, Optional, Union

from ..wrapper import OpWrapper


class Scope(object):
    def __init__(self, *args, **kwargs) -> Scope:
        super().__init__(*args, **kwargs)

    def __contains__(
        self, op: Union[OpWrapper, iree.compiler.ir.Operation, iree.compiler.ir.OpView]
    ) -> bool:
        return self.contains(op)

    def __iadd__(self, op: OpWrapper) -> Scope:
        return self.put(op)

    def __iter__(self) -> Iterator[OpWrapper]:
        return self.iter()

    def __str__(self) -> str:
        return "{}(\n{}\n)".format(
            self.__class__.__name__,
            "\n".join(
                map(
                    lambda wrapper: f"  {wrapper}",
                    self,
                )
            ),
        )

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
        assert op in self, f"Op {op} is not in the graph."
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

    def get_input(self, op: OpWrapper) -> Optional[iree.compiler.ir.Value]:
        inputs: List[iree.compiler.ir.Value] = self.get_inputs(op)
        if inputs:
            [input] = inputs
            return input
        else:
            return None

    def get_inputs(self, op: OpWrapper) -> List[iree.compiler.ir.Value]:
        return [input for input in op.inputs if input.owner.operation in self]

    def get_prev(self, op: OpWrapper) -> Optional[OpWrapper]:
        prevs: List[OpWrapper] = self.get_prevs(op)
        if prevs:
            [prev] = prevs
            return prev
        else:
            return None

    def get_prevs(self, op: OpWrapper) -> List[OpWrapper]:
        return [self.get(input.owner) for input in self.get_inputs(op)]

    def get_output(self, op: OpWrapper) -> Optional[iree.compiler.ir.Value]:
        outputs: List[iree.compiler.ir.Value] = self.get_outputs(op)
        if outputs:
            [output] = outputs
            return output
        else:
            return None

    def get_outputs(self, op: OpWrapper) -> List[iree.compiler.ir.Value]:
        return [
            output
            for output in op.outputs
            if any(use.owner.operation in self for use in output.uses)
        ]

    def get_next(self, op: OpWrapper) -> Optional[OpWrapper]:
        nexts: List[OpWrapper] = self.get_nexts(op)
        if nexts:
            [next] = nexts
            return next
        else:
            return None

    def get_nexts(self, op: OpWrapper) -> List[OpWrapper]:
        return [
            self.get(use.owner.operation)
            for output in self.get_outputs(op)
            for use in output.uses
            if use.owner.operation in self
        ]

    @abstractmethod
    def iter(self) -> Iterator[OpWrapper]:
        raise NotImplementedError(
            f"Method `iter` is not implemented for {self.__class__.__name__}."
        )

    @abstractmethod
    def put(self, op: OpWrapper) -> Scope:
        raise NotImplementedError(
            f"Method `put` is not implemented for {self.__class__.__name__}."
        )
