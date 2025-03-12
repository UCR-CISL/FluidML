from __future__ import annotations

import iree.compiler.dialects.arith
import iree.compiler.dialects.flow
import iree.compiler.dialects.hal
import iree.compiler.dialects.util
import iree.compiler.ir

from functools import cached_property
from typing import TYPE_CHECKING, List, Optional, Union

if TYPE_CHECKING:
    from .scope import Scope


class OpWrapper(object):
    def __init__(
        self,
        op: iree.compiler.ir.OpView,
        scope: Optional[Scope] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._scope: Optional[Scope] = scope
        self._op: iree.compiler.ir.OpView = op

    def __eq__(
        self,
        value: Union[iree.compiler.ir.Operation, iree.compiler.ir.OpView, "OpWrapper"],
    ) -> bool:
        if isinstance(value, iree.compiler.ir.Operation):
            return self._op == value.opview
        elif isinstance(value, iree.compiler.ir.OpView):
            return self._op == value
        elif isinstance(value, OpWrapper):
            return self._op == value._op
        else:
            raise TypeError(
                f"Unsupported type {type(value)} for {self.__class__.__name__}.__eq__."
            )

    def __hash__(self) -> int:
        return hash(self._op) ^ object.__hash__(OpWrapper)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self._op})"

    @cached_property
    def inputs(self) -> List[iree.compiler.ir.Value]:
        return [
            op
            for op in self._op.operands
            if isinstance(op.owner, iree.compiler.ir.Operation)
            and not isinstance(op.owner.opview, iree.compiler.dialects.arith.ConstantOp)
        ]

    @cached_property
    def scope_inputs(self) -> List[iree.compiler.ir.Value]:
        if self._scope:
            return self._scope.get_inputs(self)
        else:
            return []

    @cached_property
    def input_names(self) -> List[str]:
        return [input.get_name() for input in self.inputs]

    @cached_property
    def outputs(self) -> List[iree.compiler.ir.Value]:
        return [
            op
            for op in self._op.results
            if isinstance(op.owner, iree.compiler.ir.Operation)
            and not isinstance(op.owner.opview, iree.compiler.dialects.arith.ConstantOp)
        ]

    @cached_property
    def scope_outputs(self) -> List[iree.compiler.ir.Value]:
        if self._scope:
            return self._scope.get_outputs(self)
        else:
            return []

    @cached_property
    def output_names(self) -> List[str]:
        return [output.get_name() for output in self.outputs]

    @cached_property
    def neighbors(self) -> List[iree.compiler.ir.Value]:
        return self.inputs + self.outputs

    @cached_property
    def scope_neighbors(self) -> List[iree.compiler.ir.Value]:
        return self.scope_inputs + self.scope_outputs

    @cached_property
    def neighbor_names(self) -> List[str]:
        return self.input_names + self.output_names

    @cached_property
    def is_source(self) -> bool:
        return not self.scope_inputs

    @cached_property
    def is_destination(self) -> bool:
        return not self.scope_outputs

    @cached_property
    def is_intermediate(self) -> bool:
        return self.scope_inputs and self.scope_outputs
