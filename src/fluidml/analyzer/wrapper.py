from __future__ import annotations

import enum
import iree.compiler.dialects.arith
import iree.compiler.dialects.flow
import iree.compiler.dialects.hal
import iree.compiler.dialects.util
import iree.compiler.ir

from functools import cached_property
from typing import TYPE_CHECKING, List, Optional, Union

if TYPE_CHECKING:
    from .scope import Scope


class DummyValue(enum.Enum):
    Input = enum.auto()
    Output = enum.auto()


class OpWrapper(object):
    def __init__(
        self,
        op: Union[iree.compiler.ir.Operation, iree.compiler.ir.OpView, OpWrapper],
        scope: Optional[Scope] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._scope: Optional[Scope] = scope
        if isinstance(op, iree.compiler.ir.Operation):
            self._op: iree.compiler.ir.OpView = op.opview
        elif isinstance(op, iree.compiler.ir.OpView):
            self._op: iree.compiler.ir.OpView = op
        elif isinstance(op, OpWrapper):
            self._op: iree.compiler.ir.OpView = op._op
        else:
            raise TypeError(
                f"Unsupported type {type(op)} for {self.__class__.__name__}.__init__."
            )

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
    def args(self) -> List[iree.compiler.ir.Value]:
        return self.inputs + self.outputs

    def arg_index(self, arg: iree.compiler.ir.Value) -> int:
        inputs_len: int = len(self.inputs)
        idx: int = self.args.index(arg)
        if self.tied_operands and idx >= inputs_len:
            diff: int = idx - inputs_len
            tied_idx: int = self.tied_operands[diff]
            if tied_idx >= 0:
                idx = tied_idx
        return idx

    @cached_property
    def arg_types(self) -> List[iree.compiler.ir.RankedTensorType]:
        return [arg.type for arg in self.args]

    @cached_property
    def entry(self) -> str:
        [entry_point] = self._op.entry_points
        [_, name] = entry_point.value
        return name

    @cached_property
    def inputs(self) -> List[iree.compiler.ir.Value]:
        return [
            op
            for op in self._op.operands
            if isinstance(op.owner, iree.compiler.ir.Operation)
            and isinstance(op.type, iree.compiler.ir.RankedTensorType)
            and not isinstance(op.owner.opview, iree.compiler.dialects.arith.ConstantOp)
        ]

    @cached_property
    def is_source(self) -> bool:
        return not self.scope_prevs

    @cached_property
    def is_destination(self) -> bool:
        return not self.scope_nexts

    @cached_property
    def is_intermediate(self) -> bool:
        return self.scope_prevs and self.scope_nexts

    @cached_property
    def scope_inputs(self) -> List[iree.compiler.ir.Value]:
        return [
            *{
                result
                for prev in self.scope_prevs
                for result in prev._op.results
                if result in self.inputs
            }
        ]

    @cached_property
    def scope_input(self) -> Union[iree.compiler.ir.Value, DummyValue]:
        if self.scope_inputs:
            [scope_input] = self.scope_inputs
            return scope_input
        elif self.inputs:
            [input] = self.inputs
            return input
        else:
            return DummyValue.Input

    @cached_property
    def scope_prev(self) -> Optional[OpWrapper]:
        return self._scope.get_prev(self)

    @cached_property
    def scope_prevs(self) -> List[OpWrapper]:
        if self._scope:
            return self._scope.get_prevs(self)
        else:
            return []

    @cached_property
    def outputs(self) -> List[iree.compiler.ir.Value]:
        return [
            op
            for op in self._op.results
            if isinstance(op.owner, iree.compiler.ir.Operation)
            and isinstance(op.type, iree.compiler.ir.RankedTensorType)
            and not isinstance(op.owner.opview, iree.compiler.dialects.arith.ConstantOp)
        ]

    @cached_property
    def scope_outputs(self) -> List[iree.compiler.ir.Value]:
        if self.scope_nexts:
            return [
                *{
                    result
                    for next in self.scope_nexts
                    for result in next._op.operands
                    if result in self.outputs
                }
            ]
        else:
            return self.outputs

    @cached_property
    def scope_output(self) -> Union[iree.compiler.ir.Value, DummyValue]:
        if self.scope_outputs:
            [scope_output] = self.scope_outputs
            return scope_output
        else:
            return DummyValue.Output

    @cached_property
    def scope_next(self) -> Optional[OpWrapper]:
        return self._scope.get_next(self)

    @cached_property
    def scope_nexts(self) -> List[OpWrapper]:
        if self._scope:
            return self._scope.get_nexts(self)
        else:
            return []

    @cached_property
    def scope_neighbors(self) -> List[OpWrapper]:
        return self.scope_prevs + self.scope_nexts

    @cached_property
    def schedule_layout(self) -> bool:
        return isinstance(self._op, iree.compiler.dialects.flow.DispatchOp)

    @cached_property
    def force_layout(self) -> bool:
        return (
            isinstance(self._op, iree.compiler.dialects.flow.TensorReshapeOp)
            or isinstance(self._op, iree.compiler.dialects.flow.TensorUpdateOp)
            or isinstance(self._op, iree.compiler.dialects.hal.TensorBarrierOp)
            or isinstance(self._op, iree.compiler.dialects.hal.TensorExportOp)
            or isinstance(self._op, iree.compiler.dialects.hal.TensorImportOp)
            or isinstance(self._op, iree.compiler.dialects.util.ReturnOp)
        )

    @cached_property
    def any_layout(self) -> bool:
        return (
            isinstance(self._op, iree.compiler.dialects.arith.ConstantOp)
            or isinstance(self._op, iree.compiler.dialects.flow.TensorEmptyOp)
            or isinstance(self._op, iree.compiler.dialects.flow.TensorSplatOp)
            or isinstance(self._op, iree.compiler.dialects.util.GlobalLoadOp)
        )

    @cached_property
    def tied_operands(self) -> Optional[List[int]]:
        if any(attr for attr in self._op.attributes if attr.name == "tied_operands"):
            return [i.value for i in self._op.tied_operands]
        else:
            return None
