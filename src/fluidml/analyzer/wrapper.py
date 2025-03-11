from __future__ import annotations

import iree.compiler.dialects.arith
import iree.compiler.dialects.flow
import iree.compiler.dialects.hal
import iree.compiler.dialects.util
import iree.compiler.ir

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from .graph import Graph


class OpWrapper(object):
    def __init__(
        self,
        op: iree.compiler.ir.OpView,
        scope: Optional[Graph] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._scope: Optional[Graph] = scope
        self._op: iree.compiler.ir.OpView = op

    def __eq__(self, value: "OpWrapper") -> bool:
        assert isinstance(
            value, OpWrapper
        ), f"{value} is not an instance of `OpWrapper`."
        return self._op == value._op and type(self) == type(value)

    def __hash__(self) -> int:
        return hash(self._op) ^ object.__hash__(OpWrapper)

    @property
    def inputs(self) -> List[iree.compiler.ir.Value]:
        return []

    @property
    def scope_inputs(self) -> List[iree.compiler.ir.Value]:
        return self._scope.get_inputs(self)

    @property
    def input_names(self) -> List[str]:
        return [input.get_name() for input in self.inputs]

    @property
    def outputs(self) -> List[iree.compiler.ir.Value]:
        return []

    @property
    def scope_outputs(self) -> List[iree.compiler.ir.Value]:
        return self._scope.get_outputs(self)

    @property
    def output_names(self) -> List[str]:
        return [output.get_name() for output in self.outputs]

    @property
    def neighbors(self) -> List[iree.compiler.ir.Value]:
        return self.inputs + self.outputs

    @property
    def scope_neighbors(self) -> List[iree.compiler.ir.Value]:
        return self.scope_inputs + self.scope_outputs

    @property
    def neighbor_names(self) -> List[str]:
        return self.input_names + self.output_names


class InputOpWrapper(OpWrapper):
    def __init__(
        self,
        op: iree.compiler.ir.OpView,
        scope: Graph = None,
        *args,
        **kwargs,
    ):
        super().__init__(op=op, scope=scope, *args, **kwargs)

    @property
    def inputs(self) -> List[iree.compiler.ir.Value]:
        return [
            op
            for op in self._op.operands
            if isinstance(op.owner, iree.compiler.ir.Operation)
            and not isinstance(op.owner.opview, iree.compiler.dialects.arith.ConstantOp)
        ]


class OutputOpWrapper(OpWrapper):
    def __init__(
        self,
        op: iree.compiler.ir.OpView,
        scope: Graph = None,
        *args,
        **kwargs,
    ):
        super().__init__(op=op, scope=scope, *args, **kwargs)

    @property
    def outputs(self) -> List[iree.compiler.ir.Value]:
        return [
            op
            for op in self._op.results
            if isinstance(op.owner, iree.compiler.ir.Operation)
            and not isinstance(op.owner.opview, iree.compiler.dialects.arith.ConstantOp)
        ]


class SourceOpWrapper(OutputOpWrapper):
    def __init__(
        self,
        op: iree.compiler.ir.OpView,
        scope: Graph = None,
        *args,
        **kwargs,
    ):
        super().__init__(op=op, scope=scope, *args, **kwargs)


class DestinationOpWrapper(InputOpWrapper):
    def __init__(
        self,
        op: iree.compiler.ir.OpView,
        scope: Graph = None,
        *args,
        **kwargs,
    ):
        super().__init__(op=op, scope=scope, *args, **kwargs)


class IntermediateOpWrapper(InputOpWrapper, OutputOpWrapper):
    def __init__(
        self,
        op: iree.compiler.ir.OpView,
        scope: Graph = None,
        *args,
        **kwargs,
    ):
        super().__init__(op=op, scope=scope, *args, **kwargs)


class InterfaceOpWrapper(SourceOpWrapper, DestinationOpWrapper):
    def __init__(
        self,
        op: iree.compiler.ir.OpView,
        scope: Graph = None,
        *args,
        **kwargs,
    ):
        super().__init__(op=op, scope=scope, *args, **kwargs)
