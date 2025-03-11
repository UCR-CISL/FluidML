import iree.compiler.dialects.arith
import iree.compiler.dialects.flow
import iree.compiler.dialects.hal
import iree.compiler.dialects.util
import iree.compiler.ir

from typing import List


class OpWrapper(object):
    def __init__(self, op: iree.compiler.ir.OpView, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._op = op

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
    def input_names(self) -> List[str]:
        return [input.get_name() for input in self.inputs]

    @property
    def outputs(self) -> List[iree.compiler.ir.Value]:
        return []

    @property
    def output_names(self) -> List[str]:
        return [output.get_name() for output in self.outputs]

    @property
    def tensors(self) -> List[iree.compiler.ir.Value]:
        return self.inputs + self.outputs

    @property
    def tensor_names(self) -> List[str]:
        return self.input_names + self.output_names


class InputOpWrapper(OpWrapper):
    def __init__(
        self,
        op: iree.compiler.ir.OpView,
        *args,
        **kwargs,
    ):
        super().__init__(op=op, *args, **kwargs)

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
        *args,
        **kwargs,
    ):
        super().__init__(op=op, *args, **kwargs)

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
        *args,
        **kwargs,
    ):
        super().__init__(op=op, *args, **kwargs)


class DestinationOpWrapper(InputOpWrapper):
    def __init__(
        self,
        op: iree.compiler.ir.OpView,
        *args,
        **kwargs,
    ):
        super().__init__(op=op, *args, **kwargs)


class IntermediateOpWrapper(InputOpWrapper, OutputOpWrapper):
    def __init__(
        self,
        op: iree.compiler.ir.OpView,
        *args,
        **kwargs,
    ):
        super().__init__(op=op, *args, **kwargs)


class InterfaceOpWrapper(SourceOpWrapper, DestinationOpWrapper):
    def __init__(
        self,
        op: iree.compiler.ir.OpView,
        *args,
        **kwargs,
    ):
        super().__init__(op=op, *args, **kwargs)
