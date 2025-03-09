import iree.compiler.dialects.arith
import iree.compiler.dialects.flow
import iree.compiler.dialects.hal
import iree.compiler.dialects.util
import iree.compiler.ir

from typing import List, Union


class OpWrapper(object):
    def __init__(self, op: iree.compiler.ir.OpView, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._op = op

    def __eq__(self, value: "OpWrapper") -> bool:
        assert isinstance(
            value, OpWrapper
        ), f"{value} is not an instance of `OpWrapper`."
        return self._op == value._op

    def __hash__(self) -> int:
        return self._op.__hash__() ^ object.__hash__(OpWrapper)

    @staticmethod
    def from_op(
        op: Union[iree.compiler.ir.Operation, iree.compiler.ir.OpView]
    ) -> "OpWrapper":
        if isinstance(op, iree.compiler.ir.Operation):
            op: iree.compiler.ir.OpView = op.opview
        assert isinstance(
            op, iree.compiler.ir.OpView
        ), f"Op {op} is not an instance of `iree.compiler.ir.OpView`."
        if (
            isinstance(op, iree.compiler.dialects.flow.TensorEmptyOp)
            or isinstance(op, iree.compiler.dialects.hal.TensorImportOp)
            or isinstance(op, iree.compiler.dialects.flow.TensorSplatOp)
            or isinstance(op, iree.compiler.dialects.util.GlobalLoadOp)
        ):
            return SourceOpWrapper(op)
        elif isinstance(op, iree.compiler.dialects.hal.TensorExportOp) or isinstance(
            op, iree.compiler.dialects.flow.TensorUpdateOp
        ):
            return DestinationOpWrapper(op)
        elif (
            isinstance(op, iree.compiler.dialects.flow.DispatchOp)
            or isinstance(op, iree.compiler.dialects.flow.TensorReshapeOp)
            or isinstance(op, iree.compiler.dialects.hal.TensorBarrierOp)
        ):
            return IntermediateOpWrapper(op)
        else:
            raise NotImplementedError(
                f"Op {op} is not supported yet, whose type is {type(op)}."
            )

    @property
    def inputs(self) -> List[iree.compiler.ir.Value]:
        return [
            op
            for op in self._op.operands
            if isinstance(op.owner, iree.compiler.ir.Operation)
            and not isinstance(op.owner.opview, iree.compiler.dialects.arith.ConstantOp)
        ]

    @property
    def input_names(self) -> List[str]:
        return [input.get_name() for input in self.inputs]

    @property
    def outputs(self) -> List[iree.compiler.ir.Value]:
        return [
            op
            for op in self._op.results
            if isinstance(op.owner, iree.compiler.ir.Operation)
            and not isinstance(op.owner.opview, iree.compiler.dialects.arith.ConstantOp)
        ]

    @property
    def output_names(self) -> List[str]:
        return [output.get_name() for output in self.outputs]

    @property
    def tensors(self) -> List[iree.compiler.ir.Value]:
        return self.inputs + self.outputs

    @property
    def tensor_names(self) -> List[str]:
        return self.input_names + self.output_names


class SourceOpWrapper(OpWrapper):
    def __init__(
        self,
        op: iree.compiler.ir.OpView,
        outputs: List[OpWrapper] = [],
        *args,
        **kwargs,
    ):
        super().__init__(op=op, *args, **kwargs)
        self._outputs: List[OpWrapper] = outputs

    @property
    def inputs(self) -> List[iree.compiler.ir.Value]:
        return []


class DestinationOpWrapper(OpWrapper):
    def __init__(
        self,
        op: iree.compiler.ir.OpView,
        inputs: List[OpWrapper] = [],
        *args,
        **kwargs,
    ):
        super().__init__(op=op, *args, **kwargs)
        self._inputs: List[OpWrapper] = inputs

    @property
    def outputs(self) -> List[iree.compiler.ir.Value]:
        return []


class IntermediateOpWrapper(OpWrapper):
    def __init__(
        self,
        op: iree.compiler.ir.OpView,
        inputs: List[OpWrapper] = [],
        outputs: List[OpWrapper] = [],
        *args,
        **kwargs,
    ):
        super().__init__(op=op, *args, **kwargs)
        self._inputs: List[OpWrapper] = inputs
        self._outputs: List[OpWrapper] = outputs
