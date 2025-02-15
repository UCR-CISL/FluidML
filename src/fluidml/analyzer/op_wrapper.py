import iree.compiler.dialects._flow_ops_gen
import iree.compiler.dialects._hal_ops_gen
import iree.compiler.dialects._util_ops_gen
import iree.compiler.ir

from typing import List


class OpWrapper(object):
    def __init__(self, op: iree.compiler.ir.Operation, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.op = op

    @staticmethod
    def from_op(op: iree.compiler.ir.Operation) -> "OpWrapper":
        if (
            isinstance(op, iree.compiler.dialects._flow_ops_gen.TensorSplatOp)
            or isinstance(op, iree.compiler.dialects._hal_ops_gen.TensorImportOp)
            or isinstance(op, iree.compiler.dialects._util_ops_gen.GlobalLoadOp)
        ):
            return SourceOpWrapper(op)
        elif isinstance(
            op, iree.compiler.dialects._flow_ops_gen.TensorUpdateOp
        ) or isinstance(op, iree.compiler.dialects._hal_ops_gen.TensorExportOp):
            return DestinationOpWrapper(op)
        elif isinstance(
            op, iree.compiler.dialects._flow_ops_gen.DispatchOp
        ) or isinstance(op, iree.compiler.dialects._flow_ops_gen.TensorReshapeOp):
            return IntermediateOpWrapper(op)
        else:
            raise NotImplementedError(f"Op {op} is not supported yet.")


class SourceOpWrapper(OpWrapper):
    def __init__(
        self,
        op: iree.compiler.ir.Operation,
        outputs: List[OpWrapper] = [],
        *args,
        **kwargs,
    ):
        super().__init__(op=op, *args, **kwargs)
        self.outputs: List[OpWrapper] = outputs


class DestinationOpWrapper(OpWrapper):
    def __init__(
        self,
        op: iree.compiler.ir.Operation,
        inputs: List[OpWrapper] = [],
        *args,
        **kwargs,
    ):
        super().__init__(op=op, *args, **kwargs)
        self.inputs: List[OpWrapper] = inputs


class IntermediateOpWrapper(OpWrapper):
    def __init__(
        self,
        op: iree.compiler.ir.Operation,
        inputs: List[OpWrapper] = [],
        outputs: List[OpWrapper] = [],
        *args,
        **kwargs,
    ):
        super().__init__(op=op, *args, **kwargs)
        self.inputs: List[OpWrapper] = inputs
        self.outputs: List[OpWrapper] = outputs
