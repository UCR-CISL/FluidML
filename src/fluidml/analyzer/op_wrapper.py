import iree.compiler.dialects._flow_ops_gen
import iree.compiler.dialects._hal_ops_gen
import iree.compiler.dialects._util_ops_gen
import iree.compiler.ir


class OpWrapper(object):
    def __init__(self, op: iree.compiler.ir.Operation):
        self.op = op

    @staticmethod
    def from_op(op: iree.compiler.ir.Operation) -> "OpWrapper":
        if isinstance(
            op, iree.compiler.dialects._flow_ops_gen.TensorSplatOp
        ) or isinstance(op, iree.compiler.dialects._hal_ops_gen.TensorImportOp):
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
    def __init__(self, op):
        super().__init__(op)


class DestinationOpWrapper(OpWrapper):
    def __init__(self, op):
        super().__init__(op)


class IntermediateOpWrapper(OpWrapper):
    def __init__(self, op):
        super().__init__(op)
