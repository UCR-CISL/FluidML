import iree.compiler.dialects.flow
import iree.compiler.dialects.hal
import iree.compiler.dialects.util
import iree.compiler.ir

from typing import List

from .graph import Graph


class Analyzer(object):
    def __init__(self, ctx: iree.compiler.ir.Context, *args, **kwargs) -> "Analyzer":
        super().__init__(*args, **kwargs)
        self._ctx: iree.compiler.ir.Context = ctx

    def run(self, func_op: iree.compiler.dialects.util.FuncOp) -> None:
        with self._ctx:
            ops: List[iree.compiler.ir.OpView] = [
                op
                for region in func_op.regions
                for block in region.blocks
                for op in block.operations
                if (
                    isinstance(op, iree.compiler.dialects.flow.DispatchOp)
                    or isinstance(op, iree.compiler.dialects.flow.TensorEmptyOp)
                    or isinstance(op, iree.compiler.dialects.flow.TensorReshapeOp)
                    or isinstance(op, iree.compiler.dialects.flow.TensorSplatOp)
                    or isinstance(op, iree.compiler.dialects.flow.TensorUpdateOp)
                    or isinstance(op, iree.compiler.dialects.hal.TensorBarrierOp)
                    or isinstance(op, iree.compiler.dialects.hal.TensorImportOp)
                    or isinstance(op, iree.compiler.dialects.hal.TensorExportOp)
                    or isinstance(op, iree.compiler.dialects.util.GlobalLoadOp)
                )
            ]
            graph: Graph = Graph(ops)
            graph.partitioned()
            # TODO(Jinjie Liu): Do something more here.
