import iree.compiler.dialects.flow
import iree.compiler.dialects.hal
import iree.compiler.dialects.util
import iree.compiler.ir

from typing import List

from ..utils.kstat import KStat
from .graph import Graph


class Analyzer(object):
    def __init__(self, *args, **kwargs) -> "Analyzer":
        super().__init__(*args, **kwargs)

    def run(self, mod: str, entry: str, kstat: KStat) -> None:
        with iree.compiler.ir.Context():
            mod: iree.compiler.ir.Module = iree.compiler.ir.Module.parse(mod)
            func_ops: List[iree.compiler.dialects.util.FuncOp] = list(
                filter(
                    lambda op: isinstance(op, iree.compiler.dialects.util.FuncOp)
                    and op.sym_name.value == f"{entry}$async",
                    mod.body.operations,
                )
            )
            if not func_ops:
                func_ops = list(
                    filter(
                        lambda op: isinstance(op, iree.compiler.dialects.util.FuncOp)
                        and op.sym_name.value == entry,
                        mod.body.operations,
                    )
                )
            [func_op] = func_ops
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
            for subgraph in graph.partitioned():
                pass
            # TODO(Jinjie Liu): Do something more here.
