import iree.compiler.dialects.flow
import iree.compiler.dialects.hal
import iree.compiler.dialects.util
import iree.compiler.ir

from typing import List

from ..utils.kstat import KStat
from .graph import Graph
from .op_wrapper import (
    DestinationOpWrapper,
    IntermediateOpWrapper,
    OpWrapper,
    SourceOpWrapper,
)


class Analyzer(object):
    def __init__(self, *args, **kwargs) -> "Analyzer":
        super().__init__(*args, **kwargs)

    def run(self, mod: str, kstat: KStat) -> None:
        with iree.compiler.ir.Context():
            mod: iree.compiler.ir.Module = iree.compiler.ir.Module.parse(mod)
            func_ops: List[iree.compiler.dialects.util.FuncOp] = list(
                filter(
                    lambda op: isinstance(op, iree.compiler.dialects.util.FuncOp),
                    mod.body.operations,
                )
            )
            if len(func_ops) == 1:
                [func_op] = func_ops
            elif len(func_ops) == 2:
                [func_op] = list(
                    filter(lambda op: op.sym_name.value.endswith("$async"), func_ops)
                )
            else:
                raise NotImplementedError(f"Unsupported number of FuncOps: {func_ops}")
            wrappers: List[OpWrapper] = []
            for region in func_op.regions:
                for block in region.blocks:
                    for op in block.operations:
                        if (
                            isinstance(op, iree.compiler.dialects.flow.TensorEmptyOp)
                            or isinstance(op, iree.compiler.dialects.hal.TensorImportOp)
                            or isinstance(op, iree.compiler.dialects.flow.TensorSplatOp)
                            or isinstance(op, iree.compiler.dialects.util.GlobalLoadOp)
                        ):
                            wrappers += [SourceOpWrapper(op)]
                        elif isinstance(op, iree.compiler.dialects.hal.TensorExportOp):
                            wrappers += [DestinationOpWrapper(op)]
                        elif (
                            isinstance(op, iree.compiler.dialects.flow.DispatchOp)
                            or isinstance(
                                op, iree.compiler.dialects.flow.TensorReshapeOp
                            )
                            or isinstance(
                                op, iree.compiler.dialects.hal.TensorBarrierOp
                            )
                        ):
                            wrappers += [IntermediateOpWrapper(op)]
                        elif isinstance(op, iree.compiler.dialects.flow.TensorUpdateOp):
                            # Update Operation is a special case. It should be treated as a SourceOp in some graphs, and a DestinationOp in others.
                            wrappers += [SourceOpWrapper(op), DestinationOpWrapper(op)]
            graph: Graph = Graph(wrappers)
            for subgraph in graph.partitioned():
                pass
            # TODO(Jinjie Liu): Do something more here.
