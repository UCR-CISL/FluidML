import iree.compiler.dialects.util
import iree.compiler.ir

from typing import List

from ..utils.kstat import KStat
from ..utils.schedule import Schedule, ScheduleGroup
from .scope import Graph
from .wrapper import OpWrapper


class Analyzer(object):
    def __init__(self, *args, **kwargs) -> "Analyzer":
        super().__init__(*args, **kwargs)

    def run(self, mod: str, kstat: KStat) -> Schedule:
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
            wrappers: List[OpWrapper] = [
                OpWrapper(op)
                for region in func_op.regions
                for block in region.blocks
                for op in block.operations
            ]
            graph: Graph = Graph(wrappers)
            group: ScheduleGroup = ScheduleGroup()
            for subgraph in graph.partitioned():
                for seq in subgraph.pathify():
                    group |= seq.schedule(kstat)
            schedule: Schedule = group.merge()
            return schedule
