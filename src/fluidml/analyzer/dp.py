from __future__ import annotations

import iree.compiler.ir

from ..utils import KStat, Schedule, ScheduleGroup
from .analyzer import Analyzer
from .scope import Graph
from .wrapper import OpWrapper

from typing import List


class DynamicProgramAnalyzer(Analyzer):
    def __init__(self, *args, **kwargs) -> DynamicProgramAnalyzer:
        super().__init__(*args, **kwargs)

    def run(self, mod: str, kstat: KStat) -> Schedule:
        with iree.compiler.ir.Context():
            mod: iree.compiler.ir.Module = iree.compiler.ir.Module.parse(mod)
            wrappers: List[OpWrapper] = [
                OpWrapper(op)
                for region in self._filter_func_ops(mod.body.operations).regions
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
