from __future__ import annotations

import iree.compiler.ir

from ..utils.stat import KStat
from ..utils.schedule import Schedule
from .analyzer import Analyzer
from .scope.graph import Graph
from .wrapper import OpWrapper

from typing import Dict, List, Tuple


class GreedyAnalyzer(Analyzer):
    def __init__(self, *args, **kwargs) -> GreedyAnalyzer:
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
            save_table: Dict[
                OpWrapper, Tuple[Tuple[int, ...], Tuple[int, ...], float]
            ] = {}
            schedule: Dict[str, Tuple[int, ...]] = {}
            for wrapper in graph.iter():
                if wrapper.schedule_layout:
                    entry: str = wrapper.entry
                    ktable: Dict[Tuple[Tuple[int, ...], ...], float] = kstat[entry]
                    [(default_layout, default_timecost)] = [
                        (k, v)
                        for k, v in ktable.items()
                        if all(
                            shape == tuple(i for i, _ in enumerate(shape))
                            for shape in k
                        )
                    ]
                    (best_layout, best_timecost) = min(
                        ktable.items(), key=lambda x: x[1]
                    )
                    save_table[wrapper] = (
                        default_layout,
                        best_layout,
                        default_timecost - best_timecost,
                    )
            for wrapper, (_, best_layout, _) in sorted(
                save_table.items(), key=lambda x: -x[1][2]
            ):
                for v in wrapper.args:
                    vname: str = v.get_name()
                    index: int = wrapper.arg_index(v)
                    if vname not in schedule:
                        schedule[vname] = best_layout[index]
        return Schedule(schedule)
