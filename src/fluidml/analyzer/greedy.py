from __future__ import annotations

import iree.compiler.ir

from ..utils import KStat, Schedule, is_default_layout
from .analyzer import Analyzer
from .scope.graph import Graph
from .wrapper import OpWrapper

from typing import Dict, List, Optional, Set, Tuple


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
            schedule: Schedule = Schedule()
            schedule_wrappers: Set[OpWrapper] = {
                wrapper for wrapper in graph.iter() if wrapper.schedule_layout
            }
            while schedule_wrappers:
                candidates: List[Tuple[OpWrapper, Tuple[int, ...], float]] = []
                for wrapper in schedule_wrappers:
                    entry: str = wrapper.entry
                    ktable: Dict[Tuple[Tuple[int, ...], ...], float] = kstat[entry]
                    [(_, default_timecost)] = [
                        (k, v) for k, v in ktable.items() if is_default_layout(k)
                    ]
                    (best_layout, best_timecost) = self._find_best_layout(
                        wrapper, schedule, ktable
                    )
                    candidates += [
                        (
                            wrapper,
                            best_layout,
                            default_timecost - best_timecost,
                        )
                    ]
                candidate, best_layout, _ = max(candidates, key=lambda x: x[2])
                schedule_wrappers.remove(candidate)
                for arg in candidate.args:
                    name: str = arg.get_name()
                    if name not in schedule:
                        index: int = candidate.arg_index(arg)
                        schedule[name] = best_layout[index]
        return schedule

    @staticmethod
    def _find_best_layout(
        wrapper: OpWrapper,
        schedule: Schedule,
        ktable: Dict[Tuple[Tuple[int, ...], ...], float],
    ) -> Tuple[Tuple[int, ...], float]:
        assigned_layouts: Tuple[Optional[Tuple[int, ...]], ...] = tuple(
            v
            for _, v in sorted(
                (
                    (
                        wrapper.arg_index(arg),
                        schedule.get(arg.get_name()),
                    )
                    for arg in wrapper.args
                ),
                key=lambda x: x[0],
            )
        )
        fktable: Dict[Tuple[Tuple[int, ...], ...], float] = {
            k: v
            for k, v in ktable.items()
            if all(b is None or a == b for a, b in zip(k, assigned_layouts))
        }
        return min(fktable.items(), key=lambda x: x[1])
