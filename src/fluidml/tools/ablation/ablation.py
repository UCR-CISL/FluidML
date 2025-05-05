from __future__ import annotations

import iree.compiler.dialects.flow
import iree.compiler.dialects.util
import json
import matplotlib.pyplot as plt
import pandas

from collections import defaultdict
from itertools import chain
from typing import BinaryIO, Dict, List, TextIO, Tuple

from ...utils.stat import KStat
from ...utils.schedule import Schedule


class Ablation(object):
    def __init__(self, time_map, *args, **kwargs) -> Ablation:
        super().__init__(*args, **kwargs)
        self._time_map: Dict[str, Tuple[float, float]] = time_map

    def dump(self, f: TextIO) -> None:
        json.dump(
            {
                "expect": self.expect,
                "schedule": self.schedule,
                "improvement": self.improvement,
                "time_map": self._time_map,
            },
            f,
        )

    def savefig(self, f: BinaryIO, format: str) -> None:
        df = pandas.DataFrame(
            {"expect": self.expects, "schedule": self.schedules},
            index=self._time_map.keys(),
        )
        df.plot(kind="bar")
        plt.savefig(f, format=format, bbox_inches="tight")

    @property
    def expect(self) -> float:
        return sum(self.expects)

    @property
    def expects(self) -> List[float]:
        return [e for _, e in self._time_map.values()]

    @property
    def schedule(self) -> float:
        return sum(self.schedules)

    @property
    def schedules(self) -> List[float]:
        return [s for s, _ in self._time_map.values()]

    @property
    def improvement(self) -> float:
        return self.expect - self.schedule

    @staticmethod
    def build(
        mod: str,
        kstat: KStat,
        schedule: Schedule,
    ) -> Ablation:
        """Build the ablation tool."""
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
            time_map: Dict[str, Tuple[float, float]] = defaultdict(lambda: (0.0, 0.0))
            for region in func_op.regions:
                for block in region.blocks:
                    for op in block.operations:
                        if isinstance(op, iree.compiler.dialects.flow.DispatchOp):
                            [entry_points] = op.entry_points
                            [_, func_name] = entry_points.value
                            layouts: Tuple[Tuple[int, ...], ...] = tuple(
                                tuple(range(len(value.type.shape)))
                                for value in chain(op.operands, op.results)
                            )
                            if hasattr(op, "tied_operands") and any(
                                index.value >= 0 for index in op.tied_operands
                            ):
                                layouts = layouts[: len(op.operands)]
                            etime: float = kstat[func_name, layouts]
                            layouts: Tuple[Tuple[int, ...], ...] = tuple(
                                schedule[value.get_name()]
                                for value in chain(op.operands, op.results)
                            )
                            if hasattr(op, "tied_operands") and any(
                                index.value >= 0 for index in op.tied_operands
                            ):
                                layouts = layouts[: len(op.operands)]
                            stime: float = kstat[func_name, layouts]
                            pstime, petime = time_map[func_name]
                            time_map[func_name] = (
                                pstime + stime,
                                petime + etime,
                            )
            return Ablation(time_map)

    @staticmethod
    def from_(f: TextIO) -> Ablation:
        time_map: Dict[str, Tuple[float, float]] = json.load(f).get("time_map", {})
        return Ablation(time_map)
