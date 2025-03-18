import argparse
import iree.compiler.dialects.flow
import iree.compiler.dialects.util
import iree.compiler.ir
import json

from itertools import chain
from typing import Dict, List, Tuple

from ..utils.kstat import KStat
from ..utils.schedule import Schedule


def main():
    parser = argparse.ArgumentParser(
        description="ablation-tool for FluidML pipelines", allow_abbrev=True
    )
    parser.add_argument("filename", type=str, help="path to the IREE flow module file")
    parser.add_argument(
        "--kstat",
        type=str,
        required=True,
        help="kstat pkl",
    )
    parser.add_argument(
        "--schedule",
        type=str,
        required=True,
        help="schedule pkl",
    )
    parser.add_argument("--output", type=str, required=True, help="output file path")
    args: argparse.Namespace = parser.parse_args()
    filename: str = args.filename
    kstat: str = args.kstat
    schedule: str = args.schedule
    output: str = args.output
    with open(filename, "r") as f:
        mod: str = f.read()
    with open(kstat, "rb") as f:
        kstat: KStat = KStat.build(f)
    with open(schedule, "rb") as f:
        schedule: Schedule = Schedule.build(f)
    time_map: Dict[str, Tuple[float, float]] = _run(mod, kstat, schedule)
    etime: float = sum(e for _, e in time_map.values())
    stime: float = sum(s for s, _ in time_map.values())
    improve: float = sum(e - s for s, e in time_map.values())
    with open(output, "w") as f:
        json.dump(
            {
                "expect": etime,
                "schedule": stime,
                "improvement": improve,
                "time_map": time_map,
            },
            f,
        )


def _run(mod: str, kstat: KStat, schedule: Schedule) -> Dict[str, Tuple[float, float]]:
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
        time_map: Dict[str, Tuple[float, float]] = {}
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
                        time_map[func_name] = (stime, etime)
        return time_map


if __name__ == "__main__":
    main()
