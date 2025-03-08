import iree.compiler.dialects.flow
import iree.compiler.dialects.hal
import iree.compiler.dialects.util
import iree.compiler.ir
import os

from typing import List, Union

from .analyzer import Analyzer
from .profiler import Profiler

times: int = int(os.getenv("FLUIDML_TIME", 5))
worker_num: int = int(os.getenv("FLUIDML_WORKER_NUM", os.cpu_count()))
check_period: float = float(os.getenv("FLUIDML_CHECK_PERIOD", 5.0))


def run(flow: Union[str, bytes], entry: str, **kwargs):
    with iree.compiler.ir.Context() as ctx:
        mod: iree.compiler.ir.Module = iree.compiler.ir.Module.parse(flow, ctx)
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
        assert (
            len(func_ops) == 1
        ), f"For entry function {entry}, expected only one async function {entry}$async, but got {len(func_ops)}."
        [func_op] = func_ops
        analyzer: Analyzer = Analyzer(ctx)
        analyzer.run(func_op)
        mod: str = str(mod)
        profiler: Profiler = Profiler(times, worker_num, check_period, kwargs)
        profiler.run(mod)
