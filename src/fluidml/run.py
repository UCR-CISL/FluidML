import iree.compiler.dialects.flow
import iree.compiler.dialects.hal
import iree.compiler.dialects.util
import iree.compiler.ir

from typing import List, Union

from .analyzer import Analyzer
from .profiler import Profiler


def run(flow: Union[str, bytes], entry: str):
    with iree.compiler.ir.Context() as ctx:
        mod: iree.compiler.ir.Module = iree.compiler.ir.Module.parse(flow, ctx)
        func_ops: List[iree.compiler.dialects.util.FuncOp] = list(
            filter(
                lambda op: isinstance(op, iree.compiler.dialects.util.FuncOp)
                and op.sym_name.value == f"{entry}$async",
                mod.body.operations,
            )
        )
        assert (
            len(func_ops) == 1
        ), f"For entry function {entry}, expected only one async function {entry}$async, but got {len(func_ops)}."
        [func_op] = func_ops
        analyzer: Analyzer = Analyzer(ctx)
        analyzer.run(func_op)
        profiler: Profiler = Profiler(ctx)
        profiler.run(mod)
