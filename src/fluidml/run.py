import iree.compiler.dialects.flow
import iree.compiler.dialects.hal
import iree.compiler.dialects.util
import iree.compiler.ir

from typing import List, Union

from .analyzer import Analyzer


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
        [
            func_op,
        ] = func_ops
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
        analyzer: Analyzer = Analyzer(ctx)
        analyzer.run(ops)
