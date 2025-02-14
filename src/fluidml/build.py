import iree.compiler.dialects._flow_ops_gen
import iree.compiler.dialects._util_ops_gen
import iree.compiler.ir

from typing import List, Union


def run(flow: Union[str, bytes], entry: str):
    with iree.compiler.ir.Context():
        mod: iree.compiler.ir.Module = iree.compiler.ir.Module.parse(flow)
        func_ops: List[iree.compiler.dialects._util_ops_gen.FuncOp] = list(
            filter(
                lambda op: isinstance(op, iree.compiler.dialects._util_ops_gen.FuncOp)
                and op.sym_name.value == f"{entry}$async",
                mod.body.operations,
            )
        )
        assert (
            len(func_ops) == 1
        ), f"For entry function {entry}, expected only one async function {entry}$async, but got {len(func_ops)}."
        (func_op,) = func_ops
        ops: List[iree.compiler.ir.Operation] = [
            op
            for region in func_op.regions
            for block in region.blocks
            for op in block.operations
            if (
                isinstance(op, iree.compiler.dialects._flow_ops_gen.DispatchOp)
                or isinstance(op, iree.compiler.dialects._flow_ops_gen.TensorReshapeOp)
                or isinstance(op, iree.compiler.dialects._flow_ops_gen.TensorSplatOp)
            )
        ]
        # TODO(Jinjie Liu): Do something here.
