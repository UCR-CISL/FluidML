import iree.compiler.dialects.flow
import iree.compiler.dialects.util
import iree.compiler.ir

from collections import defaultdict
from itertools import chain
from typing import Dict, List, Tuple

from ..utils.schedule import Schedule


class Generator(object):
    def __init__(self, *args, **kwargs) -> "Generator":
        super().__init__(*args, **kwargs)

    def run(self, mod: str, schedule: Schedule) -> str:
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
            plan: Dict[str, Tuple[int, ...]] = defaultdict(set)
            for region in func_op.regions:
                for block in region.blocks:
                    for op in block.operations:
                        if isinstance(op, iree.compiler.dialects.flow.DispatchOp):
                            [entry_points] = op.entry_points
                            [_, func_name] = entry_points.value
                            layouts: Tuple[Tuple[int, ...], ...] = tuple(
                                schedule[value.get_name()]
                                for value in chain(op.operands, op.results)
                            )
                            # TODO(Jinjie Liu): we just randomly assign the layout for now, and it needs improvement in the future
                            plan[func_name] = layouts
            for operation in mod.body.operations:
                if isinstance(
                    operation.opview, iree.compiler.dialects.flow.ExecutableOp
                ):
                    [block] = operation.opview.body.blocks
                    _, builtin_mod, _ = block.operations
                    [block] = builtin_mod.body.region.blocks
                    [kernel] = block.operations
                    name: str = kernel.sym_name.value
                    layouts: Tuple[Tuple[int, ...], ...] = plan[name]
                    for idx, layout in enumerate(layouts):
                        kernel.attributes[
                            f"fluidml.arg{idx}axes"
                        ] = iree.compiler.ir.ArrayAttr.parse(
                            f"[{', '.join([str(dim) for dim in layout])}]"
                        )
            return str(mod)
