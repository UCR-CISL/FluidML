from __future__ import annotations

import iree.compiler.dialects.flow
import iree.compiler.dialects.util
import iree.compiler.ir
import numpy as np

from itertools import chain
from typing import List, Tuple

from ..utils.schedule import Schedule
from ..utils.utils import map_str_dtype
from .ktable import KTable


class Generator(object):
    def __init__(self, *args, **kwargs) -> Generator:
        super().__init__(*args, **kwargs)

    def run(self, mod: str, schedule: Schedule) -> str:
        with iree.compiler.ir.Context():
            mod: iree.compiler.ir.Module = iree.compiler.ir.Module.parse(mod)
            ktable: KTable = KTable(mod)
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
                            entry_points: iree.compiler.ir.ArrayAttr = ktable[
                                func_name, layouts
                            ]
                            op.entry_points = entry_points
                        elif isinstance(op, iree.compiler.dialects.util.GlobalLoadOp):
                            for global_ in mod.body.operations:
                                if (
                                    isinstance(
                                        global_, iree.compiler.dialects.util.GlobalOp
                                    )
                                    and op.global_.value == global_.sym_name.value
                                ):
                                    layout: Tuple[int, ...] = schedule[
                                        op.result.get_name()
                                    ]
                                    np_type: np.dtype = map_str_dtype(
                                        str(global_.type_.value.element_type)
                                    )
                                    array: np.ndarray = np.array(
                                        [elem for elem in global_.initial_value]
                                    ).astype(np_type)
                                    array = array.reshape(
                                        global_.type_.value.shape
                                    ).transpose(layout)
                                    if array.dtype == np.bool_:
                                        value: str = (
                                            np.packbits(array).tobytes().hex().upper()
                                        )
                                    else:
                                        value: str = array.tobytes().hex().upper()
                                    global_.initial_value = iree.compiler.ir.Attribute.parse(
                                        f'dense<"0x{value}"> : {global_.type_.value}'
                                    )
            return str(mod)
