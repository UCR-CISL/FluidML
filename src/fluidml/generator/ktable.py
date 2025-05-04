from __future__ import annotations

import iree.compiler.ir
import iree.compiler.dialects.flow

from collections import defaultdict
from typing import Dict, Iterator, Optional, Tuple


class KTable(object):
    def __init__(self, mod: iree.compiler.ir.Module, *args, **kwargs) -> KTable:
        super().__init__(*args, **kwargs)
        self._mod: iree.compiler.ir.Module = mod
        self._table: Dict[str, Dict[str, iree.compiler.ir.ArrayAttr]] = defaultdict(
            dict
        )

    def __getitem__(
        self, input: Tuple[str, Iterator[Tuple[int, ...]]]
    ) -> iree.compiler.ir.ArrayAttr:
        name, layouts = input
        entry_points: Optional[iree.compiler.ir.ArrayAttr] = self._table[name].get(
            layouts
        )
        if entry_points:
            return entry_points
        with self._mod.context:
            for operation in filter(
                lambda operation: isinstance(
                    operation.opview, iree.compiler.dialects.flow.ExecutableOp
                ),
                self._mod.body.operations,
            ):
                [block] = operation.opview.body.blocks
                _, builtin_mod, _ = block.operations
                [block] = builtin_mod.body.region.blocks
                [kernel] = block.operations
                if kernel.sym_name.value == name:
                    with iree.compiler.ir.InsertionPoint.at_block_begin(self._mod.body):
                        layout_asm: str = "_".join(
                            map(lambda layout: "x".join(map(str, layout)), layouts)
                        )
                        mod_: iree.compiler.dialects.flow.ExecutableOp = (
                            operation.clone()
                        )
                        mod_name: str = f"{mod_.sym_name.value}_{layout_asm}"
                        mod_.sym_name = iree.compiler.ir.StringAttr.get(mod_name)
                        [block_] = mod_.opview.body.blocks
                        export, builtin_mod_, _ = block_.operations
                        [block_] = builtin_mod_.body.region.blocks
                        [kernel_] = block_.operations
                        kernel_name: str = f"{kernel_.sym_name.value}_{layout_asm}"
                        export.function_ref = iree.compiler.ir.FlatSymbolRefAttr.get(
                            kernel_name
                        )
                        export.sym_name = iree.compiler.ir.StringAttr.get(kernel_name)
                        kernel_.sym_name = iree.compiler.ir.StringAttr.get(kernel_name)
                        for idx, layout in enumerate(layouts):
                            kernel_.attributes[
                                f"fluidml.{idx}"
                            ] = iree.compiler.ir.ArrayAttr.parse(
                                f"array<i64: {', '.join([str(dim) for dim in layout])}>"
                            )
                        entry_points: iree.compiler.ir.ArrayAttr = (
                            iree.compiler.ir.Attribute.parse(
                                f'[@"{mod_name}"::@"{kernel_name}"]'
                            )
                        )
                        self._table[name][layouts] = entry_points
                        return entry_points
