import iree.compiler.ir
import iree.compiler.dialects.flow
import iree.compiler.dialects.func
import iree.compiler.dialects.util
import numpy as np
import os
import re

from typing import Dict, List, Tuple


class Profiler(object):
    def __init__(
        self,
        ctx: iree.compiler.ir.Context,
        worker_num: int = os.cpu_count(),
        *args,
        **kwargs
    ) -> "Profiler":
        super().__init__(*args, **kwargs)
        self.ctx = ctx
        self.worker_num: int = worker_num
        self.__pattern: re.Pattern = re.compile(
            r"^!flow\.dispatch\.tensor<(?:readonly|writeonly|readwrite):tensor<((?:\d+x)+f\d+)>>$"
        )
        self.__str2dtype: Dict[str, np.dtype] = {
            "f16": np.float16,
            "f32": np.float32,
            "f64": np.float64,
        }

    def run(
        self, mod: iree.compiler.ir.Module
    ) -> Dict[iree.compiler.dialects.flow.DispatchOp, Dict[Tuple[int, ...], float]]:
        kernels: List[iree.compiler.dialects.util.FuncOp] = []
        for operation in mod.body.operations:
            op_view: iree.compiler.ir.OpView = operation.opview
            if isinstance(op_view, iree.compiler.dialects.flow.ExecutableOp):
                for block in op_view.body.blocks:
                    _, builtin_mod, _ = block.operations
                    for region in builtin_mod.regions:
                        for block in region.blocks:
                            for operation in block.operations:
                                if isinstance(
                                    operation, iree.compiler.dialects.func.FuncOp
                                ):
                                    kernels += [operation]
        for kernel in kernels:
            for argument in kernel.arguments:
                tensor_type: str = str(argument.type)
                match: List[str] = self.__pattern.findall(tensor_type)
                [seq] = match
                elems: List[str] = seq.split("x")
                dtype: str = elems.pop()
                shape: Tuple[int, ...] = tuple(map(int, elems))
                dtype: np.dtype = self.__str2dtype[dtype]
