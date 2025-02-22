import iree.compiler.ir
import iree.compiler.dialects.flow
import iree.compiler.dialects.func
import iree.compiler.dialects.hal
import iree.compiler.dialects.util
import numpy as np
import os
import re

from typing import Dict, List, Tuple, Union


class Profiler(object):
    def __init__(
        self,
        ctx: iree.compiler.ir.Context,
        worker_num: int = os.cpu_count(),
        *args,
        **kwargs,
    ) -> "Profiler":
        super().__init__(*args, **kwargs)
        self.ctx = ctx
        self.worker_num: int = worker_num
        self.__pattern: re.Pattern = re.compile(
            r"^!flow\.dispatch\.tensor<(readonly|writeonly|readwrite):tensor<((?:\d+x)+f\d+)>>$"
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
            kernel_name: str = str(kernel.sym_name)
            kernel_name = kernel_name.strip('"')
            input_types: List[Tuple[Tuple[int, ...], str, iree.compiler.ir.Type]] = []
            result_types: List[Tuple[Tuple[int, ...], str, iree.compiler.ir.Type]] = []
            types: List[iree.compiler.ir.Type] = []
            for argument in kernel.arguments:
                arg_type: str = str(argument.type)
                match: List[str] = self.__pattern.findall(arg_type)
                [
                    (rw, seq),
                ] = match
                elems: List[str] = seq.split("x")
                dtype: str = elems.pop()
                shape: Tuple[int, ...] = tuple(map(int, elems))
                types += [argument.type]
                info: Tuple[Tuple[int, ...], str, iree.compiler.ir.Type] = (
                    shape,
                    dtype,
                )
                if rw in {"readonly", "readwrite"}:
                    input_types += [info]
                if rw in {"writeonly", "readwrite"}:
                    result_types += [info]
            with mod.context, kernel.location:
                # TODO(Jinjie Liu): lose the module name, and remember to add it back
                fname: str = f"invoke_{kernel_name}$async"
                ftype: iree.compiler.ir.Attribute = iree.compiler.ir.Attribute.parse(
                    f"({', '.join(['!hal.buffer_view' for _ in input_types])}) -> ({', '.join(['!hal.buffer_view' for _ in result_types])})"
                )
                func_op: iree.compiler.dialects.util.FuncOp = (
                    iree.compiler.dialects.util.func(
                        fname,
                        ftype,
                        ip=iree.compiler.ir.InsertionPoint(mod.body),
                        sym_visibility="public",
                    )
                )
                block: iree.compiler.ir.Block = func_op.body.blocks.append(
                    *(
                        iree.compiler.ir.Type.parse("!hal.buffer_view")
                        for _ in input_types
                    )
                )
                with iree.compiler.ir.InsertionPoint(block):
                    arguments: List[iree.compiler.ir.OpResult] = []
                    export_types: List[iree.compiler.ir.Type] = [
                        iree.compiler.ir.Type.parse(
                            f'tensor<{"x".join([*[str(elem) for elem in result_type[0]], result_type[1]])}>'
                        )
                        for result_type in result_types
                    ]
                    for source, input_type in zip(block.arguments, input_types):
                        shape, dtype = input_type
                        source_type: iree.compiler.ir.Type = iree.compiler.ir.Type.parse(
                            f'tensor<{"x".join([*[str(elem) for elem in shape], dtype])}>'
                        )
                        source_encoding: iree.compiler.ir.TypeAttr = (
                            iree.compiler.ir.TypeAttr.get(source_type)
                        )
                        argument: iree.compiler.ir.OpResult = (
                            iree.compiler.dialects.hal.tensor_import(
                                source_type, source, source_encoding, []
                            )
                        )
                        arguments += [argument]
                    exports: Union[
                        iree.compiler.ir.OpResult, List[iree.compiler.ir.OpResult]
                    ] = iree.compiler.dialects.flow.dispatch(
                        export_types, [], [kernel_name], arguments, [], []
                    )
                    if isinstance(exports, iree.compiler.ir.OpResult):
                        exports = [exports]
                    rets: List[iree.compiler.ir.OpResult] = []
                    for export in exports:
                        target: iree.compiler.ir.Type = iree.compiler.ir.Type.parse(
                            "!hal.buffer_view"
                        )
                        source_encoding: iree.compiler.ir.TypeAttr = (
                            iree.compiler.ir.TypeAttr.get(source_type)
                        )
                        # TODO(Jinjie Liu): `export` op has an unexpected `as` attribute, and figure out how to avoid it
                        ret: iree.compiler.ir.OpResult = (
                            iree.compiler.dialects.hal.tensor_export(
                                target, export, source_encoding, []
                            )
                        )
                        rets += [ret]
                    iree.compiler.dialects.util.return_(rets)
