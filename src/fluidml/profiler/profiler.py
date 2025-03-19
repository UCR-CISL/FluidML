import concurrent.futures
import gc
import multiprocessing
import iree.compiler.ir
import iree.compiler.dialects.flow
import iree.compiler.dialects.func
import iree.compiler.dialects.hal
import iree.compiler.dialects.util
import iree.runtime
import numpy as np
import os
import sys
import time

from itertools import product
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


from ..utils.kstat import KStat
from ..utils.utils import permute_shape, map_str_dtype
from .util import get_signature


class Profiler(object):
    def __init__(
        self,
        times: int,
        worker_num: int,
        check_period: float,
        profile_cache: Optional[str],
        compile_options: Dict[str, Any],
        *args,
        **kwargs,
    ) -> "Profiler":
        super().__init__(*args, **kwargs)
        self._times: int = times
        self._worker_num: int = worker_num
        self._check_period: float = check_period
        self._profile_cache: Optional[str] = profile_cache
        self._compile_commands: Dict[str, Any] = compile_options
        extra_args: str = compile_options.get("extra_args", [])
        extra_args = [
            arg for arg in extra_args if not arg.startswith("--compile-from=")
        ] + ["--compile-from=flow"]
        compile_options["extra_args"] = extra_args

    def run(self, mod: str) -> KStat:
        mp_context: multiprocessing.context.BaseContext = multiprocessing.get_context(
            "spawn"
        )
        kstat: KStat = KStat()
        with iree.compiler.ir.Context(), concurrent.futures.ProcessPoolExecutor(
            max_workers=self._worker_num, mp_context=mp_context
        ) as executor:
            mod: iree.compiler.ir.Module = iree.compiler.ir.Module.parse(mod)
            for operation in mod.body.operations:
                if isinstance(operation.opview, iree.compiler.dialects.util.GlobalOp):
                    global_op: iree.compiler.dialects.util.GlobalOp = operation.opview
                if isinstance(
                    operation.opview, iree.compiler.dialects.flow.ExecutableOp
                ):
                    with iree.compiler.ir.Context(), iree.compiler.ir.Location.unknown():
                        sub_mod: iree.compiler.ir.Module = (
                            iree.compiler.ir.Module.create(mod.operation.location)
                        )
                        for attr in mod.operation.attributes:
                            sub_mod.operation.attributes[attr.name] = attr.attr
                        [sub_block] = sub_mod.body.region.blocks
                        iree.compiler.dialects.util.global_(
                            global_op.sym_name,
                            global_op.type_,
                            sym_visibility=global_op.sym_visibility,
                            is_mutable=global_op.is_mutable,
                            initial_value=global_op.initial_value,
                            inlining_policy=global_op.inlining_policy,
                            loc=global_op.location,
                            ip=iree.compiler.ir.InsertionPoint(sub_mod.body),
                        )
                        sub_block.append(operation)
                        [_, operation] = sub_mod.body.operations
                        [block] = operation.opview.body.blocks
                        _, builtin_mod, _ = block.operations
                        [block] = builtin_mod.body.region.blocks
                        [kernel] = block.operations
                        assert isinstance(kernel, iree.compiler.dialects.func.FuncOp)
                        (
                            kernel_name,
                            mod_name,
                            input_types,
                            result_types,
                            _,
                        ) = get_signature(kernel)
                        fname: str = f"invoke_{kernel_name}$async"
                        ftype: iree.compiler.ir.Attribute = iree.compiler.ir.Attribute.parse(
                            f"({', '.join(['!hal.buffer_view' for _ in input_types])}) -> ({', '.join(['!hal.buffer_view' for _ in result_types])})"
                        )
                        func_op: iree.compiler.dialects.util.FuncOp = (
                            iree.compiler.dialects.util.func(
                                fname,
                                ftype,
                                ip=iree.compiler.ir.InsertionPoint(sub_mod.body),
                                sym_visibility="public",
                            )
                        )
                        block: iree.compiler.ir.Block = func_op.body.blocks.append(
                            *(
                                iree.compiler.ir.Type.parse("!hal.buffer_view")
                                for _ in input_types
                            ),
                        )
                        with iree.compiler.ir.InsertionPoint(block):
                            arguments: List[iree.compiler.ir.OpResult] = []
                            export_types: List[iree.compiler.ir.Type] = [
                                iree.compiler.ir.Type.parse(
                                    f'tensor<{"x".join([*[str(elem) for elem in shape], ftype])}>'
                                )
                                for (shape, ftype) in result_types
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
                                iree.compiler.ir.Operation,
                                iree.compiler.ir.OpResult,
                                List[iree.compiler.ir.Operation],
                                List[iree.compiler.ir.OpResult],
                            ] = iree.compiler.dialects.flow.dispatch(
                                export_types,
                                [],
                                iree.compiler.ir.Attribute.parse(
                                    f'[@"{mod_name}"::@"{kernel_name}"]'
                                ),
                                arguments,
                                [],
                                [],
                            )
                            if not isinstance(exports, list):
                                exports = [exports]
                            rets: List[iree.compiler.ir.Operation] = []
                            for export in exports:
                                if isinstance(export, iree.compiler.ir.OpResult):
                                    target: iree.compiler.ir.Type = (
                                        iree.compiler.ir.Type.parse("!hal.buffer_view")
                                    )
                                    source_encoding: iree.compiler.ir.TypeAttr = (
                                        iree.compiler.ir.TypeAttr.get(export.type)
                                    )
                                    ret: iree.compiler.ir.OpResult = (
                                        iree.compiler.dialects.hal.tensor_export(
                                            target, export, source_encoding, []
                                        )
                                    )
                                    rets += [ret]
                            iree.compiler.dialects.util.return_(rets)
                            combinations: List[List[Tuple[int, ...]]] = [
                                list(permute_shape(shape))
                                for shape, _ in input_types + result_types
                            ]
                            combinations: List[Tuple[Tuple[int, ...], ...]] = list(
                                product(*combinations)
                            )
                            sub_mod_texts: List[str] = []
                            for combination in combinations:
                                for idx, layout in enumerate(combination):
                                    kernel.attributes[
                                        f"fluidml.{idx}"
                                    ] = iree.compiler.ir.Attribute.parse(
                                        f"array<i64: {', '.join([str(dim) for dim in layout])}>"
                                    )
                                sub_mod_text: str = str(sub_mod)
                                if self._profile_cache is not None:
                                    sub_mod_path: str = os.path.join(
                                        self._profile_cache,
                                        f'{kernel_name}_{"_".join("x".join(map(str,layout)) for layout in combination)}.mlir',
                                    )
                                    with open(sub_mod_path, "w") as f:
                                        f.write(sub_mod_text)
                                sub_mod_texts += [sub_mod_text]
                            sub_mods: List[
                                Tuple[Tuple[Tuple[int, ...], ...], Optional[bytes]]
                            ] = list(
                                zip(
                                    combinations,
                                    executor.map(
                                        Profiler._compile_sub_modules_wrapper,
                                        map(
                                            lambda sub_mod_text: (
                                                sub_mod_text,
                                                self._compile_commands,
                                            ),
                                            sub_mod_texts,
                                        ),
                                    ),
                                )
                            )
                            sub_mods: List[
                                Tuple[Tuple[Tuple[int, ...], ...], bytes]
                            ] = [
                                (layouts, buffer)
                                for layouts, buffer in sub_mods
                                if buffer is not None
                            ]
                            for layouts, buffer in sub_mods:
                                config: iree.runtime.Config = iree.runtime.Config(
                                    "local-task"
                                )
                                ctx: iree.runtime.SystemContext = (
                                    iree.runtime.SystemContext(config=config)
                                )
                                vm_module: iree.runtime.VmModule = (
                                    iree.runtime.VmModule.copy_buffer(
                                        ctx.instance, buffer
                                    )
                                )
                                ctx.add_vm_module(vm_module)
                                inputs: List[np.ndarray] = [
                                    np.random.rand(*shape).astype(map_str_dtype(dtype))
                                    for shape, dtype in input_types
                                ]
                                f: Callable = ctx.modules.module[fname]
                                for _ in range(self._times // 10):
                                    f(*inputs)
                                exec_time: float = sys.float_info.max
                                for _ in range(self._times):
                                    gc.disable()
                                    start: int = time.perf_counter_ns()
                                    try:
                                        f(*inputs)
                                    except Exception as e:
                                        gc.enable()
                                        raise e
                                    end: int = time.perf_counter_ns()
                                    gc.enable()
                                    exec_time = min(exec_time, end * 1.0 - start)
                                kstat[kernel_name, layouts] = exec_time
        return kstat

    @staticmethod
    def _compile_sub_modules_wrapper(args: Tuple[str, Dict[str, Any]]) -> Optional[str]:
        return Profiler._compile_sub_module(*args)

    @staticmethod
    def _compile_sub_module(
        sub_mod_text: str, compile_options: Dict[str, Any]
    ) -> Optional[str]:
        try:
            return iree.compiler.compile_str(sub_mod_text, **compile_options)
        except iree.compiler.CompilerToolError:
            return None
