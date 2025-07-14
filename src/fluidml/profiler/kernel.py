from __future__ import annotations

import concurrent.futures
import cuda.bindings.runtime
import gc
import multiprocessing
import iree.compiler.ir
import iree.compiler.dialects.flow
import iree.compiler.dialects.func
import iree.compiler.dialects.util
import iree.runtime
import numpy as np
import os
import sys
import time

from itertools import product
from typing import Any, Callable, Dict, List, Optional, Tuple


from ..utils import KStat, permute_shape, map_str_dtype
from .profiler import Profiler
from .util import get_signature


class KernelProfiler(Profiler):
    def __init__(
        self,
        times: int,
        worker_num: int,
        check_period: float,
        driver: str,
        profile_cache: Optional[str],
        compile_options: Dict[str, Any],
        *args,
        **kwargs,
    ) -> KernelProfiler:
        super().__init__(
            times,
            worker_num,
            check_period,
            driver,
            profile_cache,
            compile_options,
            *args,
            **kwargs,
        )

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
                            _,
                            input_types,
                            result_types,
                            _,
                        ) = get_signature(kernel)
                        fname: str = self._build_benchmark(
                            kernel, iree.compiler.ir.InsertionPoint(sub_mod.body)
                        )
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
                                kernel.attributes[f"fluidml.{idx}"] = (
                                    iree.compiler.ir.Attribute.parse(
                                        f"array<i64: {', '.join([str(dim) for dim in layout])}>"
                                    )
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
                                    KernelProfiler._compile_sub_modules_wrapper,
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
                        sub_mods: List[Tuple[Tuple[Tuple[int, ...], ...], bytes]] = [
                            (layouts, buffer)
                            for layouts, buffer in sub_mods
                            if buffer is not None
                        ]
                        for layouts, buffer in sub_mods:
                            config: iree.runtime.Config = iree.runtime.Config(
                                self._driver
                            )
                            ctx: iree.runtime.SystemContext = (
                                iree.runtime.SystemContext(config=config)
                            )
                            vm_module: iree.runtime.VmModule = (
                                iree.runtime.VmModule.copy_buffer(ctx.instance, buffer)
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
                                if self._driver == "cuda":
                                    (
                                        error,
                                        start_event,
                                    ) = cuda.bindings.runtime.cudaEventCreate()
                                    assert (
                                        error
                                        == cuda.bindings.runtime.cudaError_t.cudaSuccess
                                    ), f"cudaEventCreate failed: {error}"
                                    (
                                        error,
                                        end_event,
                                    ) = cuda.bindings.runtime.cudaEventCreate()
                                    assert (
                                        error
                                        == cuda.bindings.runtime.cudaError_t.cudaSuccess
                                    ), f"cudaEventCreate failed: {error}"
                                    cuda.bindings.runtime.cudaEventRecord(
                                        start_event, 0
                                    )
                                else:
                                    start: int = time.perf_counter_ns()
                                try:
                                    f(*inputs)
                                except Exception as e:
                                    gc.enable()
                                    raise e
                                if self._driver == "cuda":
                                    (error,) = cuda.bindings.runtime.cudaEventRecord(
                                        end_event, 0
                                    )
                                    assert (
                                        error
                                        == cuda.bindings.runtime.cudaError_t.cudaSuccess
                                    ), f"cudaEventRecord failed: {error}"
                                    (error,) = (
                                        cuda.bindings.runtime.cudaEventSynchronize(
                                            end_event
                                        )
                                    )
                                    assert (
                                        error
                                        == cuda.bindings.runtime.cudaError_t.cudaSuccess
                                    ), f"cudaEventSynchronize failed: {error}"
                                    (
                                        error,
                                        cur_time,
                                    ) = cuda.bindings.runtime.cudaEventElapsedTime(
                                        start_event, end_event
                                    )
                                    assert (
                                        error
                                        == cuda.bindings.runtime.cudaError_t.cudaSuccess
                                    ), f"cudaEventElapsedTime failed: {error}"
                                    cur_time = cur_time * 1e6
                                else:
                                    end: int = time.perf_counter_ns()
                                    cur_time: float = (end - start) * 1.0
                                gc.enable()
                                exec_time = min(exec_time, cur_time)
                            kstat[kernel_name, layouts] = exec_time
        return kstat

    @staticmethod
    def _compile_sub_modules_wrapper(args: Tuple[str, Dict[str, Any]]) -> Optional[str]:
        return KernelProfiler._compile_sub_module(*args)

    @staticmethod
    def _compile_sub_module(
        sub_mod_text: str, compile_options: Dict[str, Any]
    ) -> Optional[str]:
        try:
            return iree.compiler.compile_str(sub_mod_text, **compile_options)
        except iree.compiler.CompilerToolError:
            return None
