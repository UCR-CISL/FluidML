from __future__ import annotations

import cuda.bindings.runtime
import gc
import iree.compiler.dialects.flow
import iree.compiler.dialects.func
import iree.compiler.dialects.hal
import iree.compiler.dialects.util
import iree.compiler.ir
import iree.runtime
import numpy as np
import sys
import time

from typing import Any, Callable, Dict, List, Optional

from ..utils import IOStat, map_str_dtype
from .profiler import Profiler


class IOProfiler(Profiler):
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
    ) -> IOProfiler:
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

    def run(self, mod: str) -> IOStat:
        with iree.compiler.ir.Context():
            mod: iree.compiler.ir.Module = iree.compiler.ir.Module.parse(mod)
            for operation in mod.body.operations:
                if isinstance(
                    operation.opview, iree.compiler.dialects.flow.ExecutableOp
                ):
                    [block] = operation.body.blocks
                    _, builtin_mod, _ = block.operations
                    [block] = builtin_mod.body.region.blocks
                    [kernel] = block.operations
                    [block] = kernel.body.blocks
                    with kernel.location:
                        func_op: iree.compiler.dialects.func.FuncOp = (
                            iree.compiler.dialects.func.FuncOp(
                                kernel.sym_name.value,
                                kernel.type,
                                ip=iree.compiler.ir.InsertionPoint(kernel),
                            )
                        )
                        kblock: iree.compiler.ir.Block = func_op.add_entry_block()
                        iree.compiler.dialects.func.ReturnOp = (
                            iree.compiler.dialects.func.return_(
                                [],
                                ip=iree.compiler.ir.InsertionPoint(kblock),
                            )
                        )
                        self._build_benchmark(
                            kernel, iree.compiler.ir.InsertionPoint(mod.body)
                        )
                    kernel.erase()
        buffer: bytes = iree.compiler.compile_str(
            str(mod),
            **self._compile_commands,
        )
        config: iree.runtime.Config = iree.runtime.Config(self._driver)
        ctx: iree.runtime.SystemContext = iree.runtime.SystemContext(config=config)
        vm_module: iree.runtime.VmModule = iree.runtime.VmModule.copy_buffer(
            ctx.instance, buffer
        )
        ctx.add_vm_module(vm_module)
        iostat: IOStat = IOStat()
        for operation in mod.body.operations:
            if isinstance(
                operation.opview, iree.compiler.dialects.util.FuncOp
            ) and operation.sym_name.value.endswith("_benchmark"):
                kernel_name: str = operation.sym_name.value.removesuffix("_benchmark")
                [block] = operation.body.blocks
                inputs: List[np.ndarray] = [
                    np.random.rand(*op.target.type.shape).astype(
                        map_str_dtype(str(op.target.type.element_type))
                    )
                    for op in block.operations
                    if isinstance(op, iree.compiler.dialects.hal.TensorImportOp)
                ]
                f: Callable = ctx.modules.module[operation.sym_name.value]
                for _ in range(self._times // 10):
                    f(*inputs)
                exec_time: float = sys.float_info.max
                for _ in range(self._times):
                    gc.disable()
                    if self._driver == "cuda":
                        error, start_event = cuda.bindings.runtime.cudaEventCreate()
                        assert (
                            error == cuda.bindings.runtime.cudaError_t.cudaSuccess
                        ), f"cudaEventCreate failed: {error}"
                        error, end_event = cuda.bindings.runtime.cudaEventCreate()
                        assert (
                            error == cuda.bindings.runtime.cudaError_t.cudaSuccess
                        ), f"cudaEventCreate failed: {error}"
                        cuda.bindings.runtime.cudaEventRecord(start_event, 0)
                    else:
                        start: int = time.perf_counter_ns()
                    try:
                        f(*inputs)
                    except Exception as e:
                        gc.enable()
                        if self._driver == "cuda":
                            cuda.bindings.runtime.cudaEventDestroy(start_event)
                            cuda.bindings.runtime.cudaEventDestroy(end_event)
                        raise e
                    if self._driver == "cuda":
                        (error,) = cuda.bindings.runtime.cudaEventRecord(end_event, 0)
                        assert (
                            error == cuda.bindings.runtime.cudaError_t.cudaSuccess
                        ), f"cudaEventRecord failed: {error}"
                        (error,) = cuda.bindings.runtime.cudaEventSynchronize(end_event)
                        assert (
                            error == cuda.bindings.runtime.cudaError_t.cudaSuccess
                        ), f"cudaEventSynchronize failed: {error}"
                        error, cur_time = cuda.bindings.runtime.cudaEventElapsedTime(
                            start_event, end_event
                        )
                        assert (
                            error == cuda.bindings.runtime.cudaError_t.cudaSuccess
                        ), f"cudaEventElapsedTime failed: {error}"
                        cur_time = cur_time * 1e6
                    else:
                        end: int = time.perf_counter_ns()
                        cur_time: float = (end - start) * 1.0
                    gc.enable()
                    exec_time = min(exec_time, cur_time)
                iostat[kernel_name] = exec_time
        return iostat
