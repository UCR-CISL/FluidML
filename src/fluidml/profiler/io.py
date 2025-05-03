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
# TODO: can we replace torch with cuda python bindings?
import torch

from typing import Any, Callable, Dict, List, Optional

from ..utils.utils import map_str_dtype
from ..utils.kstat import KStat
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
    ) -> "IOProfiler":
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

    def run(self, mod: str) -> "KStat":
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
        for operation in mod.body.operations:
            if isinstance(
                operation.opview, iree.compiler.dialects.util.FuncOp
            ) and operation.sym_name.value.endswith("_benchmark"):
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
                        torch.cuda.synchronize()
                        start_event: torch.cuda.Event = torch.cuda.Event(
                            enable_timing=True
                        )
                        end_event: torch.cuda.Event = torch.cuda.Event(
                            enable_timing=True
                        )
                        start_event.record()
                    else:
                        start: int = time.perf_counter_ns()
                    try:
                        f(*inputs)
                    except Exception as e:
                        gc.enable()
                        raise e
                    if self._driver == "cuda":
                        end_event.record()
                        torch.cuda.synchronize()
                        cur_time: float = start_event.elapsed_time(end_event) * 1e6
                    else:
                        end: int = time.perf_counter_ns()
                        cur_time: float = (end - start) * 1.0
                    gc.enable()
                    exec_time = min(exec_time, cur_time)
        # TODO: rewrite `KStat` to support `IOProfiler`
        return KStat()
