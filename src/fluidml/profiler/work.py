import multiprocessing.process
import iree.compiler
import iree.compiler.ir
import iree.compiler.dialects.func
import iree.compiler.dialects.util
import iree.runtime
import multiprocessing
import multiprocessing.connection
import multiprocessing.context
import multiprocessing.spawn
import numpy as np
import os
import time

from itertools import product
from typing import Any, Callable, Dict, Generator, List, Tuple, Union

from ..utils.utils import permute_shape, map_str_dtype
from .job import CreateSubModJob, BenchSubModJob, Job, ResultJob, JobPool
from .rwlock import RWLock
from .util import get_signature

is_debug: bool = os.getenv("FLUIDML_DEBUG", "0") == "1"


class Master(object):
    def __init__(
        self,
        times: int,
        worker_num: int,
        check_period: float,
        compile_options: Dict[str, Any],
        *args,
        **kwargs,
    ) -> "Master":
        super().__init__(*args, **kwargs)
        self._mp_context: multiprocessing.context.SpawnContext = (
            multiprocessing.get_context("spawn")
        )
        self._rwlock: RWLock = RWLock(self._mp_context)
        self._job_pool: JobPool = JobPool(self._mp_context)
        self._check_period: float = check_period
        self._workers: List[Worker] = [
            Worker(
                self._mp_context, self._rwlock, self._job_pool, times, compile_options
            )
            for _ in range(worker_num)
        ]

    def run(
        self, sub_mods: List[str]
    ) -> List[Tuple[str, Tuple[Tuple[int, ...]], float]]:
        for sub_mod in sub_mods:
            self._job_pool.put(CreateSubModJob(sub_mod))
        return self._job_pool.wait(self._check_period)


class Worker(object):
    def __init__(
        self,
        mp_context: multiprocessing.context.SpawnContext,
        rwlock: RWLock,
        job_pool: JobPool,
        times: int,
        compile_options: Dict[str, Any],
        *args,
        **kwargs,
    ) -> "Worker":
        super().__init__(*args, **kwargs)
        self._mp_context: multiprocessing.context.SpawnContext = mp_context
        self._rwlock: RWLock = rwlock
        self._job_pool: JobPool = job_pool
        self._times: int = times
        self._compile_options: Dict[str, Any] = compile_options
        self._worker = self._mp_context.Process(
            target=Worker.run,
            args=(self._rwlock, self._job_pool, times, self._compile_options),
            daemon=True,
        )
        self._worker.start()

    def __del__(self) -> None:
        self._worker.terminate()
        self._worker.join()

    @staticmethod
    def run(
        rwlock: RWLock, job_pool: JobPool, times: int, compile_commands: Dict[str, Any]
    ) -> None:
        try:
            while True:
                job: Job = job_pool.get()
                if isinstance(job, CreateSubModJob):
                    for mod in Worker._work4create(job._mod):
                        job_pool.put(BenchSubModJob(mod))
                    job_pool.free()
                elif isinstance(job, BenchSubModJob):
                    try:
                        kernel, axes, exec_time = Worker._work4bench(
                            rwlock, job.mod, times, compile_commands
                        )
                        job_pool.put(ResultJob(kernel, axes, exec_time))
                    except iree.compiler.tools.binaries.CompilerToolError as e:
                        pass
                    job_pool.free()
                else:
                    raise NotImplementedError(
                        f"Unsupported job {job} of type {type(job)}."
                    )
        except Exception as e:
            job_pool.throw(str(e))

    @staticmethod
    def _work4create(sub_mod: str) -> Generator[str, None, None]:
        with iree.compiler.ir.Context(), iree.compiler.ir.Location.unknown():
            sub_mod: iree.compiler.ir.Module = iree.compiler.ir.Module.parse(sub_mod)
            [_, operation] = sub_mod.body.operations
            [block] = operation.opview.body.blocks
            _, builtin_mod, _ = block.operations
            [block] = builtin_mod.body.region.blocks
            [kernel] = block.operations
            assert isinstance(kernel, iree.compiler.dialects.func.FuncOp)
            kernel_name, mod_name, input_types, result_types, _ = get_signature(kernel)
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
                *(iree.compiler.ir.Type.parse("!hal.buffer_view") for _ in input_types)
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
                        target: iree.compiler.ir.Type = iree.compiler.ir.Type.parse(
                            "!hal.buffer_view"
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
                for combination in product(*combinations):
                    for idx, layout in enumerate(combination):
                        kernel.attributes[
                            f"fluidml.{idx}"
                        ] = iree.compiler.ir.Attribute.parse(
                            f"array<i64: {', '.join([str(dim) for dim in layout])}>"
                        )
                    yield str(sub_mod)

    @staticmethod
    def _work4bench(
        rwlock: RWLock, sub_mod: str, times: int, compile_commands: Dict[str, Any]
    ) -> Tuple[str, Tuple[Tuple[int, ...]], float]:
        with iree.compiler.ir.Context():
            mod: iree.compiler.ir.Module = iree.compiler.ir.Module.parse(sub_mod)
            [_, operation, invoke] = mod.body.operations
            [block] = operation.opview.body.blocks
            _, builtin_mod, _ = block.operations
            [block] = builtin_mod.body.region.blocks
            [kernel] = block.operations
            assert isinstance(kernel.opview, iree.compiler.dialects.func.FuncOp)
            _, _, input_types, _, axes = get_signature(kernel)
            assert isinstance(invoke.opview, iree.compiler.dialects.util.FuncOp)
            entry: str = invoke.sym_name.value
            if is_debug:
                exec_time: float = 0.0
            else:
                with rwlock.read():
                    compiled_flatbuffer: bytes = iree.compiler.compile_str(
                        sub_mod, **compile_commands
                    )
                config: iree.runtime.Config = iree.runtime.Config("local-task")
                ctx: iree.runtime.SystemContext = iree.runtime.SystemContext(
                    config=config
                )
                vm_module: iree.runtime.VmModule = iree.runtime.VmModule.copy_buffer(
                    ctx.instance, compiled_flatbuffer
                )
                ctx.add_vm_module(vm_module)
                inputs: Tuple[np.ndarray] = [
                    np.random.rand(*(map(int, input_shape))).astype(
                        map_str_dtype(dtype)
                    )
                    for input_shape, dtype in input_types
                ]
                f: Callable = ctx.modules.module[entry]
                with rwlock.write():
                    start: int = time.perf_counter_ns()
                    for _ in range(times):
                        f(*inputs)
                    end: int = time.perf_counter_ns()
                exec_time: float = (end - start) / times
            return kernel.sym_name.value, axes, exec_time
