import iree.compiler.dialects.flow
import iree.compiler.dialects.hal
import iree.compiler.dialects.util
import iree.compiler.ir

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union

from ..utils.stat.stat import Stat
from .util import get_signature


class Profiler(object):
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
    ) -> "Profiler":
        super().__init__(*args, **kwargs)
        self._times: int = times
        self._worker_num: int = worker_num
        self._check_period: float = check_period
        self._driver: str = driver
        self._profile_cache: Optional[str] = profile_cache
        extra_args: str = compile_options.get("extra_args", [])
        extra_args = [
            arg for arg in extra_args if not arg.startswith("--compile-from=")
        ] + ["--compile-from=flow", "--iree-llvmcpu-enable-ukernels=none"]
        compile_options["extra_args"] = extra_args
        self._compile_commands: Dict[str, Any] = compile_options

    @abstractmethod
    def run(self, mod: str) -> Stat:
        raise NotImplementedError(
            f"Profiler.run() must be implemented in {self.__class__.__name__} class"
        )

    @staticmethod
    def _build_benchmark(
        kernel: iree.compiler.ir.Operation, ip: iree.compiler.ir.InsertionPoint
    ) -> str:
        (
            kernel_name,
            mod_name,
            input_types,
            result_types,
            _,
        ) = get_signature(kernel)
        fname: str = f"{kernel_name}_benchmark"
        ftype: iree.compiler.ir.Attribute = iree.compiler.ir.Attribute.parse(
            f"({', '.join(['!hal.buffer_view' for _ in input_types])}) -> ({', '.join(['!hal.buffer_view' for _ in result_types])})"
        )
        with kernel.location:
            func_op: iree.compiler.dialects.util.FuncOp = (
                iree.compiler.dialects.util.func(
                    fname,
                    ftype,
                    sym_visibility="public",
                    loc=kernel.location,
                    ip=ip,
                )
            )
            block: iree.compiler.ir.Block = func_op.body.blocks.append(
                *(iree.compiler.ir.Type.parse("!hal.buffer_view") for _ in input_types),
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
                            source_type,
                            source,
                            source_encoding,
                            [],
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
        return fname
