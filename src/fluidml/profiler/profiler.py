import iree.compiler.ir
import iree.compiler.dialects.flow
import iree.compiler.dialects.func
import iree.compiler.dialects.hal
import iree.compiler.dialects.util
import iree.runtime

from typing import Any, Dict, List, Optional, Tuple

from ..utils.kstat import KStat
from .work import Master


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
        extra_args: str = compile_options.get("extra_args", [])
        extra_args = [
            arg for arg in extra_args if not arg.startswith("--compile-from=")
        ] + ["--compile-from=flow", "--iree-llvmcpu-disable-distribution"]
        compile_options["extra_args"] = extra_args
        self._master: Master = Master(
            times, worker_num, check_period, profile_cache, compile_options
        )

    def run(self, mod: str) -> KStat:
        with iree.compiler.ir.Context():
            mod: iree.compiler.ir.Module = iree.compiler.ir.Module.parse(mod)
            sub_mods: List[iree.compiler.ir.Module] = []
            for operation in mod.body.operations:
                if isinstance(operation.opview, iree.compiler.dialects.util.GlobalOp):
                    global_op: iree.compiler.dialects.util.GlobalOp = operation.opview
                if isinstance(
                    operation.opview, iree.compiler.dialects.flow.ExecutableOp
                ):
                    sub_mod: iree.compiler.ir.Module = iree.compiler.ir.Module.create(
                        mod.operation.location
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
                    sub_mods += [str(sub_mod)]
            table: List[Tuple[str, Tuple[Tuple[int, ...]], float]] = self._master.run(
                sub_mods
            )
            kstat: KStat = KStat()
            for element in table:
                kernel, axes, exec_time = element
                kstat[kernel, axes] = exec_time
            return kstat
