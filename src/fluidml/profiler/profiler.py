import iree.compiler.ir
import iree.compiler.dialects.flow
import iree.compiler.dialects.func
import iree.compiler.dialects.hal
import iree.compiler.dialects.util
import iree.runtime

from collections import defaultdict
from typing import Any, Dict, List, Tuple

from .work import Master


class Profiler(object):
    def __init__(
        self,
        ctx: iree.compiler.ir.Context,
        times: int,
        worker_num: int,
        compile_options: Dict[str, Any],
        *args,
        **kwargs,
    ) -> "Profiler":
        super().__init__(*args, **kwargs)
        self._ctx: iree.compiler.ir.Context = ctx
        self._master: Master = Master(times, worker_num, compile_options)

    def run(
        self, mod: iree.compiler.ir.Module
    ) -> Dict[str, Dict[Tuple[int, ...], float]]:
        sub_mods: List[iree.compiler.ir.Module] = []
        for operation in mod.body.operations:
            if isinstance(operation.opview, iree.compiler.dialects.util.GlobalOp):
                global_op: iree.compiler.dialects.util.GlobalOp = operation.opview
            if isinstance(operation.opview, iree.compiler.dialects.flow.ExecutableOp):
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
        results: List[Tuple[str, Tuple[Tuple[int, ...]], float]] = self._master.run(
            sub_mods
        )
        bench_map: Dict[str, Dict[Tuple[int, ...], float]] = defaultdict(dict)
        for result in results:
            kernel, axes, exec_time = result
            bench_map[kernel][axes] = exec_time
        return bench_map
