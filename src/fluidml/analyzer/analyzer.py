from __future__ import annotations

import iree.compiler.dialects.util
import iree.compiler.ir


from ..utils.stat import KStat
from ..utils.schedule import Schedule


from abc import abstractmethod
from typing import List


class Analyzer(object):
    def __init__(self, *args, **kwargs) -> Analyzer:
        super().__init__(*args, **kwargs)

    @abstractmethod
    def run(self, mod: str, kstat: KStat) -> Schedule:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement run() method."
        )

    @staticmethod
    def _filter_func_ops(
        operations: List[iree.compiler.ir.Operation],
    ) -> iree.compiler.dialects.util.FuncOp:
        func_ops: List[iree.compiler.dialects.util.FuncOp] = list(
            filter(
                lambda op: isinstance(op, iree.compiler.dialects.util.FuncOp),
                operations,
            )
        )
        if len(func_ops) == 1:
            [func_op] = func_ops
        elif len(func_ops) == 2:
            [func_op] = list(
                filter(lambda op: op.sym_name.value.endswith("$async"), func_ops)
            )
        else:
            raise NotImplementedError(f"Unsupported number of FuncOps: {func_ops}")
        return func_op
