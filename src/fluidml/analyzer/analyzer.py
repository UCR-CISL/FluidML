import iree.compiler.ir

from typing import List

from .op_wrapper import OpWrapper


class Analyzer(object):
    def run(self, ops: List[iree.compiler.ir.Operation]):
        wrappers: List[iree.compiler.ir.Operation] = [
            OpWrapper.from_op(op) for op in ops
        ]
