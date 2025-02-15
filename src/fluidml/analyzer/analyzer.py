import iree.compiler.ir

from typing import List

from .op_wrapper import OpWrapper


class Analyzer(object):
    def __init__(self, ctx: iree.compiler.ir.Context, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ctx: iree.compiler.ir.Context = ctx
        self.wrappers: List[OpWrapper] = None

    def _analyze(self):
        for wrapper in self.wrappers:
            for operand in wrapper.op.operands:
                # TODO(Jinjie Liu): Do something here.
                pass

    def run(self, ops: List[iree.compiler.ir.Operation]):
        wrappers: List[OpWrapper] = [OpWrapper.from_op(op) for op in ops]
        self.wrappers = wrappers
        self._analyze()
