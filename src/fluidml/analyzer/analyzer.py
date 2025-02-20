import iree.compiler.ir

from typing import List

from .graph import Graph


class Analyzer(object):
    def __init__(self, ctx: iree.compiler.ir.Context, *args, **kwargs) -> "Analyzer":
        super().__init__(*args, **kwargs)
        self.ctx: iree.compiler.ir.Context = ctx

    def run(self, ops: List[iree.compiler.ir.OpView]):
        graph: Graph = Graph(ops)
        graph.partitioned()
        # TODO(Jinjie Liu): Do something more here.
