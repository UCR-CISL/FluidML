import iree.compiler.ir
import multiprocessing
import multiprocessing.synchronize


class Context(object):
    def __init__(self, ctx: iree.compiler.ir.Context, *args, **kwargs) -> "Context":
        super().__init__(*args, **kwargs)
        self.ctx: iree.compiler.ir.Context = ctx
        self.lock: multiprocessing.synchronize.Lock = multiprocessing.Lock()
