import iree.compiler.dialects.flow
import iree.compiler.dialects.hal
import iree.compiler.dialects.util
import iree.compiler.ir
import os

from typing import List, Union

from .utils.kstat import KStat
from .analyzer import Analyzer
from .profiler import Profiler

times: int = int(os.getenv("FLUIDML_TIME", 5))
worker_num: int = int(os.getenv("FLUIDML_WORKER_NUM", os.cpu_count()))
check_period: float = float(os.getenv("FLUIDML_CHECK_PERIOD", 5.0))


def run(flow: Union[str, bytes], **kwargs):
    if isinstance(flow, bytes):
        mod: str = flow.decode()
    elif isinstance(flow, str):
        mod: str = flow
    else:
        raise TypeError(f"Unsupported type {type(flow)} for fulidml.run")
    profiler: Profiler = Profiler(times, worker_num, check_period, kwargs)
    kstat: KStat = profiler.run(mod)
    analyzer: Analyzer = Analyzer()
    analyzer.run(mod, kstat)
