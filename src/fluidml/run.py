import os

from typing import Union

from .analyzer import Analyzer
from .generator import Generator
from .profiler import Profiler
from .utils.kstat import KStat
from .utils.schedule import Schedule

times: int = int(os.getenv("FLUIDML_TIME", 1000))
worker_num: int = int(os.getenv("FLUIDML_WORKER_NUM", os.cpu_count()))
check_period: float = float(os.getenv("FLUIDML_CHECK_PERIOD", 5.0))


def run(flow: Union[str, bytes], **kwargs) -> str:
    if isinstance(flow, bytes):
        mod: str = flow.decode()
    elif isinstance(flow, str):
        mod: str = flow
    else:
        raise TypeError(f"Unsupported type {type(flow)} for fulidml.run")
    profiler: Profiler = Profiler(times, worker_num, check_period, kwargs)
    kstat: KStat = profiler.run(mod)
    analyzer: Analyzer = Analyzer()
    schedule: Schedule = analyzer.run(mod, kstat)
    generator: Generator = Generator()
    return generator.run(mod, schedule)
