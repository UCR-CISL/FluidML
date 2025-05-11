import os

from typing import Optional, Union

from .analyzer import DynamicProgramAnalyzer
from .generator import Generator
from .profiler import KernelProfiler
from .utils.stat import KStat
from .utils.schedule import Schedule

times: int = int(os.getenv("FLUIDML_TIME", 50))
worker_num: int = int(os.getenv("FLUIDML_WORKER_NUM", os.cpu_count()))
check_period: float = float(os.getenv("FLUIDML_CHECK_PERIOD", 5.0))
profile_cache: Optional[str] = os.getenv("FLUIDML_PROFILE_CACHE", None)


def run(flow: Union[str, bytes], driver: str, **kwargs) -> str:
    if isinstance(flow, bytes):
        mod: str = flow.decode()
    elif isinstance(flow, str):
        mod: str = flow
    else:
        raise TypeError(f"Unsupported type {type(flow)} for fulidml.run")
    profiler: KernelProfiler = KernelProfiler(
        times, worker_num, check_period, driver, profile_cache, kwargs
    )
    kstat: KStat = profiler.run(mod)
    analyzer: DynamicProgramAnalyzer = DynamicProgramAnalyzer()
    schedule: Schedule = analyzer.run(mod, kstat)
    generator: Generator = Generator()
    return generator.run(mod, schedule)
