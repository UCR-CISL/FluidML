from __future__ import annotations

from ..utils import IOStat, KStat
from .io import IOProfiler
from .kernel import KernelProfiler
from .profiler import Profiler

from typing import Any, Dict, Optional


class PipelineProfiler(Profiler):
    def __init__(
        self,
        times: int,
        worker_num: int,
        check_period: float,
        driver: str,
        profile_cache: Optional[str],
        compile_options: Dict[str, Any],
        *args,
        **kwargs,
    ) -> PipelineProfiler:
        super().__init__(
            times,
            worker_num,
            check_period,
            driver,
            profile_cache,
            compile_options,
            *args,
            **kwargs,
        )
        self._io_profiler: IOProfiler = IOProfiler(
            times,
            worker_num,
            check_period,
            driver,
            profile_cache,
            compile_options,
            *args,
            **kwargs,
        )
        self._kernel_profiler: KernelProfiler = KernelProfiler(
            times,
            worker_num,
            check_period,
            driver,
            profile_cache,
            compile_options,
            *args,
            **kwargs,
        )

    def run(self, mod: str) -> KStat:
        io_stat: IOStat = self._io_profiler.run(mod)
        kernel_stat: KStat = self._kernel_profiler.run(mod)
        return kernel_stat.reduce(io_stat)
