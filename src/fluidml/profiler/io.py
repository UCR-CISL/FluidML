from typing import Any, Dict, Optional

from ..utils.kstat import KStat

from .profiler import Profiler


class IOProfiler(Profiler):
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
    ) -> "IOProfiler":
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

    def run(self, mod: str) -> "KStat":
        return KStat()
