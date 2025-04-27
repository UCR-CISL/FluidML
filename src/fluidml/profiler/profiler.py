from abc import abstractmethod
from typing import Any, Dict, Optional

from ..utils.kstat import KStat


class Profiler(object):
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
    ) -> "Profiler":
        super().__init__(*args, **kwargs)
        self._times: int = times
        self._worker_num: int = worker_num
        self._check_period: float = check_period
        self._driver: str = driver
        self._profile_cache: Optional[str] = profile_cache
        extra_args: str = compile_options.get("extra_args", [])
        extra_args = [
            arg for arg in extra_args if not arg.startswith("--compile-from=")
        ] + ["--compile-from=flow", "--iree-llvmcpu-enable-ukernels=none"]
        compile_options["extra_args"] = extra_args
        self._compile_commands: Dict[str, Any] = compile_options

    @abstractmethod
    def run(self, mod: str) -> KStat:
        raise NotImplementedError(
            f"Profiler.run() must be implemented in {self.__class__.__name__} class"
        )
