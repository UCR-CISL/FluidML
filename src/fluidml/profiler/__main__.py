import argparse
import os
import sys

from typing import Any, Dict, Optional

from ..utils.kstat import KStat
from .profiler import Profiler


def main():
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="profiler for FluidML pipelines",
        allow_abbrev=True,
    )
    parser.add_argument(
        "filename",
        type=str,
        help="path to the IREE flow module file",
    )
    parser.add_argument(
        "--times",
        type=int,
        default=50,
        help="number of times to benchmark each kernel",
    )
    parser.add_argument(
        "--jobs",
        default=os.cpu_count(),
        type=int,
        help="number of workers to run benchmarks",
    )
    parser.add_argument(
        "--check-period",
        default=5,
        type=float,
        help="check period for worker status",
    )
    parser.add_argument(
        "--driver",
        default="local-task",
        type=str,
        help="IREE driver to use for the benchmark",
    )
    parser.add_argument(
        "--profile-cache",
        default=None,
        type=str,
        help="cache file for profiling results",
    )
    parser.add_argument(
        "--compile-options",
        default="{}",
        type=str,
        help="compile options for IREE",
    )
    parser.add_argument(
        "--output",
        default=None,
        type=str,
        help="output file for benchmark results",
    )
    args: argparse.Namespace = parser.parse_args()
    filename: str = args.filename
    with open(filename, "r") as f:
        mod: str = f.read()
    times: int = args.times
    worker_num: int = args.jobs
    check_period: float = args.check_period
    driver: str = args.driver
    profile_cache: Optional[str] = args.profile_cache
    compile_options: Dict[str, Any] = eval(args.compile_options)
    output: Optional[str] = args.output
    profiler: Profiler = Profiler(
        times, worker_num, check_period, driver, profile_cache, compile_options
    )
    result: KStat = profiler.run(mod)
    if output:
        with open(output, "wb") as f:
            result.dump(f)
    else:
        result.dump(sys.stdout)


if __name__ == "__main__":
    main()
