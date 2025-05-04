import argparse
import os
import sys

from typing import Any, Dict, Optional, TypeVar

from ..utils.stat.kstat import KStat
from .io import IOProfiler
from .kernel import KernelProfiler
from .profiler import Profiler

ProfilerCls = TypeVar("ProfilerCls", bound="Profiler")


def main():
    dispatch_table: Dict[str, Profiler] = {
        "kernel": KernelProfiler,
        "io": IOProfiler,
    }
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
    parser.add_argument(
        "--mode",
        type=str,
        choices=dispatch_table.keys(),
        required=True,
        help="mode for profiler",
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
    cls: ProfilerCls = dispatch_table[args.mode]
    profiler: ProfilerCls = cls(
        times, worker_num, check_period, driver, profile_cache, compile_options
    )
    result: KStat = profiler.run(mod)
    if output:
        with open(output, "w") as f:
            result.dump(f)
    else:
        result.dump(sys.stdout)


if __name__ == "__main__":
    main()
