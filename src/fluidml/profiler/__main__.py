import argparse
import os

from typing import Any, Dict, Optional, TypeVar

from ..utils import IOStat, KStat, Stat
from .io import IOProfiler
from .kernel import KernelProfiler
from .pipeline import PipelineProfiler
from .profiler import Profiler

ProfilerCls = TypeVar("ProfilerCls", bound=Profiler)


def main():
    dispatch_table: Dict[str, ProfilerCls] = {
        "io": IOProfiler,
        "kernel": KernelProfiler,
        "pipeline": PipelineProfiler,
    }
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="profiler for FluidML pipelines",
        allow_abbrev=True,
    )
    subparsers: argparse._SubParsersAction = parser.add_subparsers(
        title="subcommand",
        required=True,
        dest="subcommand",
    )
    profiler: argparse.ArgumentParser = subparsers.add_parser(
        "profile",
        help="profile a FluidML pipeline",
        allow_abbrev=True,
    )
    reducer: argparse.ArgumentParser = subparsers.add_parser(
        "reduce",
        help="reduce a FluidML pipeline",
        allow_abbrev=True,
    )
    profiler.add_argument(
        "filename",
        type=str,
        help="path to the IREE flow module file",
    )
    profiler.add_argument(
        "--times",
        type=int,
        default=50,
        help="number of times to benchmark each kernel",
    )
    profiler.add_argument(
        "--jobs",
        default=os.cpu_count(),
        type=int,
        help="number of workers to run benchmarks",
    )
    profiler.add_argument(
        "--check-period",
        default=5,
        type=float,
        help="check period for worker status",
    )
    profiler.add_argument(
        "--driver",
        default="local-task",
        type=str,
        help="IREE driver to use for the benchmark",
    )
    profiler.add_argument(
        "--profile-cache",
        default=None,
        type=str,
        help="cache file for profiling results",
    )
    profiler.add_argument(
        "--compile-options",
        default="{}",
        type=str,
        help="compile options for IREE",
    )
    profiler.add_argument(
        "--mode",
        type=str,
        choices=dispatch_table.keys(),
        default="pipeline",
        help="mode for profiler",
    )
    profiler.add_argument(
        "--output",
        default=True,
        type=str,
        help="output file for benchmark results",
    )
    reducer.add_argument(
        "--iostat",
        type=str,
        required=True,
        help="path to the JSON file containing IOStat data",
    )
    reducer.add_argument(
        "--kstat",
        type=str,
        required=True,
        help="path to the JSON file containing KStat data",
    )
    reducer.add_argument(
        "--output",
        type=str,
        required=True,
        help="output file for analysis results",
    )
    args: argparse.Namespace = parser.parse_args()
    if args.subcommand == "profile":
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
        stat: Stat = profiler.run(mod)
        with open(output, "w") as f:
            stat.dump(f)
    elif args.subcommand == "reduce":
        iostatf: str = args.iostat
        with open(iostatf, "r") as f:
            iostat: IOStat = IOStat.build(f)
        kstatf: str = args.kstat
        with open(kstatf, "r") as f:
            kstat: KStat = KStat.build(f)
        output: str = args.output
        rstat: KStat = kstat.reduce(iostat)
        with open(output, "w") as f:
            rstat.dump(f)


if __name__ == "__main__":
    main()
