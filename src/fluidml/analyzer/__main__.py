import argparse

from typing import Optional, TypeVar

from ..utils import KStat, Schedule
from .analyzer import Analyzer
from .dp import DynamicProgramAnalyzer
from .greedy import GreedyAnalyzer

AnalyzerCls = TypeVar("AnalyzerCls", bound="Analyzer")


def main():
    dispatch_table: dict[str, AnalyzerCls] = {
        "dp": DynamicProgramAnalyzer,
        "greedy": GreedyAnalyzer,
    }
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="analyzer for FluidML pipelines",
        allow_abbrev=True,
    )
    parser.add_argument(
        "filename",
        type=str,
        help="path to the IREE flow module file",
    )
    parser.add_argument(
        "--mode",
        choices=dispatch_table.keys(),
        default="dp",
        help="analyzer mode",
    )
    parser.add_argument(
        "--kstat",
        type=str,
        required=True,
        help="path to the JSON file containing kstat data",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        help="output file for analysis results",
    )
    args: argparse.Namespace = parser.parse_args()
    filename: str = args.filename
    with open(filename, "r") as f:
        mod: str = f.read()
    kstatf: str = args.kstat
    with open(kstatf, "r") as f:
        kstat: KStat = KStat.build(f)
    output: Optional[str] = args.output
    cls: AnalyzerCls = dispatch_table[args.mode]
    analyzer: Analyzer = cls()
    schedule: Schedule = analyzer.run(mod, kstat)
    with open(output, "w") as f:
        schedule.dump(f)


if __name__ == "__main__":
    main()
