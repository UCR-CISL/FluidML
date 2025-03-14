import argparse

from typing import Optional

from ..utils.kstat import KStat
from ..utils.schedule import Schedule
from .analyzer import Analyzer


def main():
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
    analyzer: Analyzer = Analyzer()
    schedule: Schedule = analyzer.run(mod, kstat)
    with open(output, "wb") as f:
        schedule.dump(f)


if __name__ == "__main__":
    main()
