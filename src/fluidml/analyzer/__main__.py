import argparse

from typing import Optional

from ..utils.kstat import KStat
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
        "--entry",
        default="main",
        type=str,
        help="entry function name",
    )
    parser.add_argument(
        "--kstat",
        type=str,
        required=True,
        help="path to the JSON file containing kstat data",
    )
    parser.add_argument(
        "--output",
        default=None,
        type=str,
        help="output file for analysis results",
    )
    args: argparse.Namespace = parser.parse_args()
    filename: str = args.filename
    entry: str = args.entry
    with open(filename, "r") as f:
        mod: str = f.read()
    kstatf: str = args.kstat
    with open(kstatf, "r") as f:
        kstat: KStat = KStat.build(f)
    output: Optional[str] = args.output
    analyzer: Analyzer = Analyzer()
    analyzer.run(mod, entry, kstat)


if __name__ == "__main__":
    main()
