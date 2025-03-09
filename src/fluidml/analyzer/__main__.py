import argparse

from typing import Optional

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
        "--json",
        type=str,
        required=True,
        help="path to the JSON file containing profiling data",
    )
    parser.add_argument(
        "--output",
        default=None,
        type=str,
        help="output file for analysis results",
    )
    args: argparse.Namespace = parser.parse_args()
    filename: str = args.filename
    with open(filename, "r") as f:
        mod: str = f.read()
    json_file: str = args.json
    output: Optional[str] = args.output
    analyzer: Analyzer = Analyzer()


if __name__ == "__main__":
    main()
