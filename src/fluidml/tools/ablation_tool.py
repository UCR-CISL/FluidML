import argparse

from ..utils import KStat, Schedule
from .ablation import Ablation


def main():
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="ablation-tool for FluidML pipelines", allow_abbrev=True
    )
    parser.add_argument("filename", type=str, help="path to the IREE flow module file")
    parser.add_argument(
        "--kstat",
        type=str,
        required=True,
        help="kstat pkl",
    )
    parser.add_argument(
        "--schedule",
        type=str,
        required=True,
        help="schedule pkl",
    )
    parser.add_argument("--output", type=str, required=True, help="output file path")
    args: argparse.Namespace = parser.parse_args()
    filename: str = args.filename
    kstat: str = args.kstat
    schedule: str = args.schedule
    output: str = args.output
    with open(filename, "r") as f:
        mod: str = f.read()
    with open(kstat, "r") as f:
        kstat: KStat = KStat.build(f)
    with open(schedule, "r") as f:
        schedule: Schedule = Schedule.build(f)
    ablation: Ablation = Ablation.build(mod, kstat, schedule)
    with open(output, "w") as f:
        ablation.dump(f)


if __name__ == "__main__":
    main()
