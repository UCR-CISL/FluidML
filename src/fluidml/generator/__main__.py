import argparse

from ..utils.schedule import Schedule
from .generator import Generator


def main():
    parser = argparse.ArgumentParser(
        description="generator for FluidML pipelines", allow_abbrev=True
    )
    parser.add_argument("filename", type=str, help="path to the IREE flow module file")
    parser.add_argument(
        "--schedule",
        type=str,
        required=True,
        help="schedule for generated pipeline",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="output file for generated pipeline"
    )
    args: argparse.Namespace = parser.parse_args()
    filename: str = args.filename
    schedule: str = args.schedule
    output: str = args.output
    with open(filename, "r") as f:
        mod: str = f.read()
    with open(schedule, "r") as f:
        schedule: Schedule = Schedule.build(f)
    generator: Generator = Generator()
    mod: str = generator.run(mod, schedule)
    with open(output, "w") as f:
        f.write(mod)


if __name__ == "__main__":
    main()
