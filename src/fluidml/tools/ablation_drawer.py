import argparse

from .ablation import Ablation


def main():
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="ablation-drawer for FluidML pipelines", allow_abbrev=True
    )
    parser.add_argument("filename", type=str, help="path to the ablation json file")
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        help="output format (default: png)",
    )
    parser.add_argument("--output", type=str, required=True, help="output file path")
    args: argparse.Namespace = parser.parse_args()
    filename: str = args.filename
    format: str = args.format
    output: str = args.output
    with open(filename, "r") as f:
        ablation: Ablation = Ablation.from_(f)
    with open(output, "wb") as f:
        ablation.savefig(f, format)


if __name__ == "__main__":
    main()
