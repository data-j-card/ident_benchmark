import argparse
from pathlib import Path

from benchid.adherence import Judge, JudgeConfig, score_run
from benchid.plan import Pilot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--salidas", type=Path, default=Path("salidas"))
    args = parser.parse_args()

    cfg = JudgeConfig.load()
    scored = score_run(args.salidas, Judge(cfg), cfg, Pilot.load().scenarios)
    print(f"{scored} imagenes juzgadas en {args.salidas / 'adherencia'}")


if __name__ == "__main__":
    main()
