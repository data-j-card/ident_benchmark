import argparse
from pathlib import Path

from benchid.backends import FakeBackend
from benchid.plan import Pilot
from benchid.runner import BudgetExhausted, Runner
from benchid.splits import split_by_angle
from benchid.tracking import Tracker

BACKENDS = {"falso": FakeBackend}


def tufts_names(identity: str) -> list[str]:
    return [f"{identity}/TD_RGB_E_{k}.jpg" for k in range(1, 6)] + [
        f"{identity}/TD_RGB_A_{r}_{c}.jpg" for r in range(1, 5) for c in range(9)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fases", nargs="+")
    parser.add_argument("--backend", choices=BACKENDS, required=True)
    parser.add_argument("--salidas", type=Path, default=Path("salidas"))
    parser.add_argument("--fotos", type=Path, default=Path("salidas/fotos"))
    args = parser.parse_args()

    pilot = Pilot.load()
    backend = BACKENDS[args.backend]()
    runner = Runner(
        pilot, backend, args.salidas, lambda i: split_by_angle(i, tufts_names(i)), args.fotos,
        Tracker(args.salidas, pilot.raw["usd_por_hora"]),
    )
    try:
        for name in args.fases:
            phase = pilot.phase(name)
            print(f"{name}: {len(phase.trains)} entrenamientos, {len(phase.gens)} imagenes")
            runner.run(phase)
    except BudgetExhausted as stop:
        raise SystemExit(f"stopped: {stop}")
    print(f"gasto acumulado USD {runner.spent_usd():.2f}")


if __name__ == "__main__":
    main()
