from __future__ import annotations

import argparse
from pathlib import Path

from iotbotnet.experiments.nbaiot_lof_per_device import (
    run_nbaiot_lof_per_device_stage_1,
    run_nbaiot_lof_per_device_stage_2,
)


DEFAULT_CONFIGS = {
    1: Path("configs/nbaiot_lof_per_device_stage_1.yaml"),
    2: Path("configs/nbaiot_lof_per_device_stage_2.yaml"),
}

STAGES = {
    1: run_nbaiot_lof_per_device_stage_1,
    2: run_nbaiot_lof_per_device_stage_2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ejecuta una etapa del experimento LOF por dispositivo sobre N-BaIoT."
    )

    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2],
        required=True,
        help="Etapa del experimento a ejecutar.",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Archivo YAML de configuración. Si no se indica, se utilizará el correspondiente a la etapa.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = args.config or DEFAULT_CONFIGS[args.stage]

    STAGES[args.stage](config)


if __name__ == "__main__":
    main()