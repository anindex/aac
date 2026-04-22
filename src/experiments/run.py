"""Hydra-based experiment runner entry point."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Main experiment entry point with Hydra configuration.

    Loads config, sets seed for reproducibility, and dispatches
    to the appropriate experiment runner. Runner dispatch will be
    added in Plan 03.

    Args:
        cfg: Hydra-composed configuration.
    """
    from experiments.runners import get_runner
    from experiments.utils import seed_everything

    seed_everything(cfg.seed)
    runner_cls = get_runner(cfg.track.name)
    runner = runner_cls(cfg)
    try:
        runner.run(cfg)
    finally:
        runner.close()


if __name__ == "__main__":
    main()
