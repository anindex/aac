#!/usr/bin/env python3
"""Generate synthetic Warcraft .npz terrain maps for testing.

Creates random cost maps with spatially coherent terrain by
interpolating a small base grid of random values to the target
resolution using bilinear interpolation.

Each map is saved as map_XXXX.npz with a "cost_map" key.

Usage:
    python scripts/generate_warcraft_data.py --output-dir data/warcraft --num-maps 50 --grid-size 12 --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def generate_smooth_cost_map(
    grid_size: int,
    rng: np.random.Generator,
    base_grid_size: int = 3,
    cost_min: float = 1.0,
    cost_max: float = 5.0,
) -> np.ndarray:
    """Generate a spatially coherent cost map via bilinear interpolation.

    Creates a small (base_grid_size x base_grid_size) grid of random values
    and interpolates to (grid_size x grid_size) for smooth terrain.

    Args:
        grid_size: Target grid resolution (H = W).
        rng: NumPy random generator for reproducibility.
        base_grid_size: Size of the base random grid (controls smoothness).
        cost_min: Minimum terrain cost.
        cost_max: Maximum terrain cost.

    Returns:
        (grid_size, grid_size) float64 cost map with values in [cost_min, cost_max].
    """
    # Generate small base grid of random values in [0, 1]
    base = rng.random((base_grid_size, base_grid_size))

    # Bilinear interpolation to target size
    # Compute coordinates in base grid space for each target cell
    y_target = np.linspace(0, base_grid_size - 1, grid_size)
    x_target = np.linspace(0, base_grid_size - 1, grid_size)

    cost_map = np.zeros((grid_size, grid_size), dtype=np.float64)

    for i, y in enumerate(y_target):
        for j, x in enumerate(x_target):
            # Bilinear interpolation
            y0 = int(np.floor(y))
            x0 = int(np.floor(x))
            y1 = min(y0 + 1, base_grid_size - 1)
            x1 = min(x0 + 1, base_grid_size - 1)

            fy = y - y0
            fx = x - x0

            val = (
                base[y0, x0] * (1 - fy) * (1 - fx)
                + base[y0, x1] * (1 - fy) * fx
                + base[y1, x0] * fy * (1 - fx)
                + base[y1, x1] * fy * fx
            )
            cost_map[i, j] = val

    # Scale from [0, 1] to [cost_min, cost_max]
    cost_map = cost_min + cost_map * (cost_max - cost_min)

    return cost_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic Warcraft terrain data")
    parser.add_argument("--output-dir", type=str, default="data/warcraft", help="Output directory")
    parser.add_argument("--num-maps", type=int, default=50, help="Number of maps to generate")
    parser.add_argument("--grid-size", type=int, default=12, help="Grid size (H = W)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    for i in range(args.num_maps):
        cost_map = generate_smooth_cost_map(args.grid_size, rng)
        npz_path = output_dir / f"map_{i:04d}.npz"
        np.savez(npz_path, cost_map=cost_map)

    print(f"Generated {args.num_maps} maps in {output_dir}")


if __name__ == "__main__":
    main()
