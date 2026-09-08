#!/usr/bin/env python3
"""Generate one DOM Pick episode from a row of an initial-condition CSV."""

from __future__ import annotations

import argparse
import ast
import csv
import glob
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dynamic-vla-root", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--row", type=int, default=0, help="Zero-based data row")
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--object-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--event-threshold", type=float, default=0.15)
    parser.add_argument("--event-warp", type=int, default=4)
    parser.add_argument("--event-source", choices=("hdr", "ldr"), default="hdr")
    return parser.parse_args()


def read_row(csv_path: Path, row_index: int) -> dict[str, str]:
    if row_index < 0:
        raise ValueError("--row must be non-negative")
    with csv_path.open(newline="", encoding="utf-8") as stream:
        for index, row in enumerate(csv.DictReader(stream)):
            if index == row_index:
                return row
    raise IndexError(f"CSV has no data row {row_index}")


def infer_category(row: dict[str, str], object_dir: Path) -> str:
    aliases = ast.literal_eval(row["objects"])
    text = " ".join(str(alias).lower() for alias in aliases)
    categories = sorted(path.name for path in object_dir.iterdir() if path.is_dir())
    matches = [category for category in categories if category.lower() in text]
    if not matches:
        raise ValueError(f"Cannot infer an object category from aliases: {aliases}")
    return max(matches, key=len)


def load_simulation_module(dynamic_vla_root: Path):
    sys.path.insert(0, str(dynamic_vla_root))
    sys.path.insert(0, str(dynamic_vla_root / "simulations"))
    module_path = dynamic_vla_root / "simulations" / "simulate.py"
    spec = importlib.util.spec_from_file_location("dom_simulate", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    args = parse_args()
    row = read_row(args.csv, args.row)
    category = infer_category(row, args.object_dir)
    seed = int(row["episode_index"]) if args.seed is None else args.seed

    # Omniverse must be launched before importing the DOM simulator.
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=True, enable_cameras=True, device=args.device)
    simulation_app = launcher.app

    try:
        sim = load_simulation_module(args.dynamic_vla_root)
        object_files = sorted(glob.glob(str(args.object_dir / category / "*.usd")))
        if not object_files:
            raise FileNotFoundError(f"No USD objects found for category {category!r}")

        # The CSV does not expose the original USD variant. Select one
        # deterministically while preserving the recorded category.
        object_file = object_files[seed % len(object_files)]
        object_size = sim._get_object_sizes(str(args.object_dir), [category])[object_file]
        relative_position = np.array(
            [float(row[f"obj_pos_{axis}"]) for axis in "xyz"], dtype=np.float64
        )
        relative_velocity = np.array(
            [float(row[f"obj_vel_{axis}"]) for axis in "xyz"], dtype=np.float64
        )
        relative_rotation = Rotation.from_euler(
            "xyz", [float(row[f"obj_rot_{axis}"]) for axis in "xyz"]
        )
        upstream_object_states = sim._get_object_states

        def fixed_object_states(
            sim_cfg, robot_pose, table_bbox, object_metadata, robot_reach_dist
        ):
            # Keep DOM's original seeded container placement. The CSV records
            # only the moving target and has no receptacle pose or asset ID.
            states = upstream_object_states(
                sim_cfg,
                robot_pose,
                table_bbox,
                object_metadata,
                robot_reach_dist,
            )
            robot_quaternion = np.asarray(robot_pose["quat"], dtype=np.float64)
            robot_rotation = Rotation.from_quat(robot_quaternion[[1, 2, 3, 0]])
            world_position = (
                np.asarray(robot_pose["pos"], dtype=np.float64)
                + robot_rotation.apply(relative_position)
            )
            world_velocity = robot_rotation.apply(relative_velocity)
            world_rotation = robot_rotation * relative_rotation
            quaternion_xyzw = world_rotation.as_quat()
            states["objects"] = [
                {
                    "file_path": object_file,
                    "size": object_size,
                    "category": category,
                    "tags": [category],
                    "pos": world_position.tolist(),
                    "quat": np.r_[quaternion_xyzw[3], quaternion_xyzw[:3]].tolist(),
                    "lin_vel": world_velocity.tolist(),
                    "friction": 1.0,
                    "mass": 0.05,
                }
            ]
            return states

        sim._get_object_states = fixed_object_states
        args.output_dir.mkdir(parents=True, exist_ok=True)
        event_output_dir = args.output_dir / "events"
        event_output_dir.mkdir(parents=True, exist_ok=True)

        run_args = SimpleNamespace(
            debug=True,
            device=args.device,
            disable_fabric=False,
            disable_sm=False,
            enable_cameras=True,
            event_camera=True,
            event_output_dir=str(event_output_dir),
            event_source=args.event_source,
            event_threshold=args.event_threshold,
            event_warp=args.event_warp,
            event_adaptive_warp=True,
            event_max_warp_factor=2,
            event_hybrid=True,
            event_hybrid_gate_gain=0.25,
            event_hybrid_support_radius=2,
            event_dynamic_gt=False,
            num_envs=1,
            path_tracing=False,
            robot="franka",
            scene_dir=str(args.scene_dir),
            object_dir=str(args.object_dir),
            output_dir=str(args.output_dir),
            task="pick",
            sim_cfg_file=str(
                args.dynamic_vla_root / "simulations" / "configs" / "sim_cfg.yaml"
            ),
            save=True,
            seed=seed,
            n_simulations=1,
        )

        sim.args = run_args
        outputs_before = set(args.output_dir.glob("*.h5"))
        sim.main(run_args)
        if not set(args.output_dir.glob("*.h5")) - outputs_before:
            raise RuntimeError(
                "DOM returned without saving an episode; inspect the simulator log"
            )
        print(
            f"[done] episode_index={row['episode_index']} category={category} "
            f"asset={Path(object_file).name} output={args.output_dir}",
            flush=True,
        )
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
