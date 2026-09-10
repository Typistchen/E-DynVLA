#!/usr/bin/env python3
"""Generate one reproducible DOM Pick episode with paired v4-hybrid events."""

from __future__ import annotations

import argparse
import ast
import csv
import glob
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation


DOM_TARGET_CATEGORIES = (
    "apple",
    "avocado",
    "beer",
    "bottle",
    "can",
    "cup",
    "egg",
    "kiwi",
    "lemon",
    "lime",
    "onion",
    "orange",
    "peach",
    "potato",
    "tangerine",
    "tomato",
)
DOM_SPEED_BINS = (
    (0.15, 0.30),
    (0.30, 0.45),
    (0.45, 0.60),
    (0.60, 0.75),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dynamic-vla-root", type=Path, required=True)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--row", type=int, default=0, help="Zero-based data row")
    parser.add_argument("--sample-index", type=int)
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--object-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--event-threshold", type=float, default=0.15)
    parser.add_argument("--event-warp", type=int, default=4)
    parser.add_argument("--event-source", choices=("hdr", "ldr"), default="hdr")
    parser.add_argument(
        "--random-safe-init",
        action="store_true",
        help="Ignore CSV pose/velocity fields and sample a reproducible safe initial state",
    )
    parser.add_argument(
        "--random-dom-init",
        action="store_true",
        help="Use DOM-compatible category-balanced and speed-stratified sampling",
    )
    parser.add_argument(
        "--fixed-object-asset",
        help="Fixed USD filename (for example apple01.usd) used with random-safe init",
    )
    parser.add_argument("--random-speed-min", type=float, default=0.20)
    parser.add_argument("--random-speed-max", type=float, default=0.35)
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
    if args.random_dom_init and args.random_safe_init:
        raise ValueError("Select only one random initialization mode")

    row = None
    dom_speed_range = None
    if args.random_dom_init:
        if args.sample_index is None or args.sample_index < 0:
            raise ValueError("--random-dom-init requires --sample-index >= 0")
        seed = args.sample_index if args.seed is None else args.seed
        category = DOM_TARGET_CATEGORIES[args.sample_index % len(DOM_TARGET_CATEGORIES)]
        object_files = sorted(args.object_dir.glob(f"{category}/*.usd"))
        if not object_files:
            raise FileNotFoundError(f"No USD objects found for category {category!r}")
        variant_index = (args.sample_index // len(DOM_TARGET_CATEGORIES)) % len(
            object_files
        )
        fixed_object_file = object_files[variant_index]
        speed_bin_index = args.sample_index % len(DOM_SPEED_BINS)
        dom_speed_range = DOM_SPEED_BINS[speed_bin_index]
        initial_condition_source = "dom_stratified_random"
        generation_manifest = {
            "initial_condition_source": initial_condition_source,
            "sample_index": args.sample_index,
            "seed": seed,
            "fixed_object_asset": fixed_object_file.name,
            "object_category": category,
            "category_index": args.sample_index % len(DOM_TARGET_CATEGORIES),
            "asset_variant_index": variant_index,
            "speed_bin_index": speed_bin_index,
            "speed_range_mps": list(dom_speed_range),
            "sampler": "official_dom_geometry_with_balanced_category_and_speed_bins",
        }
    else:
        if args.csv is None:
            raise ValueError("Legacy initialization requires --csv")
        row = read_row(args.csv, args.row)
        seed = int(row["episode_index"]) if args.seed is None else args.seed

    if not args.random_dom_init and args.fixed_object_asset:
        matches = sorted(args.object_dir.rglob(args.fixed_object_asset))
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one asset named {args.fixed_object_asset!r}, found {matches}"
            )
        fixed_object_file = matches[0]
        category = fixed_object_file.parent.name
    elif not args.random_dom_init:
        fixed_object_file = None
        category = infer_category(row, args.object_dir)

    if args.random_safe_init:
        if args.random_speed_min <= 0 or args.random_speed_max < args.random_speed_min:
            raise ValueError("Invalid random speed range")
        rng = np.random.default_rng(seed)
        relative_position = np.array(
            [rng.uniform(0.28, 0.42), rng.uniform(-0.16, 0.16), 0.04],
            dtype=np.float64,
        )
        table_center = np.array([0.35, 0.0], dtype=np.float64)
        center_angle = np.arctan2(
            table_center[1] - relative_position[1],
            table_center[0] - relative_position[0],
        )
        motion_angle = center_angle + rng.uniform(-np.pi / 8, np.pi / 8)
        speed = rng.uniform(args.random_speed_min, args.random_speed_max)
        relative_velocity = np.array(
            [speed * np.cos(motion_angle), speed * np.sin(motion_angle), 0.0],
            dtype=np.float64,
        )
        relative_rotation = Rotation.from_euler(
            "xyz",
            [rng.choice([np.pi / 2, 3 * np.pi / 2]), 0.0, rng.uniform(0, 2 * np.pi)],
        )
        initial_condition_source = "safe_random"
        generation_manifest = {}
    elif not args.random_dom_init:
        relative_position = np.array(
            [float(row[f"obj_pos_{axis}"]) for axis in "xyz"], dtype=np.float64
        )
        relative_velocity = np.array(
            [float(row[f"obj_vel_{axis}"]) for axis in "xyz"], dtype=np.float64
        )
        relative_rotation = Rotation.from_euler(
            "xyz", [float(row[f"obj_rot_{axis}"]) for axis in "xyz"]
        )
        initial_condition_source = "csv"
        generation_manifest = {}

    # Omniverse must be launched before importing the DOM simulator.
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(
        headless=True,
        enable_cameras=True,
        device=args.device,
        distributed=os.getenv("EDV_ISOLATE_GPU") == "1",
        multi_gpu=False,
        kit_args="--/renderer/multiGpu/enabled=false",
    )
    simulation_app = launcher.app

    try:
        sim = load_simulation_module(args.dynamic_vla_root)
        object_files = sorted(glob.glob(str(args.object_dir / category / "*.usd")))
        if not object_files:
            raise FileNotFoundError(f"No USD objects found for category {category!r}")

        # The CSV does not expose the original USD variant. Select one
        # deterministically unless the caller fixes the object asset.
        object_file = (
            str(fixed_object_file)
            if fixed_object_file is not None
            else object_files[seed % len(object_files)]
        )
        object_size = sim._get_object_sizes(str(args.object_dir), [category])[object_file]
        upstream_object_states = sim._get_object_states
        args.output_dir.mkdir(parents=True, exist_ok=True)
        generation_manifest_path = args.output_dir / "generation_manifest.json"

        def persist_generation_manifest() -> None:
            with generation_manifest_path.open("w", encoding="utf-8") as stream:
                json.dump(
                    generation_manifest,
                    stream,
                    ensure_ascii=False,
                    indent=2,
                )

        def fixed_object_states(
            sim_cfg, robot_pose, table_bbox, object_metadata, robot_reach_dist
        ):
            if args.random_dom_init:
                object_cfg = sim_cfg["scene"]["objects"]
                original_speed_range = object_cfg.get("moving_speed")
                object_cfg["moving_speed"] = list(dom_speed_range)
                try:
                    states = upstream_object_states(
                        sim_cfg,
                        robot_pose,
                        table_bbox,
                        object_metadata,
                        robot_reach_dist,
                    )
                finally:
                    object_cfg["moving_speed"] = original_speed_range

                target = states["objects"][0]
                target_position = np.asarray(target["pos"], dtype=np.float64)
                target_position[2] = sim._get_object_z(table_bbox.max[2], object_size)
                selected_metadata = object_metadata.get(Path(object_file).name, {})
                target.update(
                    {
                        "file_path": object_file,
                        "size": object_size,
                        "category": category,
                        "tags": selected_metadata.get("tags", [category]),
                        "pos": target_position.tolist(),
                    }
                )
                generation_manifest.update(
                    {
                        "world_position_xyz": target_position.tolist(),
                        "world_quaternion_wxyz": list(target["quat"]),
                        "world_velocity_xyz": list(target["lin_vel"]),
                        "sampled_speed_mps": float(
                            np.linalg.norm(np.asarray(target["lin_vel"]))
                        ),
                        "friction": float(target["friction"]),
                        "mass": float(target["mass"]),
                        "object_size_xyz": np.asarray(object_size).tolist(),
                        "table_bbox_min_xyz": np.asarray(table_bbox.min).tolist(),
                        "table_bbox_max_xyz": np.asarray(table_bbox.max).tolist(),
                        "robot_position_xyz": np.asarray(robot_pose["pos"]).tolist(),
                        "robot_quaternion_wxyz": np.asarray(robot_pose["quat"]).tolist(),
                    }
                )
                persist_generation_manifest()
                return states

            # Legacy CSV/safe mode: keep DOM's seeded container placement while
            # replacing the moving target with the requested initial condition.
            states = upstream_object_states(
                sim_cfg, robot_pose, table_bbox, object_metadata, robot_reach_dist
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
        if not args.random_dom_init:
            generation_manifest.update(
                {
                    "initial_condition_source": initial_condition_source,
                    "seed": seed,
                    "fixed_object_asset": Path(object_file).name,
                    "object_category": category,
                    "relative_position_xyz": relative_position.tolist(),
                    "relative_rotation_xyz": relative_rotation.as_euler("xyz").tolist(),
                    "relative_velocity_xyz": relative_velocity.tolist(),
                    "random_speed_range": (
                        [args.random_speed_min, args.random_speed_max]
                        if args.random_safe_init
                        else None
                    ),
                }
            )
        persist_generation_manifest()
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
            event_dynamic_gt=True,
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
        persist_generation_manifest()
        source_episode = row["episode_index"] if row is not None else "none"
        print(
            f"[done] source_episode_index={source_episode} seed={seed} "
            f"category={category} asset={Path(object_file).name} "
            f"output={args.output_dir}",
            flush=True,
        )
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
