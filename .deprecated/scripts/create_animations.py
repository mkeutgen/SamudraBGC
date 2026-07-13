#!/usr/bin/env python3
"""
Create animations comparing ocean emulator rollouts with ground truth.

This script creates animated GIFs showing temporal evolution of ocean variables.
Variables are processed one at a time to minimize memory usage.

Usage:
    python scripts/create_animations.py --config configs/eval/jra_comparison.yaml
    python scripts/create_animations.py --config configs/eval/jra_comparison.yaml --variables temp_0 chl_0
"""

import argparse
import gc
import warnings
from pathlib import Path
from typing import Dict, List

import yaml

# Import helper functions
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "notebooks"))

from eval_helpers import (
    VARIABLES,
    load_experiments,
    create_animation,
)

warnings.filterwarnings('ignore')


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Create animations for ocean emulator comparison"
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for animations'
    )
    parser.add_argument(
        '--variables',
        type=str,
        nargs='+',
        default=None,
        help='Specific variables to animate (default: from config or all)'
    )
    parser.add_argument(
        '--start-frame',
        type=int,
        default=None,
        help='Starting frame index'
    )
    parser.add_argument(
        '--n-frames',
        type=int,
        default=None,
        help='Number of frames to create'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=None,
        help='Frames per second'
    )

    args = parser.parse_args()

    # Load config
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    # Get animation settings
    anim_config = config.get('animations', {})

    # Check if animations are enabled
    if not anim_config.get('create', False) and args.variables is None:
        print("\nWARNING: Animations are disabled in config.")
        print("To enable, set 'animations.create: true' in config")
        print("Or specify --variables on command line to override.")

        if input("\nContinue anyway? [y/N] ").lower() != 'y':
            return

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        base_dir = Path(config.get('output_dir', 'outputs/comparison'))
        output_dir = base_dir / 'animations'

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Animation parameters
    start_frame = args.start_frame or anim_config.get('start_frame', 0)
    n_frames = args.n_frames or anim_config.get('n_frames', 90)
    fps = args.fps or anim_config.get('fps', 7)

    print(f"\nAnimation settings:")
    print(f"  Start frame: {start_frame}")
    print(f"  Number of frames: {n_frames}")
    print(f"  FPS: {fps}")

    # Load data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)

    time_slice = tuple(config.get('time_slice')) if 'time_slice' in config else None

    predictions, ground_truth = load_experiments(
        config['experiments'],
        config['ground_truth_path'],
        time_slice=time_slice
    )

    # Determine which variables to animate
    if args.variables:
        var_names = args.variables
    elif 'variables' in anim_config:
        var_names = anim_config['variables']
    else:
        var_names = list(VARIABLES.keys())

    # Filter out excluded variables
    if 'exclude_variables' in config:
        excluded = set(config['exclude_variables'])
        var_names = [v for v in var_names if v not in excluded]

    # Validate variables exist
    var_names = [v for v in var_names if v in VARIABLES]

    if not var_names:
        print("\nERROR: No valid variables to animate")
        return

    print(f"\nCreating animations for {len(var_names)} variables:")
    for v in var_names:
        print(f"  - {v}")

    # Warning about memory usage
    total_frames = len(ground_truth.time)
    if n_frames > 100:
        print(f"\nWARNING: Creating {n_frames} frames may use significant memory")
        print(f"Consider reducing --n-frames for large datasets")

    if len(var_names) > 5:
        print(f"\nWARNING: Animating {len(var_names)} variables will take time")
        print(f"Consider using --variables to select specific variables")

    # Confirm before proceeding
    if n_frames > 100 or len(var_names) > 5:
        response = input("\nContinue? [y/N] ")
        if response.lower() != 'y':
            print("Cancelled.")
            return

    # Create animations one at a time to minimize memory usage
    print("\n" + "="*80)
    print("CREATING ANIMATIONS")
    print("="*80)

    successful = 0
    failed = 0

    for i, varname in enumerate(var_names, 1):
        print(f"\n[{i}/{len(var_names)}] Processing {varname}...")

        try:
            props = VARIABLES[varname]

            create_animation(
                varname, props,
                predictions, ground_truth,
                output_dir=str(output_dir),
                start_frame=start_frame,
                n_frames=n_frames,
                fps=fps
            )

            successful += 1

            # Force garbage collection after each animation
            gc.collect()

        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1

    # Summary
    print("\n" + "="*80)
    print("ANIMATION CREATION COMPLETE")
    print("="*80)
    print(f"\nSuccessful: {successful}")
    print(f"Failed: {failed}")
    print(f"\nAnimations saved to: {output_dir}")

    # List created files
    gif_files = sorted(output_dir.glob("*.gif"))
    if gif_files:
        print(f"\nCreated {len(gif_files)} animation(s):")
        for gif in gif_files:
            size_mb = gif.stat().st_size / (1024 * 1024)
            print(f"  - {gif.name} ({size_mb:.1f} MB)")


if __name__ == '__main__':
    main()
