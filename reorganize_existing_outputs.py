#!/usr/bin/env python3
"""
Reorganize existing trained model outputs to match new naming scheme.

This script:
1. Maps old experiment names to new clean names
2. Renames output directories
3. Creates symlinks for backward compatibility (optional)
4. Generates a mapping file for reference
"""

import os
import shutil
from pathlib import Path
import json
from typing import Dict, Optional

# UPDATE THIS PATH
OUTPUTS_DIR = Path("/scratch/cimes/maximek/INMOS/Ocean_Emulator/outputs")

# Mapping from old experiment names to new names
EXPERIMENT_MAPPING = {
    # Baseline experiments
    "mom6_cobalt_bgc_clim_baseline": "baseline_mse",
    "mom6_cobalt_bgc_clim_baseline_correct_date": "baseline_mse_corrected",
    "mom6_cobalt_bgc_clim_mae": "baseline_mae",
    
    # Helmholtz 270x180 experiments
    "mom6_cobalt_bgc_clim_mae_grad_w01_helmholtz_l40s": "helmholtz270_mae_grad_w01",
    "mom6_cobalt_bgc_clim_mae_grad_w025_helmholtz_l40": "helmholtz270_mae_grad_w025",
    "mom6_cobalt_bgc_clim_mae_gradient": "helmholtz270_mae_grad_60ep",
    
    # Helmholtz full domain experiments
    "mom6_cobalt_bgc_clim_mae_grad_w01_helmholtz_full": "helmholtzfull_mae_grad_w01_25lev",
    "mom6_cobalt_bgc_clim_mae_grad_w025_helmholtz_full": "helmholtzfull_mae_grad_w025_25lev",
    
    # Evaluation directories (old naming)
    "mom6_cobalt_bgc_clim_baseline_eval": "baseline_mse_eval",
    "mom6_cobalt_bgc_clim_baseline_eval_long_rollout": "baseline_mse_eval_long",
    "mom6_cobalt_bgc_clim_mae_eval": "baseline_mae_eval",
    "mom6_cobalt_bgc_clim_mae_grad_eval": "helmholtz270_mae_grad_eval",
    "mom6_cobalt_bgc_clim_mae_grad_w01": "baseline_mae_grad_w01",  # Check if this is train or eval
    "mom6_cobalt_bgc_clim_mae_grad_w025": "baseline_mae_grad_w025",
    "exp1a_eval": "baseline_mae_grad_w01_eval",
    "exp1b_eval": "baseline_mae_grad_w025_eval",
    
    # Debug/temporary directories - mark for deletion
    "mom6_cobalt_bgc_clim_baselineDEBUGtobeDeleted": "DEBUG_DELETE",
    "mom6_cobalt_bgc_clim_mae_eval_debug": "DEBUG_DELETE",
    "mom6_cobalt_bgc_clim_mae_eval_debug2": "DEBUG_DELETE",
    "mom6_cobalt_bgc_clim_baseline(missed_u_and_v)": "DEBUG_DELETE",
    "mom6_cobalt_bgc_clim_baseline_eval(missed_u_and_v)": "DEBUG_DELETE",
}


def check_directory_contents(dir_path: Path) -> Dict[str, bool]:
    """Check what's in a directory to determine if it's train or eval output."""
    info = {
        "has_checkpoints": False,
        "has_predictions": False,
        "has_config": False,
        "checkpoint_count": 0,
    }
    
    if not dir_path.exists():
        return info
    
    # Check for checkpoints directory
    ckpt_dir = dir_path / "checkpoints"
    if ckpt_dir.exists():
        info["has_checkpoints"] = True
        ckpts = list(ckpt_dir.glob("*.pt"))
        info["checkpoint_count"] = len(ckpts)
    
    # Check for predictions.zarr (evaluation output)
    if (dir_path / "predictions.zarr").exists():
        info["has_predictions"] = True
    
    # Check for config.yaml
    if (dir_path / "config.yaml").exists():
        info["has_config"] = True
    
    return info


def generate_new_name(old_name: str, dir_info: Dict[str, bool]) -> Optional[str]:
    """Generate new name based on directory contents if not in mapping."""
    # If already in mapping, use that
    if old_name in EXPERIMENT_MAPPING:
        return EXPERIMENT_MAPPING[old_name]
    
    # Try to infer from name patterns
    if "_eval" in old_name and not old_name.endswith("_eval"):
        # Complex eval name, keep for manual review
        return f"REVIEW_{old_name}"
    
    return None


def reorganize_outputs(outputs_dir: Path, dry_run: bool = True, create_symlinks: bool = True):
    """Reorganize outputs directory."""
    
    if not outputs_dir.exists():
        print(f"Error: Outputs directory not found: {outputs_dir}")
        return
    
    print("=" * 80)
    print(f"Reorganizing outputs in: {outputs_dir}")
    print(f"Mode: {'DRY RUN (no changes)' if dry_run else 'LIVE (will rename)'}")
    print("=" * 80)
    print()
    
    # Collect all directories
    subdirs = [d for d in outputs_dir.iterdir() if d.is_dir()]
    
    # Analyze each directory
    rename_plan = []
    delete_plan = []
    review_needed = []
    
    for old_dir in sorted(subdirs):
        old_name = old_dir.name
        dir_info = check_directory_contents(old_dir)
        
        # Get new name
        new_name = generate_new_name(old_name, dir_info)
        
        if new_name is None:
            review_needed.append((old_name, dir_info))
            continue
        
        if new_name == "DEBUG_DELETE":
            delete_plan.append((old_name, dir_info))
            continue
        
        if new_name.startswith("REVIEW_"):
            review_needed.append((old_name, dir_info))
            continue
        
        rename_plan.append((old_name, new_name, dir_info))
    
    # Print summary
    print("SUMMARY")
    print("-" * 80)
    print(f"Total directories: {len(subdirs)}")
    print(f"  To rename: {len(rename_plan)}")
    print(f"  To delete: {len(delete_plan)}")
    print(f"  Need review: {len(review_needed)}")
    print()
    
    # Print rename plan
    if rename_plan:
        print("\nRENAME PLAN")
        print("-" * 80)
        for old_name, new_name, info in rename_plan:
            status = []
            if info["has_checkpoints"]:
                status.append(f"✓ {info['checkpoint_count']} checkpoints")
            if info["has_predictions"]:
                status.append("✓ predictions")
            if info["has_config"]:
                status.append("✓ config")
            
            status_str = ", ".join(status) if status else "empty?"
            print(f"{old_name:60s} → {new_name:30s} [{status_str}]")
    
    # Print delete plan
    if delete_plan:
        print("\nDELETE PLAN (debug/temporary directories)")
        print("-" * 80)
        for old_name, info in delete_plan:
            size = "unknown"
            old_path = outputs_dir / old_name
            if old_path.exists():
                # Get directory size
                try:
                    size_bytes = sum(f.stat().st_size for f in old_path.rglob('*') if f.is_file())
                    size = f"{size_bytes / (1024**3):.2f} GB"
                except:
                    pass
            print(f"{old_name:60s} [Size: {size}]")
    
    # Print review needed
    if review_needed:
        print("\nNEED MANUAL REVIEW")
        print("-" * 80)
        for old_name, info in review_needed:
            status = []
            if info["has_checkpoints"]:
                status.append(f"{info['checkpoint_count']} ckpts")
            if info["has_predictions"]:
                status.append("predictions")
            status_str = ", ".join(status) if status else "empty"
            print(f"{old_name:60s} [{status_str}]")
        print("\nThese directories don't match expected patterns.")
        print("Please review manually and add to EXPERIMENT_MAPPING if needed.")
    
    # Execute if not dry run
    if not dry_run:
        print("\n" + "=" * 80)
        print("EXECUTING CHANGES")
        print("=" * 80)
        
        # Create backup mapping file
        mapping_file = outputs_dir / "rename_mapping.json"
        mapping_data = {
            "renamed": {},
            "deleted": [],
            "review_needed": [name for name, _ in review_needed]
        }
        
        # Execute renames
        for old_name, new_name, info in rename_plan:
            old_path = outputs_dir / old_name
            new_path = outputs_dir / new_name
            
            if new_path.exists():
                print(f"⚠️  SKIP: {new_name} already exists!")
                continue
            
            print(f"Renaming: {old_name} → {new_name}")
            old_path.rename(new_path)
            mapping_data["renamed"][old_name] = new_name
            
            # Create symlink for backward compatibility
            if create_symlinks:
                try:
                    old_path.symlink_to(new_name)
                    print(f"  Created symlink: {old_name} → {new_name}")
                except Exception as e:
                    print(f"  Warning: Could not create symlink: {e}")
        
        # Execute deletions (move to trash folder)
        if delete_plan:
            trash_dir = outputs_dir / "_TRASH"
            trash_dir.mkdir(exist_ok=True)
            
            for old_name, info in delete_plan:
                old_path = outputs_dir / old_name
                trash_path = trash_dir / old_name
                
                print(f"Moving to trash: {old_name}")
                old_path.rename(trash_path)
                mapping_data["deleted"].append(old_name)
        
        # Save mapping file
        with open(mapping_file, 'w') as f:
            json.dump(mapping_data, f, indent=2)
        print(f"\n✓ Saved mapping to: {mapping_file}")
        
        print("\n" + "=" * 80)
        print("REORGANIZATION COMPLETE!")
        print("=" * 80)
        print(f"\nRenamed: {len(rename_plan)} directories")
        print(f"Deleted: {len(delete_plan)} directories (moved to _TRASH)")
        if create_symlinks:
            print("Created symlinks for backward compatibility")
        
        if review_needed:
            print(f"\n⚠️  {len(review_needed)} directories need manual review")
    
    else:
        print("\n" + "=" * 80)
        print("DRY RUN COMPLETE - No changes made")
        print("=" * 80)
        print("\nTo execute changes, run with: --execute")
        if create_symlinks:
            print("Symlinks will be created for backward compatibility")


def create_category_structure(outputs_dir: Path, dry_run: bool = True):
    """Optionally create category subdirectories."""
    print("\n" + "=" * 80)
    print("OPTIONAL: Create category subdirectories?")
    print("=" * 80)
    print("""
This would organize outputs like:
outputs/
├── baseline/
│   ├── baseline_mse/
│   └── baseline_mae/
├── helmholtz_270x180/
│   ├── helmholtz270_mae_grad_w01/
│   └── ...
└── helmholtz_full/
    └── ...

This is optional and may break existing paths in scripts.
Recommend keeping flat structure for now.
""")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Reorganize existing outputs directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (see what would happen)
  python reorganize_existing_outputs.py
  
  # Execute changes
  python reorganize_existing_outputs.py --execute
  
  # Execute without creating symlinks
  python reorganize_existing_outputs.py --execute --no-symlinks
"""
    )
    
    parser.add_argument(
        '--execute',
        action='store_true',
        help='Actually perform the reorganization (default is dry run)'
    )
    
    parser.add_argument(
        '--no-symlinks',
        action='store_true',
        help='Do not create symlinks for backward compatibility'
    )
    
    parser.add_argument(
        '--outputs-dir',
        type=Path,
        default=OUTPUTS_DIR,
        help=f'Path to outputs directory (default: {OUTPUTS_DIR})'
    )
    
    args = parser.parse_args()
    
    # Validate outputs directory
    if not args.outputs_dir.exists():
        print(f"Error: Outputs directory not found: {args.outputs_dir}")
        print("\nPlease update OUTPUTS_DIR in the script or use --outputs-dir")
        return 1
    
    # Run reorganization
    reorganize_outputs(
        args.outputs_dir,
        dry_run=not args.execute,
        create_symlinks=not args.no_symlinks
    )
    
    return 0


if __name__ == "__main__":
    exit(main())