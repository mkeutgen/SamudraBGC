#!/usr/bin/env python3
"""
Cleanup script to remove old directory names after successful reorganization.

This script identifies directories that have been successfully copied to new names
and removes the old versions, keeping only the new clean names.
"""

import os
from pathlib import Path
import json
import shutil

# UPDATE THIS PATH
OUTPUTS_DIR = Path("/scratch/cimes/maximek/INMOS/Ocean_Emulator/outputs")

# Mapping from the reorganization
EXPERIMENT_MAPPING = {
    # Baseline experiments
    "mom6_cobalt_bgc_clim_baseline": "baseline_mse",
    "mom6_cobalt_bgc_clim_baseline_correct_date": "baseline_mse_corrected",
    "mom6_cobalt_bgc_clim_mae": "baseline_mae",
    
    # Baseline evals
    "mom6_cobalt_bgc_clim_baseline_eval": "baseline_mse_eval",
    "mom6_cobalt_bgc_clim_baseline_eval_long_rollout": "baseline_mse_eval_long",
    "mom6_cobalt_bgc_clim_mae_eval": "baseline_mae_eval",
    
    # Helmholtz 270x180 experiments
    "mom6_cobalt_bgc_clim_mae_grad_w01_helmholtz_l40s": "helmholtz270_mae_grad_w01",
    "mom6_cobalt_bgc_clim_mae_grad_w025_helmholtz_l40": "helmholtz270_mae_grad_w025",
    "mom6_cobalt_bgc_clim_mae_gradient": "helmholtz270_mae_grad_60ep",
    "mom6_cobalt_bgc_clim_mae_grad_eval": "helmholtz270_mae_grad_eval",
    
    # Helmholtz full domain
    "mom6_cobalt_bgc_clim_mae_grad_w01_helmholtz_full": "helmholtzfull_mae_grad_w01_25lev",
    "mom6_cobalt_bgc_clim_mae_grad_w025_helmholtz_full": "helmholtzfull_mae_grad_w025_25lev",
    
    # Additional mappings from screenshot
    "mom6_cobalt_bgc_clim_mae_grad_w01": "baseline_mae_grad_w01",
    "mom6_cobalt_bgc_clim_mae_grad_w025": "baseline_mae_grad_w025",
}


def check_directories_exist(outputs_dir: Path) -> dict:
    """Check which old and new directories exist."""
    status = {
        "both_exist": [],      # Both old and new exist (safe to delete old)
        "only_old": [],        # Only old exists (not yet renamed)
        "only_new": [],        # Only new exists (already cleaned)
        "neither": [],         # Neither exists (error in mapping?)
    }
    
    for old_name, new_name in EXPERIMENT_MAPPING.items():
        old_path = outputs_dir / old_name
        new_path = outputs_dir / new_name
        
        old_exists = old_path.exists()
        new_exists = new_path.exists()
        
        if old_exists and new_exists:
            status["both_exist"].append((old_name, new_name))
        elif old_exists and not new_exists:
            status["only_old"].append((old_name, new_name))
        elif not old_exists and new_exists:
            status["only_new"].append((old_name, new_name))
        else:
            status["neither"].append((old_name, new_name))
    
    return status


def get_directory_size(path: Path) -> float:
    """Get directory size in GB."""
    try:
        size_bytes = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
        return size_bytes / (1024**3)
    except:
        return 0.0


def verify_directories_identical(old_path: Path, new_path: Path) -> bool:
    """Quick check if directories are likely identical (compare sizes and file counts)."""
    try:
        # Count files
        old_files = list(old_path.rglob('*'))
        new_files = list(new_path.rglob('*'))
        
        if len(old_files) != len(new_files):
            return False
        
        # Compare sizes
        old_size = sum(f.stat().st_size for f in old_files if f.is_file())
        new_size = sum(f.stat().st_size for f in new_files if f.is_file())
        
        # Allow 1% difference for rounding
        return abs(old_size - new_size) / max(old_size, 1) < 0.01
        
    except Exception as e:
        print(f"  Warning: Could not verify {old_path.name}: {e}")
        return False


def cleanup_duplicates(outputs_dir: Path, dry_run: bool = True, verify: bool = True):
    """Remove old directories that have been successfully copied to new names."""
    
    if not outputs_dir.exists():
        print(f"Error: Outputs directory not found: {outputs_dir}")
        return
    
    print("=" * 80)
    print("Duplicate Directory Cleanup")
    print(f"Directory: {outputs_dir}")
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    print("=" * 80)
    print()
    
    # Check status
    status = check_directories_exist(outputs_dir)
    
    print("STATUS SUMMARY")
    print("-" * 80)
    print(f"Both old & new exist: {len(status['both_exist'])} (can delete old)")
    print(f"Only old exists: {len(status['only_old'])} (not yet copied)")
    print(f"Only new exists: {len(status['only_new'])} (already cleaned)")
    print()
    
    # Handle directories where both exist
    if status["both_exist"]:
        print("\nDIRECTORIES TO DELETE (old names, new versions exist)")
        print("-" * 80)
        
        delete_list = []
        skip_list = []
        
        for old_name, new_name in status["both_exist"]:
            old_path = outputs_dir / old_name
            new_path = outputs_dir / new_name
            
            old_size = get_directory_size(old_path)
            new_size = get_directory_size(new_path)
            
            # Verify if requested
            if verify:
                is_identical = verify_directories_identical(old_path, new_path)
                if is_identical:
                    delete_list.append((old_name, new_name, old_size))
                    print(f"✓ {old_name:50s} [{old_size:6.2f} GB] → VERIFIED, safe to delete")
                else:
                    skip_list.append((old_name, new_name, old_size, new_size))
                    print(f"⚠ {old_name:50s} [{old_size:6.2f} GB] → DIFFERENT from new ({new_size:.2f} GB), SKIP")
            else:
                delete_list.append((old_name, new_name, old_size))
                print(f"  {old_name:50s} [{old_size:6.2f} GB]")
        
        total_size = sum(size for _, _, size in delete_list)
        print(f"\nTotal to delete: {len(delete_list)} directories, {total_size:.2f} GB")
        
        if skip_list:
            print(f"\n⚠️  Skipping {len(skip_list)} directories due to size mismatch!")
            print("These may need manual inspection:")
            for old_name, new_name, old_size, new_size in skip_list:
                print(f"  {old_name} (old: {old_size:.2f} GB vs new: {new_size:.2f} GB)")
    
    # Handle directories that only have old names
    if status["only_old"]:
        print("\n\nDIRECTORIES NOT YET COPIED")
        print("-" * 80)
        print("These old directories exist but new versions don't:")
        for old_name, new_name in status["only_old"]:
            print(f"  {old_name} → {new_name} (NEW MISSING!)")
        print("\nYou may need to re-run reorganize_existing_outputs.py")
    
    # Execute deletion if not dry run
    if not dry_run and delete_list:
        print("\n" + "=" * 80)
        print("EXECUTING DELETION")
        print("=" * 80)
        
        # Move to trash instead of direct deletion
        trash_dir = outputs_dir / "_TRASH_old_names"
        trash_dir.mkdir(exist_ok=True)
        
        for old_name, new_name, size in delete_list:
            old_path = outputs_dir / old_name
            trash_path = trash_dir / old_name
            
            print(f"Moving to trash: {old_name}")
            try:
                shutil.move(str(old_path), str(trash_path))
                print(f"  ✓ Moved {old_name} to _TRASH_old_names/")
            except Exception as e:
                print(f"  ✗ Error: {e}")
        
        print(f"\n✓ Moved {len(delete_list)} old directories to _TRASH_old_names/")
        print(f"  Freed up ~{total_size:.2f} GB (once trash is emptied)")
        print(f"\nTo permanently delete: rm -rf {trash_dir}")
    
    elif dry_run:
        print("\n" + "=" * 80)
        print("DRY RUN COMPLETE - No changes made")
        print("=" * 80)
        print("\nTo execute cleanup, run with: --execute")
        print("To skip verification, add: --no-verify")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Cleanup duplicate directories after reorganization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview what will be deleted (safe)
  python cleanup_duplicates.py
  
  # Execute cleanup (moves to _TRASH_old_names/)
  python cleanup_duplicates.py --execute
  
  # Execute without verification checks
  python cleanup_duplicates.py --execute --no-verify
  
After running, you can safely delete _TRASH_old_names/ after verifying:
  rm -rf outputs/_TRASH_old_names/
"""
    )
    
    parser.add_argument(
        '--execute',
        action='store_true',
        help='Actually delete old directories (default is dry run)'
    )
    
    parser.add_argument(
        '--no-verify',
        action='store_true',
        help='Skip verification that old and new directories are identical'
    )
    
    parser.add_argument(
        '--outputs-dir',
        type=Path,
        default=OUTPUTS_DIR,
        help=f'Path to outputs directory (default: {OUTPUTS_DIR})'
    )
    
    args = parser.parse_args()
    
    cleanup_duplicates(
        args.outputs_dir,
        dry_run=not args.execute,
        verify=not args.no_verify
    )


if __name__ == "__main__":
    main()
