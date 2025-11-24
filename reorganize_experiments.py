#!/usr/bin/env python3
"""
Reorganize MOM6-DG experiment configs and create eval configs + shell scripts.

This script:
1. Creates new directory structure
2. Copies and renames train configs
3. Generates matching eval configs
4. Creates train.sh and eval.sh scripts
5. Updates experiment names in configs
"""

import os
import shutil
from pathlib import Path
import yaml
from typing import Dict, List

# Base paths
REPO_ROOT = Path("/scratch/cimes/maximek/INMOS/Ocean_Emulator")  # UPDATE THIS
CONFIGS_DIR = REPO_ROOT / "configs"
SCRIPTS_DIR = REPO_ROOT / "scripts"

# Experiment definitions
EXPERIMENTS = {
    "baseline": {
        "domain": "270x180",
        "data_root": "/scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_Clim",
        "experiments": [
            {
                "old": "train_mom6dg.yaml",
                "new": "mse_baseline.yaml",
                "name": "baseline_mse",
                "loss": "mse",
                "epochs": 50,
                "ch_width": [320, 440, 600],
                "prognostic": "full_state_all",
                "checkpointing": None,
            },
            {
                "old": "train_mom6dg_mae.yaml",
                "new": "mae_baseline.yaml",
                "name": "baseline_mae",
                "loss": "mae",
                "epochs": 60,
                "ch_width": [320, 440, 600],
                "prognostic": "full_state_all",
                "checkpointing": None,
            },
        ]
    },
    "helmholtz_270x180": {
        "domain": "270x180",
        "data_root": "/scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_Clim",
        "experiments": [
            {
                "old": "train_mom6dg_mae_grad_w01.yaml",
                "new": "mae_grad_w01.yaml",
                "name": "helmholtz270_mae_grad_w01",
                "loss": "mae_gradient_weighted",
                "gradient_weight": 0.1,
                "epochs": 40,
                "ch_width": [320, 440, 600],
                "prognostic": "full_state_all",
                "checkpointing": "all",
            },
            {
                "old": "train_mom6dg_mae_grad_w025.yaml",
                "new": "mae_grad_w025.yaml",
                "name": "helmholtz270_mae_grad_w025",
                "loss": "mae_gradient_weighted",
                "gradient_weight": 0.25,
                "epochs": 40,
                "ch_width": [320, 440, 600],
                "prognostic": "full_state_all",
                "checkpointing": "all",
            },
            {
                "old": "train_mom6dg_mae_grad_60ep.yaml",
                "new": "mae_grad_60ep.yaml",
                "name": "helmholtz270_mae_grad_60ep",
                "loss": "mae_gradient",
                "epochs": 60,
                "ch_width": [320, 440, 600],
                "prognostic": "full_state_all",
                "checkpointing": None,
            },
            {
                "old": "train_mom6dg_mae_60ep_control.yaml",
                "new": "mae_control_60ep.yaml",
                "name": "helmholtz270_mae_control_60ep",
                "loss": "mae",
                "epochs": 60,
                "ch_width": [320, 440, 600],
                "prognostic": "optimized_helmholtz_skip2",
                "checkpointing": "all",
            },
        ]
    },
    "helmholtz_full": {
        "domain": "360x360",
        "data_root": "/scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_Clim_FULL",
        "experiments": [
            {
                "old": "train_mom6dg_mae_control_full.yaml",
                "new": "mae_control_50lev.yaml",
                "name": "helmholtzfull_mae_control_50lev",
                "loss": "mae",
                "epochs": 40,
                "ch_width": [160, 220, 300],
                "prognostic": "optimized_helmholtz_all",
                "checkpointing": None,
            },
            {
                "old": "train_mom6dg_mae_grad_w01_full.yaml",
                "new": "mae_grad_w01_25lev.yaml",
                "name": "helmholtzfull_mae_grad_w01_25lev",
                "loss": "mae_gradient_weighted",
                "gradient_weight": 0.1,
                "epochs": 40,
                "ch_width": [160, 220, 300],
                "prognostic": "optimized_helmholtz_25",
                "checkpointing": "all",
            },
            {
                "old": "train_mom6dg_mae_grad_w025_full.yaml",
                "new": "mae_grad_w025_25lev.yaml",
                "name": "helmholtzfull_mae_grad_w025_25lev",
                "loss": "mae_gradient_weighted",
                "gradient_weight": 0.25,
                "epochs": 40,
                "ch_width": [160, 220, 300],
                "prognostic": "optimized_helmholtz_25",
                "checkpointing": "all",
            },
        ]
    }
}

# SLURM configuration
SLURM_CONFIG = {
    "baseline": {
        "partition": "gpu",
        "nodes": 8,
        "gpus_per_node": 1,
        "time": "48:00:00",
        "mem": "100G",
    },
    "helmholtz_270x180": {
        "partition": "gpu",
        "nodes": 8,
        "gpus_per_node": 1,
        "time": "48:00:00",
        "mem": "100G",
    },
    "helmholtz_full": {
        "partition": "gpu",
        "nodes": 8,
        "gpus_per_node": 1,
        "time": "72:00:00",
        "mem": "100G",
    },
}


def create_directory_structure():
    """Create the new directory structure."""
    print("Creating directory structure...")
    
    for category in EXPERIMENTS.keys():
        # Config dirs
        (CONFIGS_DIR / "experiments" / category).mkdir(parents=True, exist_ok=True)
        (CONFIGS_DIR / "eval" / category).mkdir(parents=True, exist_ok=True)
        
        # Script dirs
        (SCRIPTS_DIR / "experiments" / category).mkdir(parents=True, exist_ok=True)
    
    # Archive dir
    (CONFIGS_DIR / "archived").mkdir(parents=True, exist_ok=True)
    
    print("✓ Directory structure created")


def update_train_config(config: Dict, experiment: Dict, category: str) -> Dict:
    """Update train config with new experiment name and settings."""
    config["experiment"]["name"] = experiment["name"]
    config["experiment"]["data_root"] = EXPERIMENTS[category]["data_root"]
    config["experiment"]["prognostic_vars_key"] = experiment["prognostic"]
    
    config["loss"] = experiment["loss"]
    if "gradient_weight" in experiment:
        config["gradient_weight"] = experiment["gradient_weight"]
    
    config["epochs"] = experiment["epochs"]
    config["model"]["unet"]["ch_width"] = experiment["ch_width"]
    
    if experiment["checkpointing"]:
        config["model"]["checkpointing"] = experiment["checkpointing"]
    elif "checkpointing" in config.get("model", {}):
        del config["model"]["checkpointing"]
    
    return config


def create_eval_config(train_config: Dict, experiment: Dict, category: str) -> Dict:
    """Generate eval config from train config."""
    eval_config = {
        "debug": False,
        "batch_size": 1,
        "ckpt_path": f"./outputs/{experiment['name']}/checkpoints/ckpt_ema_epoch_{experiment['epochs']:03d}.pt",
        
        "eval_time": {
            "start": "2025-01-01",
            "end": "2025-12-31"
        },
        
        "data_stride": train_config["data_stride"],
        "steps": train_config["steps"],
        "backend": train_config["backend"],
        
        "experiment": {
            "name": f"{experiment['name']}_eval",
            "rand_seed": 15,
            "base_output_dir": "./outputs",
            "wandb": {
                "mode": "disabled",
                "project": "ocean-emulators",
                "entity": "m2lines",
                "group": f"{category}-eval"
            },
            "prognostic_vars_key": experiment["prognostic"],
            "boundary_vars_key": train_config["experiment"]["boundary_vars_key"],
            "data_root": EXPERIMENTS[category]["data_root"]
        },
        
        "data": train_config["data"].copy(),
        "model": train_config["model"].copy(),
    }
    
    return eval_config


def create_train_script(experiment: Dict, category: str) -> str:
    """Generate SLURM training script."""
    slurm = SLURM_CONFIG[category]
    
    script = f"""#!/bin/bash
#SBATCH --job-name={experiment['name']}_train
#SBATCH --partition={slurm['partition']}
#SBATCH --nodes={slurm['nodes']}
#SBATCH --gpus-per-node={slurm['gpus_per_node']}
#SBATCH --time={slurm['time']}
#SBATCH --mem={slurm['mem']}
#SBATCH --output=logs/{experiment['name']}_train_%j.out
#SBATCH --error=logs/{experiment['name']}_train_%j.err

# Experiment: {experiment['name']}
# Category: {category}
# Domain: {EXPERIMENTS[category]['domain']}
# Loss: {experiment['loss']}
{f"# Gradient weight: {experiment['gradient_weight']}" if 'gradient_weight' in experiment else ""}
# Epochs: {experiment['epochs']}

set -e

# Load modules
module purge
module load anaconda3/2024.02
module load cuda/12.1

# Activate environment
source activate ocean_emulator

# Training
echo "Starting training: {experiment['name']}"
echo "Config: configs/experiments/{category}/{experiment['new']}"

srun --ntasks={slurm['nodes']} \\
     --ntasks-per-node=1 \\
     --gpus-per-node={slurm['gpus_per_node']} \\
     python -m ocean_emulators.train \\
     configs/experiments/{category}/{experiment['new']}

echo "Training complete!"
"""
    return script


def create_eval_script(experiment: Dict, category: str) -> str:
    """Generate SLURM evaluation script."""
    slurm = SLURM_CONFIG[category]
    
    script = f"""#!/bin/bash
#SBATCH --job-name={experiment['name']}_eval
#SBATCH --partition={slurm['partition']}
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=80G
#SBATCH --output=logs/{experiment['name']}_eval_%j.out
#SBATCH --error=logs/{experiment['name']}_eval_%j.err

# Evaluation for: {experiment['name']}
# Category: {category}

set -e

# Load modules
module purge
module load anaconda3/2024.02
module load cuda/12.1

# Activate environment
source activate ocean_emulator

# Evaluation
echo "Starting evaluation: {experiment['name']}"
echo "Config: configs/eval/{category}/{experiment['new']}"

python -m ocean_emulators.eval \\
     configs/eval/{category}/{experiment['new']}

echo "Evaluation complete!"
echo "Results saved to: ./outputs/{experiment['name']}_eval/"
"""
    return script


def process_experiments():
    """Process all experiments: copy configs, update, create scripts."""
    print("\nProcessing experiments...")
    
    for category, category_data in EXPERIMENTS.items():
        print(f"\n  Category: {category}")
        
        for exp in category_data["experiments"]:
            print(f"    - {exp['name']}")
            
            # Paths
            old_config_path = CONFIGS_DIR / exp["old"]
            new_train_path = CONFIGS_DIR / "experiments" / category / exp["new"]
            new_eval_path = CONFIGS_DIR / "eval" / category / exp["new"]
            train_script_path = SCRIPTS_DIR / "experiments" / category / f"train_{exp['new'].replace('.yaml', '.sh')}"
            eval_script_path = SCRIPTS_DIR / "experiments" / category / f"eval_{exp['new'].replace('.yaml', '.sh')}"
            
            # Load and update train config
            if old_config_path.exists():
                with open(old_config_path) as f:
                    train_config = yaml.safe_load(f)
                
                train_config = update_train_config(train_config, exp, category)
                
                # Save updated train config
                with open(new_train_path, 'w') as f:
                    f.write(f"# yaml-language-server: $schema=../../schemas/TrainConfig.json\n")
                    f.write(f"# Experiment: {exp['name']}\n")
                    f.write(f"# Category: {category}\n")
                    f.write(f"# Domain: {category_data['domain']}\n\n")
                    yaml.dump(train_config, f, default_flow_style=False, sort_keys=False)
                
                # Create eval config
                eval_config = create_eval_config(train_config, exp, category)
                with open(new_eval_path, 'w') as f:
                    f.write(f"# yaml-language-server: $schema=../../schemas/EvalConfig.json\n")
                    f.write(f"# Evaluation for: {exp['name']}\n")
                    f.write(f"# Category: {category}\n\n")
                    yaml.dump(eval_config, f, default_flow_style=False, sort_keys=False)
                
                # Create shell scripts
                with open(train_script_path, 'w') as f:
                    f.write(create_train_script(exp, category))
                train_script_path.chmod(0o755)
                
                with open(eval_script_path, 'w') as f:
                    f.write(create_eval_script(exp, category))
                eval_script_path.chmod(0o755)
                
            else:
                print(f"      WARNING: {old_config_path} not found!")
    
    print("\n✓ All experiments processed")


def create_category_readmes():
    """Create README files for each category."""
    print("\nCreating category READMEs...")
    
    readmes = {
        "baseline": """# Baseline Experiments

## Overview
Original experiments on 270×180 domain using u/v velocity fields directly.

## Experiments
- **mse_baseline**: MSE loss, 50 epochs
- **mae_baseline**: MAE loss, 60 epochs

## Purpose
Establish baseline performance before Helmholtz decomposition.
""",
        "helmholtz_270x180": """# Helmholtz Decomposition Experiments (270×180)

## Overview
Testing Helmholtz decomposition (ψ/φ) on reduced domain with different loss configurations.

## Experiments
- **mae_grad_w01**: Conservative gradient weighting (α=0.1)
- **mae_grad_w025**: Aggressive gradient weighting (α=0.25)
- **mae_grad_60ep**: Unweighted MAE+gradient, extended training
- **mae_control_60ep**: MAE only, extended training (control)

## Key Questions
1. Can weighted gradient loss preserve sharp features without bias?
2. Does extended training help convergence for multi-objective losses?
""",
        "helmholtz_full": """# Helmholtz Full Domain Experiments (360×360)

## Overview
Scaling Helmholtz decomposition to full 360×360 domain with memory optimizations.

## Experiments
- **mae_control_50lev**: All 50 depth levels, MAE only
- **mae_grad_w01_25lev**: 25 depth levels (skip2), α=0.1
- **mae_grad_w025_25lev**: 25 depth levels (skip2), α=0.25

## Architecture Changes
- Reduced channel widths: [160, 220, 300] vs [320, 440, 600]
- Gradient checkpointing enabled
- Variable selection: removed PP, kept surface Chl only
"""
    }
    
    for category, content in readmes.items():
        readme_path = CONFIGS_DIR / "experiments" / category / "README.md"
        with open(readme_path, 'w') as f:
            f.write(content)
    
    print("✓ Category READMEs created")


def create_main_readme():
    """Create main scripts README."""
    print("\nCreating scripts README...")
    
    content = """# Experiment Scripts

This directory contains SLURM scripts for training and evaluating models.

## Structure

```
scripts/experiments/
├── baseline/
├── helmholtz_270x180/
└── helmholtz_full/
```

## Usage

### Training
```bash
cd scripts/experiments/<category>
sbatch train_<experiment>.sh
```

### Evaluation
```bash
cd scripts/experiments/<category>
sbatch eval_<experiment>.sh
```

## Monitoring

Check logs in `logs/` directory:
```bash
tail -f logs/<experiment>_train_<jobid>.out
tail -f logs/<experiment>_eval_<jobid>.err
```

## Configuration

SLURM parameters can be adjusted in each script:
- `--nodes`: Number of nodes
- `--gpus-per-node`: GPUs per node
- `--time`: Wall time limit
- `--mem`: Memory per node

## Experiment Categories

See README.md in each category directory for experiment details.
"""
    
    with open(SCRIPTS_DIR / "experiments" / "README.md", 'w') as f:
        f.write(content)
    
    print("✓ Scripts README created")


def main():
    """Main reorganization workflow."""
    print("=" * 60)
    print("MOM6-DG Experiment Reorganization")
    print("=" * 60)
    
    # Step 1: Create directories
    create_directory_structure()
    
    # Step 2: Process experiments
    process_experiments()
    
    # Step 3: Create documentation
    create_category_readmes()
    create_main_readme()
    
    print("\n" + "=" * 60)
    print("Reorganization complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Review configs in configs/experiments/<category>/")
    print("2. Review eval configs in configs/eval/<category>/")
    print("3. Test one training script: sbatch scripts/experiments/baseline/train_mse_baseline.sh")
    print("4. Archive old configs: mv configs/train_mom6dg*.yaml configs/archived/")
    print("\nTo run all experiments:")
    print("  find scripts/experiments -name 'train_*.sh' -exec sbatch {} \\;")


if __name__ == "__main__":
    main()
