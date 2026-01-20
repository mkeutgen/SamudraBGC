import datetime
import logging
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from ocean_emulators.aggregator import Aggregator
from ocean_emulators.backend import init_eval_backend
from ocean_emulators.config import EvalConfig
from ocean_emulators.ensemble_perturbation import EnsemblePerturbationConfig
from ocean_emulators.constants import (
    BOUNDARY_VARS,
    PROGNOSTIC_VARS,
    BoundaryVarNames,
    Grid,
    PrognosticVarNames,
    TensorMap,
    construct_metadata,
)
from ocean_emulators.datasets import InferenceDataset
from ocean_emulators.stepper import Stepper
from ocean_emulators.utils.data import (
    Normalize,
    extract_wet_mask,
    get_inference_steps,
    spherical_area_weights,
)
from ocean_emulators.utils.device import using_gpu
from ocean_emulators.utils.distributed import get_rank, get_world_size, is_main_process, set_seed
from ocean_emulators.utils.logging import (
    get_model_summary,
    handle_logging,
    handle_warnings,
)
from ocean_emulators.utils.wandb import WandBLogger

logger = logging.getLogger(__name__)


class Eval:
    def __init__(self, cfg: EvalConfig) -> None:
        cfg.prepare_output_dirs()

        self.cfg = cfg  # Store config for ensemble support

        # Initialize backend with optional distributed support
        self.device, self.distributed = init_eval_backend(
            cfg.backend, distributed=cfg.distributed
        )

        # Store distributed info for ensemble parallelization
        self.world_size = get_world_size()
        self.rank = get_rank()

        if self.distributed is not None:
            logger.info(
                f"[Rank {self.rank}] Distributed evaluation enabled with "
                f"{self.world_size} GPUs"
            )

        # Adjust workers and memory pinning based on device
        if not using_gpu():
            cfg.data.num_workers = 0  # Disable multi-processing on CPU
        elif cfg.disk_mode:
            cfg.data.num_workers = torch.cuda.device_count() * cfg.data.num_workers

        # Set seeds
        set_seed(cfg.experiment.rand_seed)

        # Getting prognostic and boundary variables
        self.prognostic_var_names: PrognosticVarNames = PROGNOSTIC_VARS[
            cfg.experiment.prognostic_vars_key
        ]
        self.boundary_var_names: BoundaryVarNames = BOUNDARY_VARS[
            cfg.experiment.boundary_vars_key
        ]

        levels = cfg.experiment.prognostic_vars_key.split("_")[-1]
        if "all" in levels:
            self.levels = 19
        else:
            self.levels = int(levels)

        str_prognostics = ", ".join([i for i in self.prognostic_var_names])
        str_boundaries = ", ".join([i for i in self.boundary_var_names])

        logger.info(f"Prognostic variables: {str_prognostics}")
        logger.info(f"Boundary variables: {str_boundaries}")
        logger.info(f"Levels: {self.levels}")

        self.N_bound = len(self.boundary_var_names)
        self.N_prog = len(self.prognostic_var_names)

        self.num_in = int((cfg.data.hist + 1) * (self.N_prog + self.N_bound))
        self.num_out = int((cfg.data.hist + 1) * self.N_prog)

        self.tensor_map = TensorMap.init_instance(
            cfg.experiment.prognostic_vars_key, cfg.experiment.boundary_vars_key
        )

        logger.info(f"Number of inputs (prognostic + boundary): {self.num_in}")
        logger.info(f"Number of outputs (prognostic): {self.num_out}")

        # Dataloaders
        logger.info(f"Loading data")
        self.data_container = cfg.data.build(
            cfg.experiment.resolved_data_root,
            self.boundary_var_names,
        )

        self.src = self.data_container.source_using_dask
        self.data = self.src.data
        self.static_data = self.data_container.static_data

        self.metadata = construct_metadata(self.data)
        self.wet, self.wet_surface = extract_wet_mask(
            self.data, self.prognostic_var_names, cfg.data.hist
        )
        self.wet_without_hist_cpu, _ = extract_wet_mask(
            self.data, self.prognostic_var_names, 0
        )
        self.area_weights: Grid = spherical_area_weights(self.data)
        self.area_weights = self.area_weights.to(self.device)

        self.normalize = Normalize.init_instance(
            self.src,
            prognostic_var_names=self.prognostic_var_names,
            boundary_var_names=self.boundary_var_names,
            wet_mask=self.wet_without_hist_cpu,
            wet_mask_surface=self.wet_surface,
        )
        self.wet_without_hist = self.wet_without_hist_cpu.to(self.device)

        # Model
        self.model = cfg.model.build(
            in_channels=self.num_in,
            out_channels=self.num_out,
            hist=cfg.data.hist,
            wet=self.wet.to(self.device),
            area_weights=self.area_weights,
            static_data=self.static_data,
        ).to(self.device)

        get_model_summary(self.model, None, cfg.debug)

        if cfg.ckpt_path is None:
            raise ValueError(
                "ckpt_path must be set; try --ckpt_path=path/to/checkpoint"
            )
        self.load_checkpoint(cfg.ckpt_path)

        self.network = self.model.__class__.__name__

        # Initialize WandB
        self.wandb_logger = WandBLogger.init_instance()
        self.wandb_logger.configure(
            cfg.experiment.wandb.mode in ("online", "offline"), is_main_process()
        )

        # Set up wandb run
        self.wandb_id, self.wandb_name = self.wandb_logger.setup_run(
            None, cfg, finetune=False
        )

        # Eval
        self.hist = cfg.data.hist
        self.output_dir = cfg.experiment.output_dir
        self.debug = cfg.debug
        self.num_workers = cfg.data.num_workers
        self.inference_time = cfg.inference_time
        self.num_model_steps_forward = cfg.num_model_steps_forward
        self.save_zarr = cfg.save_zarr
        self.model_path = cfg.ckpt_path
        self.normalize_before_mask = cfg.data.normalize_before_mask
        self.masked_fill_value = cfg.data.masked_fill_value
        self.init_inference_store()

    def load_checkpoint(self, ckpt_path: str):
        checkpoint = torch.load(ckpt_path, map_location=torch.device(self.device))
        model_state_dict = checkpoint["model"]
        new_state_dict = OrderedDict()
        for k, v in model_state_dict.items():
            name = k.removeprefix("module.")
            new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict)

    def init_inference_store(self, ensemble_config: EnsemblePerturbationConfig | None = None):
        """Initialize inference dataset with optional ensemble configuration."""
        sliced_src = self.src.slice(self.inference_time)
        self.num_time_steps = get_inference_steps(
            sliced_src,
            hist=self.hist,
        )
        self.inference_dataset = InferenceDataset(
            src=sliced_src,
            prognostic_var_names=self.prognostic_var_names,
            boundary_var_names=self.boundary_var_names,
            wet=self.wet_without_hist_cpu,
            wet_surface=self.wet_surface,
            hist=self.hist,
            normalize_before_mask=self.normalize_before_mask,
            masked_fill_value=self.masked_fill_value,
            long_rollout=True,
            ensemble_config=ensemble_config,
        )

    def run(self) -> None:
        """Run evaluation (ensemble or single) based on config."""
        if self.cfg.ensemble.enabled:
            logger.info(
                f"Running ensemble evaluation with {self.cfg.ensemble.n_ensemble} members"
            )
            self.run_ensemble()
        else:
            logger.info("Running single evaluation")
            self.run_single()

    def run_single(self) -> None:
        """Run single evaluation (original behavior)."""
        start_time = time.perf_counter()
        inf_stats = self.standalone_inference()
        time_elapsed = time.perf_counter() - start_time

        log_stats = {
            **inf_stats,
            "eval_total_seconds": time_elapsed,
        }

        if is_main_process():
            self.wandb_logger.log(log_stats, step=None)

        total_time = time.perf_counter() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info(f"Eval time (Including wandb logging) {total_time_str}")
        self.finish()

    def run_ensemble(self) -> None:
        """Run ensemble evaluation with perturbed initial conditions.

        When distributed mode is enabled, ensemble members are distributed across GPUs.
        Each GPU processes a subset of members, then statistics are gathered.
        """
        start_time = time.perf_counter()
        n_ensemble = self.cfg.ensemble.n_ensemble

        # Determine which ensemble members this GPU will process
        if self.distributed is not None:
            # Distribute ensemble members across GPUs
            members_per_gpu = (n_ensemble + self.world_size - 1) // self.world_size
            start_idx = self.rank * members_per_gpu
            end_idx = min(start_idx + members_per_gpu, n_ensemble)
            local_members = list(range(start_idx, end_idx))

            logger.info(
                f"[Rank {self.rank}] Processing ensemble members {start_idx}-{end_idx - 1} "
                f"({len(local_members)} members out of {n_ensemble} total)"
            )
        else:
            # Single GPU: process all members
            local_members = list(range(n_ensemble))

        # Run UNPERTURBED baseline evaluation on rank 0 only (or all ranks in single-GPU mode)
        unperturbed_stats = {}
        if self.rank == 0 or self.distributed is None:
            logger.info(f"\n{'=' * 70}")
            logger.info("Running UNPERTURBED baseline evaluation")
            logger.info(f"{'=' * 70}\n")

            # Initialize dataset without perturbations
            self.init_inference_store(ensemble_config=None)

            # Run inference for unperturbed case
            logger.info("Running inference for unperturbed baseline")
            unperturbed_stats = self.standalone_inference()

            logger.info("Completed unperturbed baseline evaluation")
            logger.info(f"  Output saved to: {self.output_dir}/evaluation.zarr\n")

        # Synchronize before starting ensemble members
        if self.distributed is not None:
            torch.distributed.barrier()

        # Process local ensemble members
        local_stats = []

        for ensemble_idx in local_members:
            logger.info(f"\n{'=' * 70}")
            if self.distributed is not None:
                logger.info(
                    f"[Rank {self.rank}] Ensemble member {ensemble_idx + 1}/{n_ensemble}"
                )
            else:
                logger.info(f"Ensemble member {ensemble_idx + 1}/{n_ensemble}")
            logger.info(f"{'=' * 70}\n")

            # Create ensemble configuration with unique seed
            ensemble_config = EnsemblePerturbationConfig(
                enabled=True,
                n_ensemble=n_ensemble,
                depth_max_m=self.cfg.ensemble.depth_max_m,
                dx_km=self.cfg.ensemble.dx_km,
                corr_sigma_km=self.cfg.ensemble.corr_sigma_km,
                pert_std_temp=self.cfg.ensemble.pert_std_temp,
                pert_rel_dic=self.cfg.ensemble.pert_rel_dic,
                pert_rel_o2=self.cfg.ensemble.pert_rel_o2,
                pert_rel_salt=self.cfg.ensemble.pert_rel_salt,
                use_vertical_taper=self.cfg.ensemble.use_vertical_taper,
                seed_offset=ensemble_idx * 100,  # Unique seed for each member
            )

            # Initialize dataset with this ensemble member's perturbations
            self.init_inference_store(ensemble_config=ensemble_config)

            # Modify output directory for this ensemble member
            original_output_dir = self.output_dir
            original_model_path = self.model_path
            original_save_zarr = self.save_zarr

            if self.cfg.ensemble.output_individual_members:
                member_output_dir = Path(self.output_dir) / f"ensemble_{ensemble_idx:03d}"
                member_output_dir.mkdir(parents=True, exist_ok=True)
                self.output_dir = member_output_dir
                # Update model path to save in member directory
                self.model_path = str(member_output_dir / Path(original_model_path).name)
            else:
                # Don't save zarr for individual members
                self.save_zarr = False

            # Run inference for this ensemble member
            logger.info(f"Running inference for ensemble member {ensemble_idx}")
            member_stats = self.standalone_inference()
            local_stats.append(member_stats)

            # Restore original settings
            self.output_dir = original_output_dir
            self.model_path = original_model_path
            self.save_zarr = original_save_zarr

            logger.info(
                f"Completed ensemble member {ensemble_idx + 1}/{n_ensemble}\n"
            )

        # Gather statistics from all GPUs
        if self.distributed is not None:
            logger.info(f"[Rank {self.rank}] Gathering statistics from all GPUs...")
            all_stats = self._gather_ensemble_stats(local_stats)
        else:
            all_stats = local_stats

        # Compute ensemble statistics and log (only on main process)
        if is_main_process():
            logger.info("\n" + "=" * 70)
            logger.info("Computing ensemble statistics")
            logger.info("=" * 70 + "\n")

            # Average metrics across ensemble members
            averaged_stats = self._average_ensemble_stats(all_stats)

            time_elapsed = time.perf_counter() - start_time

            log_stats = {
                **{f"unperturbed/{k}": v for k, v in unperturbed_stats.items()},
                **{f"ensemble_mean/{k}": v for k, v in averaged_stats.items()},
                "ensemble_total_seconds": time_elapsed,
                "n_ensemble": n_ensemble,
                "n_gpus": self.world_size,
            }

            self.wandb_logger.log(log_stats, step=None)

            total_time_str = str(datetime.timedelta(seconds=int(time_elapsed)))
            logger.info(f"Total ensemble evaluation time: {total_time_str}")
            logger.info(f"Average time per member: {time_elapsed / n_ensemble:.1f}s")
            if self.distributed is not None:
                logger.info(f"Distributed across {self.world_size} GPUs")

        self.finish()

    def _gather_ensemble_stats(self, local_stats: list[dict]) -> list[dict]:
        """Gather ensemble statistics from all GPUs to all processes.

        Uses PyTorch's all_gather_object which handles serialization automatically.
        Returns the flattened list of all statistics from all GPUs.
        """
        # Gather stats from all ranks
        gathered = [None] * self.world_size
        torch.distributed.all_gather_object(gathered, local_stats)

        # Flatten list of lists
        all_stats = []
        for gpu_stats in gathered:
            if gpu_stats is not None:
                all_stats.extend(gpu_stats)

        return all_stats

    def _average_ensemble_stats(self, ensemble_stats: list[dict]) -> dict:
        """Average statistics across ensemble members."""
        if not ensemble_stats:
            return {}

        # Get all keys from first member
        all_keys = set(ensemble_stats[0].keys())

        averaged = {}
        for key in all_keys:
            values = []
            for member_stats in ensemble_stats:
                if key in member_stats:
                    val = member_stats[key]
                    if isinstance(val, (int, float)):
                        values.append(val)

            if values:
                averaged[key] = float(np.mean(values))
                averaged[f"{key}_std"] = float(np.std(values))
                averaged[f"{key}_min"] = float(np.min(values))
                averaged[f"{key}_max"] = float(np.max(values))

        return averaged

    @torch.no_grad()
    def standalone_inference(self):
        self.model.eval()
        inf_aggregator = Aggregator.get_standalone_inference_aggregator(
            self.num_time_steps,
            self.metadata,
            self.hist,
            self.area_weights,
            self.wet_without_hist,
            self.num_out,
            self.prognostic_var_names,
        )

        Stepper.inference(
            model=self.model,
            dataset=self.inference_dataset,
            inf_aggregator=inf_aggregator,
            epoch=0,
            output_dir=self.output_dir,
            model_path=self.model_path,
            num_model_steps_forward=self.num_model_steps_forward,
            save_zarr=self.save_zarr,
        )
        logs = inf_aggregator.get_summary_logs()
        return {f"inference/{k}": v for k, v in logs.items()}

    def finish(self):
        self.wandb_logger.finish()


def main():
    cfg = EvalConfig.from_yaml_and_cli()
    cfg.prepare_output_dirs()  # we do this first so logging can use them

    handle_logging(cfg.debug, cfg.experiment.output_dir)
    handle_warnings()

    Evaluator = Eval(cfg)

    try:
        Evaluator.run()
    except Exception as e:
        # Log the exception with traceback
        logger.exception("Evaluation failed with an exception")
        raise e


if __name__ == "__main__":
    main()
