"""
Main orchestrator: Hydra-based entry point that launches train.py as subprocess.

CLI Usage:
  uv run python -u -m src.main run={run_id} results_dir={path} mode=full
  uv run python -u -m src.main run={run_id} results_dir={path} mode=trial
"""

import sys
import subprocess
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import hydra


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main orchestrator: launches train.py as subprocess with configuration.
    
    Args:
        cfg: Hydra configuration loaded from config/config.yaml and CLI overrides
    """
    
    # Handle run parameter - can be either a string or a config object
    if isinstance(cfg.run, str):
        run_id = cfg.run
    else:
        run_id = cfg.run.run_id

    print(f"\n{'='*70}")
    print(f"SDSLR Experiment Orchestrator")
    print(f"{'='*70}")
    print(f"Run ID: {run_id}")
    print(f"Mode: {cfg.mode}")
    print(f"Results Directory: {cfg.results_dir}")
    print(f"{'='*70}\n")

    # Validate required parameters
    if run_id is None:
        raise ValueError("ERROR: run parameter is required. Usage: python -m src.main run={run_id} results_dir={path} mode={mode}")
    
    if cfg.mode is None:
        raise ValueError("ERROR: mode parameter is required. Choose: trial or full")
    
    if cfg.mode not in ["trial", "full"]:
        raise ValueError(f"ERROR: mode must be 'trial' or 'full', got: {cfg.mode}")
    
    # Load run-specific config from config/runs/{run_id}.yaml if it exists
    run_config_path = Path("config/runs") / f"{run_id}.yaml"
    if run_config_path.exists():
        print(f"Loading run-specific config from: {run_config_path}")
        run_config = OmegaConf.load(run_config_path)
        cfg = OmegaConf.merge(cfg, run_config)
        print(f"✓ Run config merged with base config\n")
    else:
        print(f"Note: Run-specific config not found at {run_config_path}")
        print(f"Using base config with CLI overrides\n")
    
    # Create results directory
    results_path = Path(cfg.results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    
    # Build command to launch train.py as subprocess from repository root
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "src.train",
        f"run.run_id={run_id}",
        f"results_dir={cfg.results_dir}",
        f"mode={cfg.mode}",
    ]
    
    print(f"Launching train.py with command:")
    print(f"  {' '.join(cmd)}\n")
    
    # Launch training as subprocess from repository root
    repo_root = Path(__file__).parent.parent
    result = subprocess.run(cmd, cwd=str(repo_root))
    
    if result.returncode != 0:
        print(f"\nERROR: train.py exited with code {result.returncode}")
        sys.exit(result.returncode)

    print(f"\nTraining completed successfully for run: {run_id}")


if __name__ == "__main__":
    main()
