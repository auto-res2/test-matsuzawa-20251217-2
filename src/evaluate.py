"""
Independent evaluation and visualization script.

Execution: Run independently via 'uv run python -m src.evaluate results_dir={path} run_ids='["run-1", "run-2"]'
NOT called from main.py - executes as separate workflow after all training completes.

Responsibilities:
1. Per-Run Processing: Export metrics and generate run-specific figures
2. Aggregated Analysis: Compute cross-run metrics and generate comparison figures
"""

import json
import argparse
import sys
import os
from pathlib import Path
from typing import Dict, List, Any
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from omegaconf import OmegaConf


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate SDSLR experiment runs from WandB")
    parser.add_argument("--results_dir", type=str, required=True, help="Results directory path")
    parser.add_argument("--run_ids", type=str, required=True, help="JSON string list of run IDs")
    return parser.parse_args()


def load_wandb_config() -> Dict[str, str]:
    """Load WandB configuration from config/config.yaml."""
    config_path = Path("config/config.yaml")
    if config_path.exists():
        cfg = OmegaConf.load(config_path)
        return {
            "entity": cfg.wandb.entity if "wandb" in cfg else "gengaru617-personal",
            "project": cfg.wandb.project if "wandb" in cfg else "2025-11-19",
        }
    return {"entity": "gengaru617-personal", "project": "2025-11-19"}


def retrieve_wandb_data(run_ids: List[str], wandb_config: Dict[str, str]) -> Dict[str, Any]:
    """
    Retrieve comprehensive experimental data from WandB API.
    
    Returns:
        Dictionary mapping run_id to {history, summary, config, state}
    """
    # Validate WANDB_API_KEY is set
    if "WANDB_API_KEY" not in os.environ:
        raise RuntimeError("ERROR: WANDB_API_KEY environment variable not set. Cannot authenticate with WandB API.")
    
    api = wandb.Api()
    run_data = {}
    
    for run_id in run_ids:
        try:
            run_path = f"{wandb_config['entity']}/{wandb_config['project']}/{run_id}"
            print(f"Fetching data for run: {run_path}")
            
            run = api.run(run_path)
            
            # Skip disabled runs (trial mode)
            if hasattr(run, 'mode') and run.mode == "disabled":
                print(f"  ⊘ Skipping disabled run (trial mode)")
                continue
            
            # Get time-series history
            history = run.history()
            
            # Get summary
            summary = run.summary._json_dict if hasattr(run.summary, '_json_dict') else dict(run.summary)
            
            # Get configuration
            config = dict(run.config)
            
            run_data[run_id] = {
                "history": history,
                "summary": summary,
                "config": config,
                "state": run.state,
            }
            
            print(f"  ✓ Retrieved {len(history)} history steps")
            
        except Exception as e:
            print(f"  ✗ Error fetching {run_id}: {e}")
            run_data[run_id] = {
                "history": pd.DataFrame(),
                "summary": {},
                "config": {},
                "state": "error",
            }
    
    return run_data


def export_per_run_metrics(run_data: Dict[str, Any], results_dir: Path) -> None:
    """Export comprehensive run-specific metrics to JSON files."""
    
    for run_id, data in run_data.items():
        run_dir = results_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Export metrics
        metrics_file = run_dir / "metrics.json"
        metrics = {
            "run_id": run_id,
            "summary": data["summary"],
            "config": data["config"],
            "history_stats": {
                "n_steps": len(data["history"]),
                "columns": list(data["history"].columns) if not data["history"].empty else [],
            },
            "run_state": data["state"],
        }
        
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        
        print(f"Exported metrics for {run_id} to {metrics_file}")


def generate_per_run_figures(run_data: Dict[str, Any], results_dir: Path) -> None:
    """Generate per-run figures (learning curves, accuracy plots)."""
    
    sns.set_style("whitegrid")
    
    for run_id, data in run_data.items():
        history = data["history"]
        
        if history.empty:
            print(f"Skipping figures for {run_id}: empty history")
            continue
        
        run_dir = results_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Learning curve: plot all loss-related metrics
        loss_cols = [col for col in history.columns if "loss" in col.lower()]
        if loss_cols:
            fig, ax = plt.subplots(figsize=(12, 6))
            for col in loss_cols[:3]:
                ax.plot(history.index, history[col], label=col, linewidth=2, alpha=0.8)
            
            ax.set_xlabel("Step", fontsize=12)
            ax.set_ylabel("Loss", fontsize=12)
            ax.set_title(f"Training Curves - {run_id}", fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            fig_path = run_dir / f"{run_id}_learning_curve.pdf"
            plt.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Generated learning curve: {fig_path}")
        
        # Accuracy curve
        acc_cols = [col for col in history.columns if "accuracy" in col.lower()]
        if acc_cols:
            fig, ax = plt.subplots(figsize=(12, 6))
            for col in acc_cols[:3]:
                ax.plot(history.index, history[col], label=col, marker='o', linewidth=2, alpha=0.8)
            
            ax.set_xlabel("Step", fontsize=12)
            ax.set_ylabel("Accuracy", fontsize=12)
            ax.set_title(f"Accuracy Curves - {run_id}", fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1.05])
            plt.tight_layout()
            
            fig_path = run_dir / f"{run_id}_accuracy_curve.pdf"
            plt.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Generated accuracy curve: {fig_path}")


def aggregate_metrics(run_data: Dict[str, Any], run_ids: List[str]) -> Dict[str, Any]:
    """
    Aggregate metrics across all runs for comparison.
    """
    
    aggregated = {
        "primary_metric": "convergence_speed_ratio",
        "metrics": {},
    }
    
    # Extract all unique metric names
    all_metrics = set()
    for run_id in run_ids:
        if run_id in run_data and run_data[run_id]["summary"]:
            all_metrics.update(run_data[run_id]["summary"].keys())
    
    # Aggregate each metric
    for metric in sorted(all_metrics):
        aggregated["metrics"][metric] = {}
        for run_id in run_ids:
            if run_id in run_data and metric in run_data[run_id]["summary"]:
                value = run_data[run_id]["summary"][metric]
                if isinstance(value, (np.integer, np.floating)):
                    value = float(value)
                aggregated["metrics"][metric][run_id] = value
    
    # Identify proposed vs baseline
    proposed_runs = [rid for rid in run_ids if "proposed" in rid.lower()]
    baseline_runs = [rid for rid in run_ids if "comparative" in rid.lower() or "baseline" in rid.lower()]
    
    # Find best runs by primary metric
    primary_metric = "convergence_speed_ratio"
    
    if primary_metric in aggregated["metrics"]:
        metric_values = aggregated["metrics"][primary_metric]
        
        # Determine if higher or lower is better
        higher_is_better = True
        if any(x in primary_metric.lower() for x in ["loss", "residual", "error"]):
            higher_is_better = False
        
        # Best proposed
        proposed_values = {rid: metric_values[rid] 
                          for rid in proposed_runs 
                          if rid in metric_values}
        if proposed_values:
            if higher_is_better:
                best_proposed_id = max(proposed_values, key=proposed_values.get)
            else:
                best_proposed_id = min(proposed_values, key=proposed_values.get)
            aggregated["best_proposed"] = {
                "run_id": best_proposed_id,
                "value": proposed_values[best_proposed_id],
            }
        
        # Best baseline
        baseline_values = {rid: metric_values[rid] 
                          for rid in baseline_runs 
                          if rid in metric_values}
        if baseline_values:
            if higher_is_better:
                best_baseline_id = max(baseline_values, key=baseline_values.get)
            else:
                best_baseline_id = min(baseline_values, key=baseline_values.get)
            aggregated["best_baseline"] = {
                "run_id": best_baseline_id,
                "value": baseline_values[best_baseline_id],
            }
        
        # Calculate gap
        if "best_proposed" in aggregated and "best_baseline" in aggregated:
            proposed_val = aggregated["best_proposed"]["value"]
            baseline_val = aggregated["best_baseline"]["value"]
            if baseline_val != 0:
                if higher_is_better:
                    gap = (proposed_val - baseline_val) / baseline_val * 100
                else:
                    gap = (baseline_val - proposed_val) / baseline_val * 100
                aggregated["gap"] = gap
    
    # Validate aggregated metrics structure
    assert "primary_metric" in aggregated, "Missing primary_metric in aggregation"
    assert "metrics" in aggregated, "Missing metrics in aggregation"
    
    return aggregated


def generate_comparison_figures(run_data: Dict[str, Any], aggregated: Dict[str, Any], results_dir: Path) -> None:
    """Generate comparison figures across all runs."""
    
    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    sns.set_style("whitegrid")
    
    # Bar charts for each metric (limit to first 5 metrics)
    if "metrics" in aggregated and aggregated["metrics"]:
        for metric_idx, (metric_name, metric_values) in enumerate(list(aggregated["metrics"].items())[:5]):
            if not metric_values or len(metric_values) < 2:
                continue
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            run_ids = list(metric_values.keys())
            values = list(metric_values.values())
            
            colors = ["#FF6B6B" if "proposed" in rid else "#4ECDC4" for rid in run_ids]
            
            bars = ax.bar(range(len(run_ids)), values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
            
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            ax.set_xticks(range(len(run_ids)))
            ax.set_xticklabels(run_ids, rotation=45, ha='right', fontsize=10)
            ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
            ax.set_title(f"Comparison: {metric_name}", fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            fig_path = comparison_dir / f"comparison_{metric_name.lower().replace(' ', '_')}_bar_pair{metric_idx}.pdf"
            plt.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close()
            
            print(f"Generated comparison bar chart: {fig_path}")


def main():
    """Main evaluation pipeline."""
    
    print("\n" + "="*70)
    print("SDSLR Experiment Evaluation")
    print("="*70 + "\n")
    
    # Parse arguments
    args = parse_arguments()
    
    # Parse run IDs from JSON string
    try:
        run_ids = json.loads(args.run_ids)
    except json.JSONDecodeError:
        print(f"ERROR: Could not parse run_ids as JSON: {args.run_ids}")
        return
    
    print(f"Evaluating runs: {run_ids}\n")
    
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load WandB config
    wandb_config = load_wandb_config()
    print(f"WandB Config: entity={wandb_config['entity']}, project={wandb_config['project']}\n")
    
    # Retrieve WandB data
    print("--- Retrieving WandB Data ---")
    run_data = retrieve_wandb_data(run_ids, wandb_config)
    print()
    
    # Per-run processing
    print("--- Per-Run Processing ---")
    export_per_run_metrics(run_data, results_dir)
    generate_per_run_figures(run_data, results_dir)
    print()
    
    # Aggregated analysis
    print("--- Aggregated Analysis ---")
    aggregated = aggregate_metrics(run_data, run_ids)
    
    # Export aggregated metrics
    agg_file = results_dir / "comparison" / "aggregated_metrics.json"
    agg_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(agg_file, "w") as f:
        json.dump(aggregated, f, indent=2, default=str)
    
    print(f"Exported aggregated metrics to {agg_file}\n")
    
    # Generate comparison figures
    print("--- Generating Comparison Figures ---")
    generate_comparison_figures(run_data, aggregated, results_dir)
    print()
    
    # Print summary
    print("="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"Primary Metric: {aggregated.get('primary_metric', 'N/A')}")
    if "best_proposed" in aggregated:
        print(f"Best Proposed: {aggregated['best_proposed']['run_id']} = {aggregated['best_proposed']['value']:.4f}")
    if "best_baseline" in aggregated:
        print(f"Best Baseline: {aggregated['best_baseline']['run_id']} = {aggregated['best_baseline']['value']:.4f}")
    if "gap" in aggregated:
        print(f"Performance Gap: {aggregated['gap']:.2f}%")
    print("="*70 + "\n")
    
    print(f"✓ Evaluation completed successfully!")
    print(f"Results saved to: {results_dir}\n")


if __name__ == "__main__":
    main()
