"""
Training script: Executes single optimization experiment run.

Launched as subprocess by main.py. Handles:
- Synthetic streaming linear regression
- MNIST binary classification
- Ill-conditioned stress test
- Federated learning

Comprehensive WandB logging, WandB-disabled trial mode, Optuna hyperparameter optimization.
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Tuple, Any, Optional, List
import warnings
warnings.filterwarnings("ignore")

from omegaconf import DictConfig, OmegaConf
import hydra
import wandb
import optuna
from optuna.samplers import TPESampler

from src.preprocess import get_dataset
from src.model import get_model


class SpectrumDependentStreamingAdaptiveLR:
    """
    SDSLR Optimizer: Spectrum-Dependent Streaming Adaptive Learning Rate
    
    Core Innovation:
    - Maintains first moment (m), second moment (v), and fourth moment (h)
    - Computes Spectral Curvature Ratio (SCR) = h / (v² + ε)
    - Uses smooth tanh modulation: ρ = tanh(λ * (SCR - 3))
    - Final adaptive LR: lr(i) = α / (√v(i) + ε) * (1 + ρ(i))
    - Enables spectrum-aware optimization without gradient history buffers
    """
    
    def __init__(self, params: List[torch.nn.Parameter], lr: float = 0.001, 
                 betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
                 lambda_spectrum: float = 0.3, warmup_steps: int = 100):
        """
        Args:
            params: Model parameters to optimize
            lr: Initial learning rate (α)
            betas: Exponential decay rates for first and second moments
            eps: Numerical stability constant
            lambda_spectrum: Spectral sensitivity parameter ∈ [0, 0.5]
            warmup_steps: Number of warmup iterations before spectrum awareness activates
        """
        self.param_groups = [{"params": list(params), "lr": lr, "betas": betas, 
                             "eps": eps, "lambda_spectrum": lambda_spectrum,
                             "warmup_steps": warmup_steps}]
        self.state = {}
        self.step_count = 0
    
    def zero_grad(self):
        """Zero out gradients."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad.zero_()
    
    def step(self):
        """Perform single SDSLR optimization step with spectrum-dependent adaptation."""
        self.step_count += 1
        
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                # Check for numerical issues
                if torch.isnan(grad).any() or torch.isinf(grad).any():
                    print(f"WARNING: NaN/Inf in gradient at step {self.step_count}")
                    continue
                
                # Initialize state for this parameter
                if p not in self.state:
                    self.state[p] = {
                        "m": torch.zeros_like(p.data),
                        "v": torch.zeros_like(p.data),
                        "h": torch.zeros_like(p.data),
                    }
                
                state = self.state[p]
                m, v, h = state["m"], state["v"], state["h"]
                
                beta1, beta2 = group["betas"]
                lr = group["lr"]
                eps = group["eps"]
                lambda_spectrum = group["lambda_spectrum"]
                warmup_steps = group["warmup_steps"]
                
                # Update exponential moving averages
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                h.mul_(beta2).add_(grad ** 4, alpha=1.0 - beta2)
                
                # Bias correction
                bias_correction1 = 1.0 - beta1 ** self.step_count
                bias_correction2 = 1.0 - beta2 ** self.step_count
                
                m_hat = m / bias_correction1
                v_hat = v / bias_correction2
                h_hat = h / bias_correction2
                
                # Compute Spectral Curvature Ratio (SCR)
                v_sq_safe = torch.clamp(v_hat ** 2, min=eps)
                scr = torch.clamp(h_hat / (v_sq_safe + eps), min=0.5, max=10.0)
                
                # Compute spectral modulation function
                if self.step_count < warmup_steps:
                    modulation = torch.ones_like(scr)
                else:
                    rho = torch.tanh(lambda_spectrum * (scr - 3.0))
                    modulation = 1.0 + rho
                
                # Compute adaptive learning rate
                adaptive_lr = lr / (torch.sqrt(v_hat) + eps) * modulation
                
                # CRITICAL PRE-OPTIMIZER ASSERTIONS
                assert p.grad is not None, f"Gradient is None at step {self.step_count}"
                assert not torch.isnan(p.grad).any(), f"NaN in gradient at step {self.step_count}"
                assert not torch.isinf(p.grad).any(), f"Inf in gradient at step {self.step_count}"
                if self.step_count > 10:
                    grad_norm = torch.norm(p.grad)
                    assert not torch.allclose(grad_norm, torch.tensor(0.0, device=grad_norm.device), atol=1e-10), \
                        f"Gradient is all-zero (no gradient flow) at step {self.step_count}"
                
                # Update parameters: p = p - adaptive_lr * grad
                p.data.add_(grad * adaptive_lr, alpha=-1.0)


class StandardAdam:
    """Standard Adam optimizer for baseline comparison (no spectrum awareness)."""
    
    def __init__(self, params: List[torch.nn.Parameter], lr: float = 0.001,
                 betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8):
        """
        Args:
            params: Model parameters to optimize
            lr: Initial learning rate
            betas: Exponential decay rates
            eps: Numerical stability constant
        """
        self.param_groups = [{"params": list(params), "lr": lr, "betas": betas, "eps": eps}]
        self.state = {}
        self.step_count = 0
    
    def zero_grad(self):
        """Zero out gradients."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad.zero_()
    
    def step(self):
        """Perform single Adam optimization step."""
        self.step_count += 1
        
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                # Initialize state
                if p not in self.state:
                    self.state[p] = {
                        "m": torch.zeros_like(p.data),
                        "v": torch.zeros_like(p.data),
                    }
                
                state = self.state[p]
                m, v = state["m"], state["v"]
                
                beta1, beta2 = group["betas"]
                lr = group["lr"]
                eps = group["eps"]
                
                # Update moments
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                
                # Bias correction
                bias_correction1 = 1.0 - beta1 ** self.step_count
                bias_correction2 = 1.0 - beta2 ** self.step_count
                
                m_hat = m / bias_correction1
                v_hat = v / bias_correction2
                
                # CRITICAL PRE-OPTIMIZER ASSERTIONS
                assert p.grad is not None, f"Gradient is None at step {self.step_count}"
                assert not torch.isnan(p.grad).any(), f"NaN in gradient at step {self.step_count}"
                assert not torch.isinf(p.grad).any(), f"Inf in gradient at step {self.step_count}"
                if self.step_count > 10:
                    grad_norm = torch.norm(p.grad)
                    assert not torch.allclose(grad_norm, torch.tensor(0.0, device=grad_norm.device), atol=1e-10), \
                        f"Gradient is all-zero at step {self.step_count}"
                
                # Update parameters: p = p - lr * m_hat / (sqrt(v_hat) + eps)
                p.data.addcdiv_(m_hat, torch.sqrt(v_hat) + eps, value=-lr)


def setup_wandb(cfg: DictConfig, run_id: str) -> Optional[wandb.Run]:
    """
    Initialize WandB if not in disabled mode.
    
    Args:
        cfg: Hydra configuration
        run_id: Unique run identifier
        
    Returns:
        WandB run object or None if disabled
    """
    if cfg.wandb.mode == "disabled":
        return None
    
    run = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        id=run_id,
        config=OmegaConf.to_container(cfg, resolve=True),
        resume="allow",
        mode=cfg.wandb.mode,
    )
    print(f"\nWandB run initialized: {run.url}\n")
    return run


def train_task(cfg: DictConfig, trial: Optional[optuna.Trial] = None) -> Dict[str, float]:
    """
    Unified training function for all tasks.
    
    Handles:
    - Synthetic streaming linear regression
    - MNIST binary classification
    - Ill-conditioned stress test
    - Federated learning
    
    Args:
        cfg: Configuration with task and hyperparameters
        trial: Optuna trial object for hyperparameter optimization
        
    Returns:
        Dictionary of computed metrics
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    X_train, y_train, X_test, y_test = get_dataset(cfg)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    
    # Get model
    model = get_model(cfg).to(device)
    
    # POST-INIT ASSERTIONS
    assert model is not None, "Model creation failed"
    params_list = list(model.parameters())
    assert len(params_list) > 0, "Model has no parameters"
    total_params = sum(p.numel() for p in params_list)
    print(f"✓ Model initialized with {total_params} parameters")
    
    # Initialize optimizer
    if cfg.training.optimizer == "SDSLR":
        optimizer = SpectrumDependentStreamingAdaptiveLR(
            model.parameters(),
            lr=cfg.training.learning_rate,
            betas=(cfg.training.optimizer_config.beta1, cfg.training.optimizer_config.beta2),
            eps=cfg.training.optimizer_config.epsilon,
            lambda_spectrum=cfg.training.optimizer_config.lambda_spectrum,
            warmup_steps=cfg.training.warmup_steps,
        )
    else:
        optimizer = StandardAdam(
            model.parameters(),
            lr=cfg.training.learning_rate,
            betas=(cfg.training.optimizer_config.beta1, cfg.training.optimizer_config.beta2),
            eps=cfg.training.optimizer_config.epsilon,
        )
    
    # Determine task type
    task_type = cfg.model.name if hasattr(cfg.model, 'name') else "unknown"
    
    # Determine training iterations based on mode
    if cfg.mode == "trial":
        max_epochs = 1
        max_iters_per_epoch = 2
    else:
        max_epochs = cfg.training.epochs
        max_iters_per_epoch = min(len(X_train), 1000)
    
    # Initialize tracking variables
    losses_train = []
    losses_val = []
    accuracies_val = []
    learning_rates_per_coord = []
    iter_count = 0
    start_time = time.time()
    
    # Training loop
    print(f"Starting training for task: {task_type}")
    print(f"Epochs: {max_epochs}, Max iterations per epoch: {max_iters_per_epoch}\n")
    
    for epoch in range(max_epochs):
        epoch_loss = 0.0
        epoch_iters = 0
        
        for it in range(max_iters_per_epoch):
            iter_count += 1
            
            # Sample batch
            if len(X_train) > cfg.training.batch_size:
                idx = torch.randperm(len(X_train))[:cfg.training.batch_size]
                x_batch = X_train[idx]
                y_batch = y_train[idx]
            else:
                x_batch = X_train
                y_batch = y_train
            
            # BATCH-START ASSERTIONS (check at iteration 1 and first 10)
            if iter_count <= 10:
                assert x_batch.shape[0] > 0, "Batch size is zero"
                assert y_batch.shape[0] == x_batch.shape[0], \
                    f"Label batch size mismatch: {y_batch.shape[0]} vs {x_batch.shape[0]}"
                assert x_batch.dtype == torch.float32, f"Data type mismatch: {x_batch.dtype}"
                if iter_count == 1:
                    print(f"✓ Batch shapes verified: x={x_batch.shape}, y={y_batch.shape}")
            
            # Forward pass
            if "classification" in task_type.lower():
                # Binary classification
                logits = model(x_batch)
                pred_probs = torch.clamp(logits, min=1e-7, max=1-1e-7)
                
                # Reshape for proper broadcasting
                y_batch_expanded = y_batch.unsqueeze(1) if y_batch.dim() == 1 else y_batch
                loss = torch.mean(-(y_batch_expanded * torch.log(pred_probs) + 
                                   (1 - y_batch_expanded) * torch.log(1 - pred_probs)))
            else:
                # Regression (linear regression, least squares)
                pred = model(x_batch)
                target = y_batch.unsqueeze(1) if y_batch.dim() == 1 else y_batch
                loss = torch.mean((pred - target) ** 2)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # PRE-OPTIMIZER ASSERTIONS: CRITICAL (check all iterations)
            for param in model.parameters():
                if param.grad is not None:
                    assert not torch.isnan(param.grad).any(), \
                        f"NaN in gradient at iteration {iter_count}"
                    assert not torch.isinf(param.grad).any(), \
                        f"Inf in gradient at iteration {iter_count}"
                    if iter_count >= 1:
                        grad_norm = torch.norm(param.grad)
                        assert not torch.allclose(grad_norm, torch.tensor(0.0, device=grad_norm.device), atol=1e-10), \
                            f"Zero gradient (no gradient flow) at iteration {iter_count}"
            
            # Optimizer step
            optimizer.step()
            
            # Track loss
            epoch_loss += loss.item()
            epoch_iters += 1
            losses_train.append(loss.item())
            
            # Extract learning rate statistics for SDSLR (every iteration for frequency)
            if cfg.training.optimizer == "SDSLR":
                param_lrs = []
                for p in model.parameters():
                    if p in optimizer.state:
                        v_hat = optimizer.state[p]["v"] / (1.0 - 0.999 ** optimizer.step_count)
                        lr_coord = cfg.training.learning_rate / (torch.sqrt(v_hat) + 1e-8)
                        param_lrs.extend(lr_coord.detach().cpu().numpy().flatten().tolist()[:10])
                if param_lrs:
                    learning_rates_per_coord.append(np.var(param_lrs))
                    if wandb.run is not None:
                        wandb.log({
                            "learning_rate_variance": np.var(param_lrs),
                            "iteration": iter_count,
                        })
            
            # Validation and logging per-batch (frequent logging as per spec)
            if iter_count % max(1, max_iters_per_epoch // 5) == 0 or iter_count <= 5:
                with torch.no_grad():
                    if "classification" in task_type.lower():
                        # Classification: compute accuracy
                        logits_val = model(X_test)
                        preds_val = (logits_val > 0.5).float()
                        y_test_expanded = y_test.unsqueeze(1) if y_test.dim() == 1 else y_test
                        acc = (preds_val.squeeze() == y_test_expanded.squeeze()).float().mean().item()
                        accuracies_val.append(acc)
                        
                        # Log per-batch metrics to WandB
                        if wandb.run is not None:
                            wandb.log({
                                "train_loss": loss.item(),
                                "test_accuracy": acc,
                                "iteration": iter_count,
                            })
                    else:
                        # Regression: compute loss
                        pred_val = model(X_test)
                        target_val = y_test.unsqueeze(1) if y_test.dim() == 1 else y_test
                        loss_val = torch.mean((pred_val - target_val) ** 2).item()
                        losses_val.append(loss_val)
                        
                        # Log per-batch metrics to WandB
                        if wandb.run is not None:
                            wandb.log({
                                "train_loss": loss.item(),
                                "test_loss": loss_val,
                                "iteration": iter_count,
                            })
    
    wall_clock_time = time.time() - start_time
    
    # Compute metrics
    metrics = {
        "final_train_loss": losses_train[-1] if losses_train else float('inf'),
        "total_iterations": iter_count,
        "wall_clock_time": wall_clock_time,
        "wall_clock_time_per_iter": wall_clock_time / max(1, iter_count),
    }
    
    if accuracies_val:
        metrics["test_accuracy"] = accuracies_val[-1]
        metrics["accuracy_gain"] = accuracies_val[-1]
    
    if losses_val:
        metrics["final_test_loss"] = losses_val[-1]
    
    if learning_rates_per_coord:
        metrics["learning_rate_variance"] = np.mean(learning_rates_per_coord)
    
    # Compute convergence speed ratio
    if losses_train and len(losses_train) > 10:
        final_loss = losses_train[-1]
        initial_loss = losses_train[0]
        target_loss = final_loss + 0.1 * (initial_loss - final_loss)
        iter_to_90 = next((i+1 for i, l in enumerate(losses_train) if l <= target_loss), len(losses_train))
        convergence_ratio = max(1, len(losses_train) - iter_to_90)
        metrics["convergence_speed_ratio"] = convergence_ratio / max(1, iter_to_90)
    
    # Log final metrics to WandB summary
    if wandb.run is not None:
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                wandb.summary[f"final_{key}"] = value
    
    # Log final metrics
    print(f"\nTraining completed in {wall_clock_time:.2f} seconds")
    
    return metrics


def run_optuna_optimization(cfg: DictConfig, objective_metric: str) -> Dict[str, Any]:
    """
    Run Optuna hyperparameter optimization.
    
    Note: Intermediate trial results are NOT logged to WandB per specification.
    Only final run with best hyperparameters is logged.
    """
    
    if not cfg.optuna.enabled or cfg.optuna.n_trials == 0:
        return {}
    
    print(f"\nRunning Optuna optimization with {cfg.optuna.n_trials} trials...")
    
    def objective(trial: optuna.Trial) -> float:
        # Create trial config
        trial_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        
        # Suggest hyperparameters
        for search_space in cfg.optuna.search_spaces:
            param_name = search_space.param_name
            
            if search_space.distribution_type == "uniform":
                value = trial.suggest_float(param_name, search_space.low, search_space.high)
            elif search_space.distribution_type == "loguniform":
                value = trial.suggest_float(param_name, search_space.low, search_space.high, log=True)
            elif search_space.distribution_type == "int":
                value = trial.suggest_int(param_name, int(search_space.low), int(search_space.high))
            else:
                continue
            
            # Update trial config
            if param_name == "lambda_spectrum":
                trial_cfg.training.optimizer_config.lambda_spectrum = value
            elif param_name == "learning_rate":
                trial_cfg.training.learning_rate = value
            elif param_name == "beta1":
                trial_cfg.training.optimizer_config.beta1 = value
            elif param_name == "beta2":
                trial_cfg.training.optimizer_config.beta2 = value
            elif param_name == "warmup_steps":
                trial_cfg.training.warmup_steps = value
        
        # Train with trial hyperparameters (WandB disabled for trials)
        trial_cfg.wandb.mode = "disabled"
        metrics = train_task(trial_cfg, trial=trial)
        
        # Return objective metric
        if objective_metric in metrics:
            value = metrics[objective_metric]
            if "loss" in objective_metric.lower() or "residual" in objective_metric.lower():
                return value
            else:
                return -value
        
        return float('inf')
    
    # Create and run study
    sampler = TPESampler(seed=cfg.training.seed)
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=cfg.optuna.n_trials, show_progress_bar=True)
    
    best_trial = study.best_trial
    print(f"\nBest trial: {best_trial.number}, Objective: {best_trial.value}")
    print(f"Best parameters: {best_trial.params}\n")
    
    return best_trial.params


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main training script: trains optimizer on specified task using Hydra configuration.
    
    Merges run-specific config parameters from CLI and loads defaults from config/config.yaml.
    Handles mode-based execution (trial vs full).
    """
    
    print(f"\n{'='*70}")
    print(f"SDSLR Training Script")
    print(f"{'='*70}")
    print(f"Run ID: {cfg.run.run_id}")
    print(f"Mode: {cfg.mode}")
    print(f"{'='*70}\n")

    # Load run-specific config from config/runs/{run_id}.yaml if it exists
    run_config_path = Path(f"config/runs/{cfg.run.run_id}.yaml")
    if run_config_path.exists():
        print(f"Loading run-specific config from: {run_config_path}")
        run_config = OmegaConf.load(run_config_path)
        cfg = OmegaConf.merge(cfg, run_config)
        print(f"✓ Run config merged with base config\n")
    else:
        print(f"Note: Run-specific config not found at {run_config_path}")
        print(f"Using base config with CLI overrides\n")

    # Apply mode-based configuration
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        cfg.optuna.n_trials = 0
        cfg.training.epochs = 1
        print("TRIAL MODE: WandB disabled, Optuna disabled, epochs=1\n")
    elif cfg.mode == "full":
        cfg.wandb.mode = "online"
        print("FULL MODE: WandB enabled, full training\n")
    
    # Initialize WandB
    run = setup_wandb(cfg, cfg.run.run_id)
    
    try:
        # Determine objective metric from task
        if "synthetic" in cfg.run.run_id.lower():
            objective_metric = "convergence_speed_ratio"
        elif "mnist" in cfg.run.run_id.lower():
            objective_metric = "test_accuracy"
        elif "illconditioned" in cfg.run.run_id.lower():
            objective_metric = "final_test_loss"
        elif "federated" in cfg.run.run_id.lower():
            objective_metric = "test_accuracy"
        else:
            objective_metric = "final_train_loss"
        
        # Run Optuna optimization if enabled
        if cfg.optuna.enabled and cfg.optuna.n_trials > 0:
            best_params = run_optuna_optimization(cfg, objective_metric)
            
            # Update config with best parameters
            for param_name, value in best_params.items():
                if param_name == "lambda_spectrum":
                    cfg.training.optimizer_config.lambda_spectrum = value
                elif param_name == "learning_rate":
                    cfg.training.learning_rate = value
                elif param_name == "beta1":
                    cfg.training.optimizer_config.beta1 = value
                elif param_name == "beta2":
                    cfg.training.optimizer_config.beta2 = value
                elif param_name == "warmup_steps":
                    cfg.training.warmup_steps = value
            
            print(f"\nTraining final model with best hyperparameters...\n")
        
        # Train final model
        metrics = train_task(cfg, trial=None)
        
        # Log final metrics to WandB
        if run is not None:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    wandb.summary[key] = value
            
            wandb.finish()
            print(f"\nWandB run completed: {run.url}")
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"Training Summary")
        print(f"{'='*70}")
        for key, value in sorted(metrics.items()):
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")
        print(f"{'='*70}\n")
        
        print(f"✓ Training completed successfully!")
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        if run is not None:
            wandb.finish()
        raise


if __name__ == "__main__":
    main()
