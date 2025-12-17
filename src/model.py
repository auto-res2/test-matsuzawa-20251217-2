"""Model architectures for SDSLR experiments."""

import torch
import torch.nn as nn
from omegaconf import DictConfig


def get_model(cfg: DictConfig) -> nn.Module:
    """
    Factory function to create model based on configuration.

    Args:
        cfg: Hydra configuration with model specification

    Returns:
        PyTorch model instance
    """

    model_name = cfg.model.name if hasattr(cfg.model, 'name') else "synthetic-linear-regression"

    if "linear-regression" in model_name.lower():
        dim = cfg.model.dimension if hasattr(cfg.model, 'dimension') else 1000
        return LinearRegressionModel(dim)

    elif "logistic-regression" in model_name.lower() or "classification" in model_name.lower():
        input_dim = cfg.model.input_dimension if hasattr(cfg.model, 'input_dimension') else 784
        return LogisticRegressionModel(input_dim)

    elif "least-squares" in model_name.lower():
        dim = cfg.model.dimension if hasattr(cfg.model, 'dimension') else 2000
        return LinearRegressionModel(dim)

    else:
        # Default: linear regression
        dim = cfg.model.dimension if hasattr(cfg.model, 'dimension') else 1000
        return LinearRegressionModel(dim)


class LinearRegressionModel(nn.Module):
    """
    Linear regression model for synthetic experiments.
    
    Maps d-dimensional input to scalar output: y = w^T x + b
    """
    
    def __init__(self, input_dim: int, output_dim: int = 1, bias: bool = False):
        """
        Args:
            input_dim: Input feature dimension
            output_dim: Output dimension (default 1 for regression)
            bias: Whether to include bias term
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        
        # Initialize weights to small values
        with torch.no_grad():
            self.linear.weight.normal_(0, 0.01)
            if bias and self.linear.bias is not None:
                self.linear.bias.fill_(0.0)
        
        # POST-INIT ASSERTION
        assert self.linear.weight.shape == (output_dim, input_dim), \
            f"Weight shape mismatch: {self.linear.weight.shape}"
        assert self.linear.out_features == output_dim, \
            f"Output dimension mismatch: {self.linear.out_features} vs {output_dim}"
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with input dimension validation."""
        assert x.shape[1] == self.linear.in_features, \
            f"Input feature dimension mismatch: {x.shape[1]} vs {self.linear.in_features}"
        return self.linear(x)


class LogisticRegressionModel(nn.Module):
    """
    Logistic regression model for binary classification.
    
    Maps d-dimensional input to probability: σ(w^T x + b)
    """
    
    def __init__(self, input_dim: int, bias: bool = False):
        """
        Args:
            input_dim: Input feature dimension
            bias: Whether to include bias term
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, bias=bias)
        self.sigmoid = nn.Sigmoid()
        
        # Initialize weights
        with torch.no_grad():
            self.linear.weight.normal_(0, 0.01)
            if bias and self.linear.bias is not None:
                self.linear.bias.fill_(0.0)
        
        # POST-INIT ASSERTION
        assert self.linear.out_features == 1, \
            f"Output dimension should be 1 for binary classification, got {self.linear.out_features}"
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: returns probability with input dimension validation."""
        assert x.shape[1] == self.linear.in_features, \
            f"Input feature dimension mismatch: {x.shape[1]} vs {self.linear.in_features}"
        logits = self.linear(x)
        return self.sigmoid(logits)
