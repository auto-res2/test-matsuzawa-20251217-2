"""Data preprocessing and generation pipeline."""

import numpy as np
import torch
from omegaconf import DictConfig
from typing import Tuple


def get_dataset(cfg: DictConfig) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load and preprocess dataset based on configuration.

    Args:
        cfg: Hydra configuration with dataset specification

    Returns:
        Tuple of (X_train, y_train, X_test, y_test) as torch tensors
    """

    dataset_name = cfg.dataset.name if hasattr(cfg.dataset, 'name') else "synthetic"

    if "power-law" in dataset_name.lower():
        return generate_power_law_spectrum_data(cfg)
    elif "mnist" in dataset_name.lower():
        return generate_mnist_binary_data(cfg)
    elif "ill-conditioned" in dataset_name.lower():
        return generate_illconditioned_data(cfg)
    elif "federated" in dataset_name.lower() or "cifar" in dataset_name.lower():
        return generate_federated_data(cfg)
    else:
        return generate_power_law_spectrum_data(cfg)


def generate_power_law_spectrum_data(cfg: DictConfig) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate synthetic data with power-law eigenvalue spectrum."""
    
    seed = cfg.training.seed if hasattr(cfg.training, 'seed') else 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Extract parameters
    d = cfg.model.dimension if hasattr(cfg.model, 'dimension') else 1000
    n = cfg.dataset.n_samples if hasattr(cfg.dataset, 'n_samples') else 1000
    n_test = cfg.dataset.n_test_samples if hasattr(cfg.dataset, 'n_test_samples') else 200
    
    # Extract spectrum parameter (use first variant if available)
    if hasattr(cfg.dataset, 'spectrum_variants') and cfg.dataset.spectrum_variants:
        variant = cfg.dataset.spectrum_variants[0]
        if isinstance(variant, dict):
            alpha = variant["alpha"]
        else:
            alpha = variant.alpha if hasattr(variant, 'alpha') else 1.0
    else:
        alpha = 1.0
    
    noise_level = cfg.dataset.noise_level if hasattr(cfg.dataset, 'noise_level') else 0.01
    
    # POST-INIT ASSERTIONS for data generation
    assert d > 0, f"Dimension must be positive: {d}"
    assert n > 0, f"Number of samples must be positive: {n}"
    assert 0 <= alpha <= 3, f"Spectrum exponent should be in [0, 3]: {alpha}"
    assert 0 <= noise_level <= 1.0, f"Noise level should be in [0, 1]: {noise_level}"
    
    # Generate power-law eigenvalues: λᵢ ∝ i^(-α)
    eigenvalues = np.array([max(i**(-alpha), 1e-10) for i in range(1, d+1)], dtype=np.float32)
    eigenvalues /= np.sum(eigenvalues)
    
    # Generate covariance matrix and its Cholesky decomposition
    cov_matrix = np.diag(eigenvalues) + 1e-6 * np.eye(d)
    try:
        L = np.linalg.cholesky(cov_matrix)
    except np.linalg.LinAlgError:
        # If Cholesky fails, use eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
        eigvals = np.maximum(eigvals, 1e-10)
        L = eigvecs @ np.diag(np.sqrt(eigvals))
    
    # Generate data: X ~ N(0, Σ) with power-law spectrum
    X = np.random.randn(n + n_test, d).astype(np.float32) @ L.T
    
    # Generate labels: y = X @ w_true + noise
    w_true = np.random.randn(d).astype(np.float32) * 0.1
    epsilon = np.random.randn(n + n_test).astype(np.float32) * noise_level
    y = (X @ w_true + epsilon).reshape(-1, 1).astype(np.float32)
    
    # Split into train/test (held-out test set semantics - independent samples)
    X_train = torch.from_numpy(X[:n]).float()
    y_train = torch.from_numpy(y[:n]).float().squeeze()
    X_test = torch.from_numpy(X[n:n+n_test]).float()
    y_test = torch.from_numpy(y[n:n+n_test]).float().squeeze()
    
    # POST-GENERATION ASSERTIONS
    assert X_train.shape[0] == n, f"Train set size mismatch: {X_train.shape[0]} vs {n}"
    assert X_test.shape[0] == n_test, f"Test set size mismatch: {X_test.shape[0]} vs {n_test}"
    assert X_train.shape[1] == d, f"Feature dimension mismatch: {X_train.shape[1]} vs {d}"
    assert y_train.shape[0] == n, f"Train labels size mismatch: {y_train.shape[0]} vs {n}"
    assert y_test.shape[0] == n_test, f"Test labels size mismatch: {y_test.shape[0]} vs {n_test}"
    # Verify test set is independent (different from training)
    assert not torch.allclose(X_train[:min(10, n_test)], X_test[:min(10, n_test)]), \
        "Test set should be independent from training set"
    
    return X_train, y_train, X_test, y_test


def generate_mnist_binary_data(cfg: DictConfig) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate synthetic MNIST-like binary classification data with held-out test set."""
    
    seed = cfg.training.seed if hasattr(cfg.training, 'seed') else 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    n_train = cfg.dataset.n_train_samples if hasattr(cfg.dataset, 'n_train_samples') else 1000
    n_test = cfg.dataset.n_test_samples if hasattr(cfg.dataset, 'n_test_samples') else 500
    d = cfg.model.input_dimension if hasattr(cfg.model, 'input_dimension') else 784
    
    # POST-INIT ASSERTIONS
    assert n_train > 0, f"Number of training samples must be positive: {n_train}"
    assert n_test > 0, f"Number of test samples must be positive: {n_test}"
    assert d > 0, f"Feature dimension must be positive: {d}"
    
    # Generate true weight vector (shared for train and test generation)
    w_true = torch.randn(d, dtype=torch.float32) * 0.1
    
    # Generate training data: separate random features and labels based on w_true
    X_train = torch.randn(n_train, d, dtype=torch.float32)
    noise_train = torch.randn(n_train, dtype=torch.float32) * 0.5
    y_train = (X_train @ w_true + noise_train > 0).float()
    
    # Generate test data: HELD-OUT test set with separate random features
    X_test = torch.randn(n_test, d, dtype=torch.float32)
    noise_test = torch.randn(n_test, dtype=torch.float32) * 0.5
    y_test = (X_test @ w_true + noise_test > 0).float()
    
    # POST-GENERATION ASSERTIONS
    assert X_train.shape == (n_train, d), f"Train features shape mismatch: {X_train.shape}"
    assert y_train.shape == (n_train,), f"Train labels shape mismatch: {y_train.shape}"
    assert X_test.shape == (n_test, d), f"Test features shape mismatch: {X_test.shape}"
    assert y_test.shape == (n_test,), f"Test labels shape mismatch: {y_test.shape}"
    assert y_train.min() >= 0 and y_train.max() <= 1, "Train labels should be in [0, 1]"
    assert y_test.min() >= 0 and y_test.max() <= 1, "Test labels should be in [0, 1]"
    # Verify test set is independent (different from training)
    assert not torch.allclose(X_train[:min(10, n_test)], X_test[:min(10, n_test)]), \
        "Test set should be independent from training set"
    
    return X_train, y_train, X_test, y_test


def generate_illconditioned_data(cfg: DictConfig) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate ill-conditioned least squares problem with specified condition number."""
    
    seed = cfg.training.seed if hasattr(cfg.training, 'seed') else 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    d = cfg.model.dimension if hasattr(cfg.model, 'dimension') else 2000
    n = cfg.model.n_samples if hasattr(cfg.model, 'n_samples') else 500
    kappa = cfg.model.condition_number if hasattr(cfg.model, 'condition_number') else 1000
    noise_level = cfg.dataset.noise_level if hasattr(cfg.dataset, 'noise_level') else 0.01
    
    # POST-INIT ASSERTIONS
    assert d > 0, f"Dimension must be positive: {d}"
    assert n > 0, f"Number of samples must be positive: {n}"
    assert kappa >= 1, f"Condition number must be >= 1: {kappa}"
    assert 0 <= noise_level <= 1.0, f"Noise level should be in [0, 1]: {noise_level}"
    
    # Generate eigenvalues with condition number κ
    eigenvalues = np.array([kappa**(-(i-1)/(d-1)) for i in range(1, d+1)], dtype=np.float32)
    
    # Generate orthogonal matrices via QR decomposition
    Q_temp = np.random.randn(d, d)
    Q, _ = np.linalg.qr(Q_temp)
    
    V_temp = np.random.randn(d, d)
    V, _ = np.linalg.qr(V_temp)
    
    # Construct ill-conditioned matrix: A = Q Σ V^T
    Sigma = np.diag(np.sqrt(eigenvalues))
    A = Q @ Sigma @ V.T
    A = A.astype(np.float32)
    
    # Verify condition number with tolerance assertion
    actual_kappa = np.linalg.cond(A)
    tolerance = 0.1 * kappa
    assert actual_kappa > kappa - tolerance, \
        f"Matrix condition number too small: {actual_kappa:.2e} vs target {kappa:.2e}"
    assert actual_kappa < kappa + tolerance, \
        f"Matrix condition number too large: {actual_kappa:.2e} vs target {kappa:.2e}"
    print(f"✓ Matrix condition number: {actual_kappa:.2e} (target: {kappa:.2e})")
    
    # Generate problem: minimize ||Ax - b||²
    x_true = np.random.randn(d).astype(np.float32) * 0.1
    b = A @ x_true + np.random.randn(n).astype(np.float32) * noise_level
    
    # Convert to torch
    A_tensor = torch.from_numpy(A).float()
    b_tensor = torch.from_numpy(b.reshape(-1, 1)).float().squeeze()
    
    # POST-GENERATION ASSERTIONS
    assert A_tensor.shape == (d, d), f"Matrix shape mismatch: {A_tensor.shape}"
    assert b_tensor.shape == (n,), f"Vector shape mismatch: {b_tensor.shape}"
    assert not torch.isnan(A_tensor).any(), "NaN in matrix A"
    assert not torch.isnan(b_tensor).any(), "NaN in vector b"
    
    # For least squares, use same data for validation (problem-specific)
    return A_tensor, b_tensor, A_tensor, b_tensor


def generate_federated_data(cfg: DictConfig) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate synthetic federated learning data simulating distributed clients."""

    seed = cfg.training.seed if hasattr(cfg.training, 'seed') else 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Extract federated parameters
    n_clients = cfg.dataset.n_clients if hasattr(cfg.dataset, 'n_clients') else 10
    n_local_samples = cfg.dataset.n_local_samples_per_client if hasattr(cfg.dataset, 'n_local_samples_per_client') else 5000
    n_test = cfg.dataset.n_test_samples if hasattr(cfg.dataset, 'n_test_samples') else 10000
    d = cfg.model.input_dimension if hasattr(cfg.model, 'input_dimension') else 5000
    non_iid = cfg.dataset.non_iid if hasattr(cfg.dataset, 'non_iid') else False
    data_heterogeneity = cfg.dataset.data_heterogeneity_factor if hasattr(cfg.dataset, 'data_heterogeneity_factor') else 0.3

    # POST-INIT ASSERTIONS
    assert n_clients > 0, f"Number of clients must be positive: {n_clients}"
    assert n_local_samples > 0, f"Local samples per client must be positive: {n_local_samples}"
    assert n_test > 0, f"Number of test samples must be positive: {n_test}"
    assert d > 0, f"Feature dimension must be positive: {d}"
    assert 0 <= data_heterogeneity <= 1.0, f"Data heterogeneity should be in [0, 1]: {data_heterogeneity}"

    # Generate global true weight vector
    w_true = torch.randn(d, dtype=torch.float32) * 0.1

    # Generate data for all clients
    X_train_list = []
    y_train_list = []

    for client_id in range(n_clients):
        # Generate features with potential heterogeneity
        if non_iid:
            # Non-IID: Each client has slightly different data distribution
            client_shift = torch.randn(d, dtype=torch.float32) * data_heterogeneity
            X_client = torch.randn(n_local_samples, d, dtype=torch.float32) + client_shift
        else:
            # IID: All clients have same distribution
            X_client = torch.randn(n_local_samples, d, dtype=torch.float32)

        # Generate labels based on global model with noise
        noise_client = torch.randn(n_local_samples, dtype=torch.float32) * 0.5
        y_client = (X_client @ w_true + noise_client > 0).float()

        X_train_list.append(X_client)
        y_train_list.append(y_client)

    # Concatenate all client data into single training set
    X_train = torch.cat(X_train_list, dim=0)
    y_train = torch.cat(y_train_list, dim=0)

    # Generate held-out test data (centralized)
    X_test = torch.randn(n_test, d, dtype=torch.float32)
    noise_test = torch.randn(n_test, dtype=torch.float32) * 0.5
    y_test = (X_test @ w_true + noise_test > 0).float()

    # POST-GENERATION ASSERTIONS
    expected_train_size = n_clients * n_local_samples
    assert X_train.shape == (expected_train_size, d), f"Train features shape mismatch: {X_train.shape}"
    assert y_train.shape == (expected_train_size,), f"Train labels shape mismatch: {y_train.shape}"
    assert X_test.shape == (n_test, d), f"Test features shape mismatch: {X_test.shape}"
    assert y_test.shape == (n_test,), f"Test labels shape mismatch: {y_test.shape}"
    assert y_train.min() >= 0 and y_train.max() <= 1, "Train labels should be in [0, 1]"
    assert y_test.min() >= 0 and y_test.max() <= 1, "Test labels should be in [0, 1]"

    return X_train, y_train, X_test, y_test
