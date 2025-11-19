# JacobianODE: Learning Dynamical Systems with Loop Closure

A PyTorch Lightning-based framework for learning neural network models of dynamical systems using Jacobian regularization and loop closure constraints.

**Paper**: [Learning Dynamical Systems with Loop Closure](https://arxiv.org/pdf/2507.01946) (NeurIPS)

---

## Table of Contents
1. [Quick Start](#quick-start)
2. [Understanding JacobianODE](#understanding-jacobianode)
3. [Installation](#installation)
4. [Running Your First Example](#running-your-first-example)
5. [Using Your Own Data](#using-your-own-data)
6. [Configuration Guide](#configuration-guide)
7. [Key Hyperparameters](#key-hyperparameters)
8. [Architecture & Pipeline](#architecture--pipeline)
9. [Troubleshooting](#troubleshooting)
10. [Project Status](#project-status)

---

## Quick Start

**5-minute setup to train on Lorenz system:**

```bash
# 1. Install uv package manager (if not installed)
# Follow instructions at: https://github.com/astral-sh/uv

# 2. Create environment and install
sh env_create_uv.sh

# 3. Run simple example
python simple_lorenz_test.py
```

**Output**: Results saved to `./output/` directory

---

## Understanding JacobianODE

### What Does This Package Do?

JacobianODE learns neural network models of dynamical systems (like the Lorenz attractor, neural dynamics, climate models, etc.) using three key innovations:

1. **Jacobian Regularization**: Constrains the learned model's Jacobian (∂f/∂x) to match the true system dynamics
2. **Loop Closure**: Enforces path independence—the system returns to the same state regardless of the path taken
3. **Teacher Forcing Annealing**: Gradually transitions from supervised to autonomous prediction during training

### Why Loop Closure Matters

For physically consistent dynamics, the line integral around any closed loop should be zero:

```
∮ J(x)·dx = 0
```

This constraint ensures the learned dynamics are self-consistent and prevents unrealistic long-term predictions in chaotic systems.

### Key Concepts

- **Input Format**: Time-series trajectories shaped as `(Trials, Time, Dimensions)`
- **Example**: Neural recordings might be `(50 trials, 1000 timesteps, 100 neurons)`
- **Output**: A trained model that can predict future states and estimate Jacobians

---

## Installation

### Prerequisites
- Python 3.11+
- CUDA (optional, for GPU training)

### Method 1: Using UV (Recommended)

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment and install dependencies
sh env_create_uv.sh
```

### Method 2: Using Conda/Pip

```bash
# Create conda environment
conda create -n jacobian python=3.11 -y
conda activate jacobian

# Install package in editable mode
pip install -e .
```

This installs all dependencies from `pyproject.toml` including PyTorch, Lightning, Hydra, WandB, etc.

---

## Running Your First Example

### Simple Lorenz System Test

The quickest way to verify the installation:

```bash
python simple_lorenz_test.py
```

**What this does:**
1. Generates training data from the Lorenz attractor (chaotic 3D system)
2. Trains a neural network to learn the dynamics
3. Uses loop closure regularization for physically consistent predictions
4. Saves checkpoints and logs to `./output/`

**Expected runtime**: 5-10 minutes on a laptop

### Understanding the Script

```python
from JacobianODE.jacobians.jacobian_utils import load_config
from JacobianODE.jacobians.run_jacobians import train_jacobians

# Configure for Lorenz system
overrides = [
    # Data configuration
    "data=dysts",  # Use built-in dynamical systems
    "data.flow._target_=JacobianODE.dysts_sim.flows.Lorenz",
    "data.postprocessing.obs_noise=0.01",  # Add 1% observation noise

    # Training configuration
    "training.lightning.loop_closure_weight=0.001",  # Key hyperparameter!
    "training.logger.save_dir=./output",
]

cfg = load_config(overrides=overrides)
train_jacobians(cfg)
```

### Checking Results

After training completes:

```bash
ls ./output/
# You should see:
# - wandb/  (training logs)
# - checkpoints/  (saved models)
```

To visualize results, see the example notebook: `_jupyter/Lorenz (Demo).ipynb`

---

## Using Your Own Data

### Data Format Requirements

Your data must be shaped as: `(Trials, Time, Dimensions)`

**Example**: Neural recordings
- 50 experimental trials
- 1000 timesteps each
- 100 neurons recorded
- Shape: `(50, 1000, 100)`

**Important**: If your data has shape `(Trials, Neurons, Time)`, transpose the last two axes:

```python
import numpy as np

# Original shape: (50, 100, 1000)
data = your_data_array

# Required shape: (50, 1000, 100)
data_transposed = np.transpose(data, (0, 2, 1))
```

### Custom Data Pipeline (Current Workaround)

**Note from the author**: The data pipeline is currently inflexible and designed for comparing true vs. inferred Jacobians on synthetic systems. Custom data support is a work in progress.

**Workaround approach**:

1. **Skip data generation**: Ignore everything up to "Create Dataloaders" in `run_jacobians.py`
2. **Use your data directly**:

```python
from JacobianODE.jacobians.jacobian_utils import load_config, create_dataloaders
from JacobianODE.jacobians.run_jacobians import train_model, make_model
import torch

# Load your data
# Shape: (trials, time, dimensions)
your_data = torch.load('your_data.pt')  # or np.load, etc.

# Configure for your data dimensions
overrides = [
    "data=base",  # Use base config
    f"data.dim={your_data.shape[2]}",  # Number of dimensions
    "data.train_test_params.seq_length=25",  # Sequence length for training
    "training.lightning.loop_closure_weight=0.01",  # Tune this!
    "training.logger.save_dir=./output",
]

cfg = load_config(overrides=overrides)

# Set data dimensions manually
cfg.data.dim = your_data.shape[2]
cfg.data.input_dim = your_data.shape[2]
cfg.data.output_dim = your_data.shape[2]

# Create dataloaders from your data
train_loader, val_loader, test_loader = create_dataloaders(
    your_data,
    cfg
)

# Initialize model
model, _ = make_model(cfg)

# Train
train_model(cfg, model, train_loader, val_loader)
```

### Key Configuration Changes for Custom Data

Update these config parameters:

```python
overrides = [
    "data=base",
    f"data.dim={n_dimensions}",  # Your data dimensionality

    # Sequence parameters
    "data.train_test_params.seq_length=25",  # Length of training sequences
    "data.train_test_params.train_percent=0.7",  # 70% train, 20% val, 10% test
    "data.train_test_params.test_percent=0.1",

    # Model architecture
    "model=mlp",  # or "model=neuralode"
    "model.params.hidden_dim=[256,1024,2048,2048]",  # Adjust for your data

    # Training
    "training.lightning.loop_closure_weight=0.01",  # CRITICAL: tune this
    "training.trainer_params.max_epochs=1000",
    "training.batch_size=16",
]
```

---

## Configuration Guide

JacobianODE uses [Hydra](https://hydra.cc/) for configuration management. All config files are in `JacobianODE/jacobians/conf/`.

### Configuration File Structure

```
JacobianODE/jacobians/conf/
├── config.yaml              # Main config
├── data/
│   ├── dysts.yaml          # Built-in dynamical systems
│   └── base.yaml           # Base data parameters
├── model/
│   ├── mlp.yaml            # Multi-layer perceptron
│   └── neuralode.yaml      # Neural ODE
└── training/
    └── training.yaml       # Training hyperparameters
```

### Data Configuration (`data/dysts.yaml`)

```yaml
# Dynamical system selection
flow:
  _target_: JacobianODE.dysts_sim.flows.Lorenz
  # Options: Lorenz, VanDerPol, Lorenz96, LotkaVolterra, Arneodo

# Trajectory generation
trajectory_params:
  n_periods: 12              # Number of oscillation cycles
  pts_per_period: 100        # Sampling frequency
  num_ics: 32                # Number of initial conditions (trials)
  method: Radau              # ODE solver: RK45, Radau, BDF, DOP853

# Data postprocessing
postprocessing:
  obs_noise: 0.05            # Observation noise (0.0-1.0)
  filter_data: False         # Apply Butterworth filter
  normalize: False           # Normalize to unit variance
```

### Model Configuration (`model/mlp.yaml`)

```yaml
params:
  _target_: JacobianODE.models.mlp.MLP
  hidden_dim: [256, 1024, 2048, 2048]  # Hidden layer sizes
  num_layers: 4
  residuals: False           # Use residual connections
  dropout: 0.0               # Dropout rate
  activation: 'silu'         # Activation: relu, silu, tanh, gelu
```

### Training Configuration (`training/training.yaml`)

See [Key Hyperparameters](#key-hyperparameters) section below.

### Overriding Configurations

**Method 1: Command-line style (in Python)**

```python
overrides = [
    "data=dysts",
    "data.flow._target_=JacobianODE.dysts_sim.flows.Lorenz",
    "data.postprocessing.obs_noise=0.01",
    "training.lightning.loop_closure_weight=0.001",
    "model.params.hidden_dim=[512,1024,1024]",
]
cfg = load_config(overrides=overrides)
```

**Method 2: Modify YAML files directly**

Edit files in `JacobianODE/jacobians/conf/` and run:

```python
cfg = load_config()  # Uses default configs
```

---

## Key Hyperparameters

### Critical Parameter: Loop Closure Weight

**Most important hyperparameter** that requires tuning for each dataset.

```python
"training.lightning.loop_closure_weight=0.001"
```

**Tuning strategy** (from paper supplementary materials):
1. Test multiple values: `[0.0, 0.0001, 0.001, 0.01, 0.1, 1.0]`
2. For each value, train and evaluate on validation set
3. Select weight that balances:
   - Trajectory prediction accuracy (MSE on validation)
   - Physical plausibility (loop closure loss near zero)
4. **Warning**: Best validation accuracy alone is NOT sufficient—check loop closure loss!

**Typical ranges**:
- Simple systems (Lorenz): `0.001 - 0.01`
- High-dimensional systems: `0.0001 - 0.001`
- Very noisy data: `0.01 - 0.1`

### Other Important Hyperparameters

| Parameter | Default | Description | When to Change |
|-----------|---------|-------------|----------------|
| `obs_noise` | 0.05 | Observation noise level | Match your data's noise level |
| `seq_length` | 25 | Training sequence length | Longer for slower dynamics |
| `batch_size` | 16 | Batch size | Increase for more stable training |
| `max_epochs` | 1000 | Maximum epochs | Decrease for quick tests |
| `lr` | 1e-4 | Learning rate | Decrease if unstable |
| `alpha_teacher_forcing` | 1.0 | Initial teacher forcing | Usually keep at 1.0 |
| `gamma_teacher_forcing` | 0.999 | TF decay rate | Slower decay for harder tasks |
| `hidden_dim` | [256,1024,2048,2048] | Network size | Scale with data complexity |

### Teacher Forcing Annealing

Teacher forcing helps training stability:
- **Early training**: Model uses true previous states (α = 1.0)
- **Late training**: Model uses its own predictions (α → 0)
- **Annealing**: α decays by `gamma_teacher_forcing` every few epochs

```python
# Default settings (usually work well)
"training.lightning.alpha_teacher_forcing=1.0",
"training.lightning.gamma_teacher_forcing=0.999",
"training.lightning.teacher_forcing_annealing=True",
"training.lightning.min_alpha_teacher_forcing=0.0",
```

---

## Architecture & Pipeline

### Training Pipeline Overview

```
1. Data Generation/Loading
   ↓
2. Postprocessing (noise, filtering, normalization)
   ↓
3. Create Sequences (sliding windows)
   ↓
4. DataLoader Creation (train/val/test split)
   ↓
5. Model Initialization (MLP or NeuralODE)
   ↓
6. Training Loop
   - Forward pass with teacher forcing
   - Compute losses (trajectory + loop closure + Jacobian)
   - Backpropagation
   - Update teacher forcing coefficient
   ↓
7. Save Best Model & Logs
```

### Loss Function Components

**Total Loss**:
```
L_total = L_trajectory + λ_loop × L_loop_closure + λ_jac × L_jacobian
```

**Components**:
1. **Trajectory Loss**: MSE between predicted and true trajectories
2. **Loop Closure Loss**: Path integral around random loops ∮ J·dx
3. **Jacobian Penalty**: Regularization on Jacobian magnitude (optional)

### Model Architectures

**MLP (Multi-Layer Perceptron)**:
- Direct mapping: x(t) → x(t+1)
- Jacobian via automatic differentiation or direct output
- Default: 4 layers with [256, 1024, 2048, 2048] units
- ~6.6M parameters for 3D systems

**NeuralODE**:
- Learns derivative: dx/dt = f(x)
- Integration via `torchdiffeq` (RK4, Dopri5, etc.)
- Better for systems where dt varies
- Slightly slower but more physically interpretable

### Built-in Dynamical Systems

Located in `JacobianODE/dysts_sim/flows.py`:

- **Lorenz**: Classic 3D chaotic attractor
- **VanDerPol**: 2D oscillator
- **Lorenz96**: High-dimensional chaos (N=40, 96, etc.)
- **LotkaVolterra**: Predator-prey dynamics
- **Arneodo**: 3D chaotic system

Each system provides:
- `_rhs()`: Right-hand side of ODE (dx/dt = f(x))
- `_jac()`: True Jacobian matrix (∂f/∂x)

---

## Troubleshooting

### Installation Issues

**Problem**: `uv` command not found
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
# Restart terminal
```

**Problem**: CUDA version mismatch
```bash
# Check CUDA version
nvidia-smi

# Install compatible PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Training Issues

**Problem**: Loss explodes (NaN)
- **Solution 1**: Reduce learning rate: `training.lightning.optimizer_kwargs.lr=1e-5`
- **Solution 2**: Reduce loop closure weight: `training.lightning.loop_closure_weight=0.0001`
- **Solution 3**: Enable gradient clipping (default: enabled at 1.0)

**Problem**: Training very slow
- **Solution 1**: Reduce batch size or sequence length
- **Solution 2**: Use smaller network: `model.params.hidden_dim=[128,512,512]`
- **Solution 3**: Limit batches per epoch: `training.trainer_params.limit_train_batches=100`

**Problem**: Model doesn't improve
- **Solution 1**: Increase teacher forcing decay: `gamma_teacher_forcing=0.9995`
- **Solution 2**: Add more training data (increase `num_ics`)
- **Solution 3**: Reduce observation noise if too high

**Problem**: Good training loss, bad validation loss
- **Solution 1**: Add regularization: `l2_penalty=1e-4`
- **Solution 2**: Add dropout: `model.params.dropout=0.1`
- **Solution 3**: Reduce model size

### WandB Issues

**Problem**: WandB login required
```python
# Option 1: Login to WandB
import wandb
wandb.login()

# Option 2: Disable WandB
# In config.yaml, set:
# logger: None
```

### Data Format Issues

**Problem**: "Expected 3D tensor, got 2D"
- **Solution**: Ensure data is `(Trials, Time, Dims)`, not `(Time, Dims)`
- **Fix**: Add trial dimension: `data = data.unsqueeze(0)`

**Problem**: "Trials dimension is wrong"
- **Solution**: Check if axes are `(Trials, Neurons, Time)` instead of `(Trials, Time, Neurons)`
- **Fix**: Transpose: `data = data.transpose(1, 2)`

---

## Project Status

**Current State** (as of November 2024):

> "The code is unfortunately not in a particularly usable state right now, let alone easily understandable. I took out the training infrastructure code for my NeurIPS paper, put it in this repo, and have been gradually trying to engineer it to be broadly usable. It is very much a work in progress."
>
> — Adam, First Author

### Known Limitations

1. **Data Pipeline**: Currently inflexible and designed for synthetic systems
2. **Documentation**: Sparse and incomplete
3. **Custom Data**: Requires manual workarounds (see [Using Your Own Data](#using-your-own-data))
4. **Hyperparameter Selection**: No automated tuning for loop closure weight

### Upcoming Improvements

- Flexible data pipeline for arbitrary datasets
- Automated hyperparameter selection
- Better documentation and examples
- More built-in visualization tools

### Contributing

If you develop useful data infrastructure or improvements, pull requests are welcome!

---

## Example: Minimal Custom Data Script

```python
"""
Train JacobianODE on custom neural data
Expected data shape: (Trials, Time, Neurons)
"""

import torch
import numpy as np
from JacobianODE.jacobians.jacobian_utils import load_config, create_dataloaders
from JacobianODE.jacobians.run_jacobians import train_model, make_model

# 1. Load your data
# Example: (50 trials, 1000 timesteps, 100 neurons)
data = np.load('your_neural_data.npy')
data = torch.from_numpy(data).float()

# Ensure correct shape: (Trials, Time, Dimensions)
print(f"Data shape: {data.shape}")
assert len(data.shape) == 3, "Data must be 3D: (Trials, Time, Dims)"

# 2. Configure for your data
n_dims = data.shape[2]

overrides = [
    "data=base",
    f"data.dim={n_dims}",
    "data.train_test_params.seq_length=25",
    "training.lightning.loop_closure_weight=0.01",  # Tune this!
    "training.logger.save_dir=./output",
    "training.trainer_params.max_epochs=500",
]

cfg = load_config(overrides=overrides)

# Manually set dimensions
cfg.data.dim = n_dims
cfg.data.input_dim = n_dims
cfg.data.output_dim = n_dims

# 3. Create dataloaders
train_loader, val_loader, test_loader = create_dataloaders(data, cfg)

# 4. Initialize model
model, _ = make_model(cfg)

# 5. Train
train_model(cfg, model, train_loader, val_loader)

print("Training complete! Check ./output/ for results")
```

---

## Visualization Example

After training, visualize predictions:

```python
import matplotlib.pyplot as plt
import torch

# Load trained model
from JacobianODE.jacobians.jacobian_utils import load_run

checkpoint_path = './output/checkpoints/best_model.ckpt'
model = torch.load(checkpoint_path)
model.eval()

# Get test data
test_batch = next(iter(test_loader))
test_sequence = test_batch['sequence']  # (batch, seq_len, dims)

# Predict
with torch.no_grad():
    predictions = model(test_sequence)

# Plot first trajectory
fig, axes = plt.subplots(3, 1, figsize=(10, 8))
for dim in range(3):
    axes[dim].plot(test_sequence[0, :, dim], label='True', alpha=0.7)
    axes[dim].plot(predictions[0, :, dim], label='Predicted', alpha=0.7)
    axes[dim].set_ylabel(f'Dim {dim}')
    axes[dim].legend()
axes[-1].set_xlabel('Time')
plt.tight_layout()
plt.savefig('prediction_comparison.png')
```

---

## Citation

If you use this code, please cite the paper:

```bibtex
@inproceedings{jacobianode2024,
  title={Learning Dynamical Systems with Loop Closure},
  author={Eisenmann, Adam and others},
  booktitle={NeurIPS},
  year={2024},
  url={https://arxiv.org/pdf/2507.01946}
}
```

---

## Support

**Questions about the code?**
- Open an issue on GitHub
- Check the example notebook: `_jupyter/Lorenz (Demo).ipynb`

**For research discussions**:
- Contact the first author (Adam Eisenmann)
- See paper for contact information

---

## License

[Check LICENSE file in repository]
