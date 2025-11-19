# JacobianODE Usage Guide

## Overview
JacobianODE is a framework for learning neural representations of dynamical systems using Jacobian-regularized training. It's designed to learn accurate long-term predictions of chaotic and complex dynamical systems.

## What This Code Does
- **Learns dynamics from data**: Takes time series data from dynamical systems and learns a neural network model that can predict future states
- **Jacobian regularization**: Uses the system's Jacobian (derivatives) to improve learning and ensure the model captures the true dynamics
- **Handles various systems**: Works with classical systems (Lorenz, Van der Pol) and custom ODEs

## Quick Start

### 1. Basic Usage (Minimal Example)
```python
from JacobianODE.jacobians.jacobian_utils import load_config
from JacobianODE.jacobians.run_jacobians import train_jacobians

# Configure for Lorenz system
overrides = [
    "data=dysts",
    "data.flow._target_=JacobianODE.dysts_sim.flows.Lorenz",
    "data.postprocessing.obs_noise=0.01",  # Add 1% observation noise
    "training.lightning.loop_closure_weight=0.001",
    "training.logger.save_dir=./output",
    "wandb_entity=your_wandb_entity"  # Optional: for experiment tracking
]

# Load config and train
cfg = load_config(overrides=overrides)
train_jacobians(cfg)
```

### 2. Using Different Dynamical Systems

#### Available Built-in Systems:
- `Lorenz` - Classic chaotic attractor
- `VanDerPol` - Oscillator system
- `Lorenz96` - High-dimensional atmospheric model
- `LotkaVolterra` - Predator-prey dynamics
- `Arneodo` - 3D chaotic system

```python
# Example: Van der Pol oscillator
overrides = [
    "data=dysts",
    "data.flow._target_=JacobianODE.dysts_sim.flows.VanDerPol",
    "data.flow.mu=1.0",  # System parameter
    "training.logger.save_dir=./output"
]
cfg = load_config(overrides=overrides)
train_jacobians(cfg)
```

### 3. Custom Dynamical System

Create your own system by extending the `DynSys` base class:

```python
# custom_system.py
import numpy as np
from JacobianODE.dysts_sim.base import DynSys

class MyCustomSystem(DynSys):
    def __init__(self, param1=1.0, param2=2.0):
        super().__init__(dim=3)  # Set dimension
        self.param1 = param1
        self.param2 = param2

    def _rhs(self, t, x):
        """Define the right-hand side of your ODE: dx/dt = f(x,t)"""
        x1, x2, x3 = x
        dx1dt = -self.param1 * x1 + x2
        dx2dt = -x1 * x3
        dx3dt = self.param2 * (x2 - x3)
        return np.array([dx1dt, dx2dt, dx3dt])

    def _jac(self, t, x):
        """Define the Jacobian matrix (optional but recommended)"""
        x1, x2, x3 = x
        return np.array([
            [-self.param1, 1, 0],
            [-x3, 0, -x1],
            [0, self.param2, -self.param2]
        ])
```

Use your custom system:
```python
overrides = [
    "data=custom",
    "data.flow._target_=custom_system.MyCustomSystem",
    "data.flow.param1=1.5",
    "data.flow.param2=2.5",
    "training.logger.save_dir=./output"
]
cfg = load_config(overrides=overrides)
train_jacobians(cfg)
```

### 4. Key Configuration Options

```python
overrides = [
    # Data Generation
    "data.trajectory_params.n_periods=12",        # Number of periods to simulate
    "data.trajectory_params.pts_per_period=100",  # Points per period
    "data.trajectory_params.num_ics=32",          # Number of initial conditions

    # Model Architecture
    "model.params.hidden_dim=[256,1024,2048]",    # MLP hidden layers
    "model.params.activation=gelu",               # Activation function
    "model.params.dropout=0.1",                   # Dropout rate

    # Training
    "training.lightning.batch_size=16",
    "training.lightning.max_epochs=1000",
    "training.lightning.lr=1e-3",
    "training.lightning.teacher_forcing_annealing=True",  # Gradually reduce teacher forcing

    # Noise and Preprocessing
    "data.postprocessing.obs_noise=0.01",         # 1% observation noise
    "data.postprocessing.filter_type=butterworth", # Smoothing filter

    # Loss Functions
    "training.lightning.loss_func=mse",           # Options: mse, horizon, soft_dtw
    "training.lightning.jacobian_penalty=0.01",   # Jacobian regularization weight
]
```

### 5. Working with Your Own Data

If you have time series data from a real system:

```python
import numpy as np
from JacobianODE.jacobians.data_utils import embed_signal_torch
import torch

# Your time series data
data = np.load("my_timeseries.npy")  # Shape: (time_steps, dimensions)

# Create delay embeddings if needed (for single time series)
embedded_data = embed_signal_torch(
    torch.tensor(data),
    embedding_dim=3,  # Number of delays
    delay=10          # Delay between embeddings
)

# Then use custom data loading in config
overrides = [
    "data=custom",
    "data.custom_data_path=my_timeseries.npy",
    "training.logger.save_dir=./output"
]
```

### 6. After Training - Using the Model

```python
import torch
from JacobianODE.models.mlp import MLP

# Load trained model
model = MLP(input_dim=3, output_dim=3, hidden_dim=[256, 1024, 2048])
model.load_state_dict(torch.load("./output/model_checkpoint.pt"))
model.eval()

# Predict next state
current_state = torch.tensor([1.0, 2.0, 3.0])
with torch.no_grad():
    next_state = model(current_state.unsqueeze(0))
    print(f"Next state: {next_state.squeeze()}")
```

### 7. Analyzing Results

The training produces:
- **Model checkpoints**: Saved neural network weights
- **WandB logs** (if configured): Training curves, loss metrics
- **Trajectory predictions**: Visualizations of learned dynamics

## Common Use Cases

### Case 1: Learning Unknown Dynamics
You have measurement data from a system but don't know the equations:
```python
overrides = [
    "data=custom",
    "data.custom_data_path=measurements.npy",
    "model.params.hidden_dim=[512,1024,1024,512]",  # Larger model for unknown dynamics
    "training.lightning.jacobian_penalty=0.1",       # Strong regularization
]
```

### Case 2: System Identification
You want to identify parameters of a known system:
```python
# Train on noisy observations
overrides = [
    "data=dysts",
    "data.flow._target_=JacobianODE.dysts_sim.flows.Lorenz",
    "data.postprocessing.obs_noise=0.05",  # 5% noise
    "training.lightning.loss_func=horizon",  # Better for noisy data
]
```

### Case 3: Control Applications
Learn dynamics for control design:
```python
overrides = [
    "data=dysts",
    "data.flow._target_=JacobianODE.dysts_sim.flows.VanDerPol",
    "training.lightning.loop_closure_weight=0.01",  # Ensure stable cycles
    "training.lightning.jacobian_penalty=0.05",     # Accurate derivatives for control
]
```

## Tips for Best Results

1. **Start simple**: Begin with known systems (Lorenz, Van der Pol) to verify setup
2. **Data quality**: More trajectories with different initial conditions improve generalization
3. **Teacher forcing**: Enable annealing for better long-term predictions
4. **Noise handling**: Add observation noise during training for robustness
5. **Model size**: Start with default architecture, increase if underfitting
6. **Jacobian penalty**: Higher values (0.01-0.1) for more accurate dynamics

## Troubleshooting

- **Training diverges**: Reduce learning rate, increase batch size
- **Poor long-term predictions**: Enable teacher forcing annealing, increase trajectory length
- **Overfitting**: Add dropout, reduce model size, add noise
- **Slow training**: Reduce model size, use smaller batch size

## References
- Paper: "Probing Dynamical Systems with Deep Jacobian Estimation" (arXiv:2507.01946)
- Based on neural ODE and Jacobian regularization techniques for learning dynamical systems