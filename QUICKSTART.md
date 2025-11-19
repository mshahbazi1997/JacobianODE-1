# Quick Start Guide for JacobianODE

## What You Need to Know

**What does this package do?**
- Learns neural network models of dynamical systems (like the Lorenz attractor)
- Uses Jacobian regularization to improve accuracy
- Can predict long-term behavior of chaotic systems

**What is setup.py?**
- It's a Python packaging file that tells `pip` how to install JacobianODE
- When you run `pip install -e .`, it installs the package in "editable" mode
- This means you can modify the code and changes take effect immediately

## Setup (5 minutes)

### 1. Create and activate conda environment
```bash
conda create -n jacobian python=3.11 -y
conda activate jacobian
```

### 2. Install the package
```bash
cd /Users/mahdiyarshahbazi/Documents/GitHub/JacobianODE
pip install -e .
```

This will install all dependencies from `pyproject.toml` including PyTorch, Lightning, etc.

### 3. Run your first example
```bash
python simple_lorenz_test.py
```

This will:
- Train a neural network to learn the Lorenz system dynamics
- Save results to `./output` directory
- Take ~5-10 minutes on a laptop

## What to Expect

The script will:
1. Generate training data from the Lorenz system
2. Train a neural network to predict the dynamics
3. Validate the model
4. Save checkpoints and logs

**Output location**: `./output/`

## Common Issues

**Issue**: "ddp_notebook strategy not supported on MPS"
- **Solution**: Use the provided `simple_lorenz_test.py` which sets `accelerator=mps`

**Issue**: Slow on Mac
- **Solution**: If MPS is slow, change `accelerator=mps` to `accelerator=cpu` in the script

**Issue**: Wandb errors
- **Solution**: Either create a free account at wandb.ai or disable wandb in config

## Next Steps

After running the simple example:
1. Check `USAGE_GUIDE.md` for more advanced usage
2. Look at the Jupyter notebook: `_jupyter/Lorenz (Demo).ipynb`
3. Try different dynamical systems (Van der Pol, Lorenz96, etc.)

## Parameters You Can Change

In `simple_lorenz_test.py`, you can modify:
- `max_epochs`: Number of training epochs (default: 100)
- `obs_noise`: Observation noise level (default: 0.01)
- `accelerator`: 'mps' for Apple Silicon GPU, 'cpu' for CPU, 'cuda' for NVIDIA GPU
