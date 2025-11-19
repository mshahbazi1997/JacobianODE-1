"""
Simple test script for JacobianODE with Lorenz system
Uses CPU to avoid DDP strategy conflicts with MPS
"""

from JacobianODE.jacobians.jacobian_utils import load_config
from JacobianODE.jacobians.run_jacobians import train_jacobians

# Configuration for Lorenz system - using CPU
overrides = [
    # Data configuration
    "data=dysts",
    "data.flow._target_=JacobianODE.dysts_sim.flows.Lorenz",
    "data.postprocessing.obs_noise=0.01",

    # Training configuration
    "training.lightning.loop_closure_weight=0.001",
    "training.logger.save_dir=./output",

    # Use CPU to avoid DDP+MPS incompatibility
    "+training.trainer_params.accelerator=cpu",

    # Reduce epochs for quick testing (default is 1000)
    "training.trainer_params.max_epochs=50",

    # Reduce batch iterations for faster testing
    "training.trainer_params.limit_train_batches=100",
    "training.trainer_params.limit_val_batches=20",
]

print("Loading configuration...")
cfg = load_config(overrides=overrides)

print("Starting training on CPU...")
print("This will take a few minutes. Watch for progress in the terminal.")
train_jacobians(cfg)

print("\n✓ Training complete! Check the ./output directory for results.")
