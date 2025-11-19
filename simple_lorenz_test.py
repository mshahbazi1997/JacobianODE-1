"""
Simple test script for JacobianODE with Lorenz system
This script is configured to work on Mac (MPS accelerator)
"""

from JacobianODE.jacobians.jacobian_utils import load_config
from JacobianODE.jacobians.run_jacobians import train_jacobians

# Configuration for Lorenz system optimized for Mac
overrides = [
    # Data configuration
    "data=dysts",
    "data.flow._target_=JacobianODE.dysts_sim.flows.Lorenz",
    "data.postprocessing.obs_noise=0.01",

    # Training configuration
    "training.lightning.loop_closure_weight=0.001",
    "training.logger.save_dir=./output",

    # Optional: disable wandb if you don't have an account
    # If you want to use wandb, uncomment and set your entity:
    # "wandb_entity=your_wandb_username",
]

print("Loading configuration...")
cfg = load_config(overrides=overrides)

print("Starting training...")
train_jacobians(cfg)

print("\nTraining complete! Check the ./output directory for results.")
