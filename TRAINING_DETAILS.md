# JacobianODE Training: Complete Technical Breakdown

## Table of Contents
1. [Overview](#overview)
2. [Theoretical Foundation](#theoretical-foundation)
3. [Data Generation](#data-generation)
4. [Training Architecture](#training-architecture)
5. [Path Integration Theory](#path-integration-theory)
6. [Loop Closure](#loop-closure)
7. [Training Step-by-Step](#training-step-by-step)

---

## Overview

JacobianODE learns neural network models of dynamical systems by:
1. **Training on trajectories** - Learn to predict next states
2. **Using Jacobian information** - Learn local linearizations of dynamics
3. **Enforcing path independence** - Use loop closure to ensure consistent dynamics

---

## Theoretical Foundation

### 1. Dynamical Systems
A continuous dynamical system is defined by an ODE:

```
dx/dt = f(x, t)
```

Where:
- `x ∈ ℝⁿ` is the state vector
- `f: ℝⁿ × ℝ → ℝⁿ` is the vector field
- `t` is time

**Example - Lorenz System:**
```
dx/dt = σ(y - x)
dy/dt = ρx - xz - y
dz/dt = xy - βz
```

### 2. The Jacobian Matrix
The Jacobian captures local linear approximation:

```
J(x, t) = ∂f/∂x
```

For Lorenz:
```
J = [[-σ,    σ,   0  ],
     [ρ-z,  -1,  -x  ],
     [y,     x,  -β  ]]
```

**Why it matters:**
- Describes how small perturbations grow/decay
- Determines stability of equilibria
- Essential for accurate long-term predictions

### 3. Path Integration
For a conservative vector field, the integral along a path is:

```
∫_path J(x) · dx
```

**Key property:** For path-independent (conservative) fields:
```
∮_loop J(x) · dx = 0
```

This is the basis for **loop closure training**.

---

## Data Generation

### Function: `make_trajectories()`
**Location:** `jacobian_utils.py:149`

### Step 1: Initialize System
```python
# Example: Lorenz system
eq = Lorenz(sigma=10, rho=28, beta=8/3)
```

**Code:** `flows.py:48-63`
- Defines `_rhs(x, y, z, t, ...)`: The ODE equations
- Defines `_jac(x, y, z, t, ...)`: The Jacobian matrix

### Step 2: Generate Initial Conditions
```python
num_ics = 32  # Number of different trajectories
initial_conditions = random_initial_conditions(num_ics, dim=3)
```

**Shape:** `(32, 3)` - 32 starting points in 3D space

### Step 3: Numerical Integration
```python
# For each initial condition
for ic in initial_conditions:
    # Integrate ODE forward in time
    trajectory = odeint(
        func=eq.rhs,        # dx/dt = f(x)
        y0=ic,              # Starting point
        t=time_points,      # Times to evaluate
        method='RK45'       # Runge-Kutta solver
    )
```

**Parameters:**
- `n_periods = 12`: Number of oscillation cycles
- `pts_per_period = 100`: Sampling rate
- Total time steps: `12 × 100 = 1200`

**Output shape:** `(32, 1200, 3)`
- 32 trajectories
- 1200 time points each
- 3 dimensions (x, y, z)

---

## Training Architecture

### Neural Network Model
**Code:** `models/mlp.py:44-192`

```python
class MLP(nn.Module):
    # Architecture from your run:
    # Input: 3 dimensions (x, y, z)
    # Hidden layers: [256, 1024, 2048, 2048]
    # Output: 3 dimensions (dx/dt, dy/dt, dz/dt)

    def forward(self, x):
        # x: current state (batch, 3)
        # returns: derivative (batch, 3)
        for layer in self.layers:
            x = layer(x)
        return x  # Predicted dx/dt
```

**What it learns:**
```
Neural Net: f_θ(x) ≈ dx/dt

Given current state x = [x, y, z]
Predict rate of change dx/dt = [dx/dt, dy/dt, dz/dt]
```

---

## Path Integration Theory

### The Core Idea

Instead of just predicting one step ahead, we want to integrate along entire paths through state space.

### Path Integral Formulation

**Code:** `jacobianODE.py:142-210`

The key function `H(s, t)` computes:

```
H(s, t) = ∫_s^t J(c(r)) · c'(r) dr
```

Where:
- `c(r)`: A path through state space from time `s` to time `t`
- `c'(r)`: Derivative of the path (velocity along path)
- `J(c(r))`: Jacobian evaluated along the path
- Result: Change in state from `x_s` to `x_t`

### Why Use Path Integration?

**Traditional approach:**
```
x_{t+1} = x_t + dt · f(x_t)    # Euler step
```

**Path integration approach:**
```
x_t = x_s + ∫_s^t J(c(r)) · c'(r) dr
```

**Advantages:**
1. **More accurate** - Uses Jacobian information
2. **Captures path structure** - Not just point-to-point
3. **Enables loop closure** - Can verify consistency

### Two Types of Paths

#### 1. Spline Path (`inner_path='spline'`)
**Code:** `jacobianODE.py:20-50`

```python
# Fit cubic spline through trajectory points
spline = PiecewiseCubicSpline(time_vals, trajectory)
c(t) = spline.evaluate(t)       # Position at time t
c'(t) = spline.derivative(t)    # Velocity at time t
```

**Use:** Smooth interpolation of actual trajectories

#### 2. Line Path (`inner_path='line'`)
**Code:** `paths.py` and `jacobianODE.py:183-184`

```python
# Linear interpolation between two points
c(t) = x_s + (t - s)/(t_f - s) * (x_t - x_s)
c'(t) = (x_t - x_s)/(t_f - s)
```

**Use:** Direct paths for loop closure

### Integration Methods

**Code:** `jacobianODE.py:99-108`

Three numerical integration schemes available:

1. **Trapezoid Rule** (default for your run)
```
∫_a^b f(x)dx ≈ (b-a)/2 · [f(a) + f(b)]
```

2. **Simpson's Rule**
```
∫_a^b f(x)dx ≈ (b-a)/6 · [f(a) + 4f((a+b)/2) + f(b)]
```

3. **Gauss-Legendre**
- Higher order accuracy
- Uses optimal quadrature points

---

## Loop Closure

### What is Loop Closure?

**Mathematical Property:**
For a conservative (path-independent) vector field:
```
∮_closed_loop J(x) · dx = 0
```

Any closed path should return to starting point → integral should be zero.

### Why Use Loop Closure?

**Problem:** Without constraints, the neural network might learn inconsistent dynamics:
- Path A→B→A might not return to starting point
- Different paths between same points might give different results

**Solution:** Add loop closure loss to enforce consistency.

### Function: `make_loops()`
**Code:** `lightning_base.py:65-84`

```python
def make_loops(pts, n_loops, n_loop_pts):
    """
    Generate random closed loops by:
    1. Randomly sample points from trajectories
    2. Concatenate them to form a path
    3. Add starting point at end to close the loop
    """
    # pts shape: (batch, time, dims)

    # Randomly sample n_loop_pts points
    indices = random.choice(all_points, size=(n_loops, n_loop_pts))
    loop_pts = pts[indices]  # (n_loops, n_loop_pts, dims)

    # Close the loop: add first point at the end
    loop_pts = cat([loop_pts, loop_pts[:, [0], :]], dim=1)
    # Now shape: (n_loops, n_loop_pts+1, dims)

    return loop_pts
```

**Example:**
```
Original trajectory points: [A, B, C, D, E, F, G, H]

Sample 4 random points: [C, G, B, E]

Create loop: [C → G → B → E → C]
             ↑                 ↓
             └─────────────────┘
```

### Function: `loop_closure()`
**Code:** `lightning_base.py:115-181`

```python
def loop_closure(batch, jac_func, dt, n_loops, n_loop_pts,
                 loop_path='line', int_method='Trapezoid'):
    """
    Compute loop closure integrals

    Steps:
    1. Generate random closed loops
    2. For each segment of loop, compute path integral
    3. Sum all segments
    4. Result should be ≈ 0 for consistent dynamics
    """

    # Step 1: Create loops
    loop_pts = make_loops(batch, n_loops, n_loop_pts)
    # Shape: (n_loops, n_loop_pts+1, dims)

    # Step 2: Initialize integral accumulator
    loop_int = zeros(n_loops, dims)

    # Step 3: Integrate along each segment
    for i in range(n_loop_pts):
        x_start = loop_pts[:, i, :]
        x_end = loop_pts[:, i+1, :]

        # Compute integral from x_start to x_end
        segment_int = H(
            s=i*dt,
            t=(i+1)*dt,
            x_s=x_start,
            x_t=x_end,
            inner_path='line'  # Use straight line between points
        )

        loop_int += segment_int

    # Step 4: loop_int should be ≈ 0
    return loop_int
```

### Configuration

**Your settings:**
```python
loop_closure_weight = 0.001         # Weight in loss function
loop_closure_training = True        # Enable loop closure
n_loop_pts = 20                     # Points per loop
loop_closure_interp_pts = 20        # Integration points per segment
loop_path = 'line'                  # Use linear interpolation
loop_closure_int_method = 'Trapezoid'  # Integration method
mix_trajectories = True             # Sample from all trajectories
```

---

## Training Step-by-Step

### Main Training Loop
**Code:** `lightning_base.py:715-782`

### Step 1: Get Batch
```python
# training_step() is called for each batch
batch shape: (16, 25, 3)
# 16 sequences
# 25 time steps per sequence
# 3 dimensions (x, y, z)
```

### Step 2: Compute Jacobians
**Code:** `lightning_base.py:729`

```python
jacs_pred = self.get_pred_jacs(batch)
# Compute Jacobian using automatic differentiation
# Shape: (16, 25, 3, 3)
# For each of 16×25 states, compute 3×3 Jacobian matrix
```

**How it works - Code:** `mlp.py:233-267`
```python
def compute_jacobians(self, batch):
    # Use PyTorch automatic differentiation
    jacs = torch.func.jacfwd(lambda x: self.model(x))(batch)
    # jacfwd: Forward-mode autodiff
    # Computes ∂f/∂x for neural network f
    return jacs
```

### Step 3: Update Teacher Forcing
**Code:** `lightning_base.py:732`

```python
self.update_alpha_teacher_forcing(jacs_pred, batch_idx)
```

**Teacher forcing schedule:**
```
Epoch 1: α = 1.0    (100% use true states)
Epoch 5: α ≈ 0.995  (99.5% true states)
Epoch 10: α ≈ 0.990
...
Eventually: α → min_alpha (default: 0)
```

**Code:** `lightning_base.py:669-683`
```python
def update_alpha_teacher_forcing(self, jacs_pred, batch_idx):
    # Get Lyapunov estimate from Jacobian
    alpha_lyap = get_alpha_lyap(exp(jacs_pred * dt))

    # Exponential decay with moving average
    alpha_new = alpha_old * γ + (1 - γ) * alpha_lyap
    # γ = 0.999 (your setting)

    self.alpha_teacher_forcing = max(alpha_new, min_alpha)
```

### Step 4: Trajectory Training
**Code:** `lightning_base.py:738`

```python
train_rets['trajectory'] = self.trajectory_model_step(batch, batch_idx)
```

**Inside `trajectory_model_step()` - Code:** `lightning_base.py:469-594`

#### 4a. Add Observation Noise
```python
# Simulate measurement errors
noise = randn_like(batch) * obs_noise_scale
noisy_batch = batch + noise
# obs_noise_scale = 0.01 (1% noise) in your config
```

#### 4b. Path Integration via JacobianODEint
**Code:** `lightning_base.py:544-557`

```python
jacobian_odeint = JacobianODEint(self.compute_jacobians, self.dt)
outputs = jacobian_odeint.generate_dynamics(
    batch,
    alpha_teacher_forcing=alpha,
    teacher_forcing_steps=1,
    traj_init_steps=15,      # Use first 15 points as initialization
    inner_path='line',
    inner_N=20,              # 20 integration points
    interp_pts=4,           # Interpolation resolution
)
```

**What `generate_dynamics()` does:**

**Code:** `jacobianODE.py` (class `JacobianODEint`)

```python
def generate_dynamics(batch, alpha_teacher_forcing, traj_init_steps):
    """
    Generate trajectory predictions using path integration

    Args:
        batch: (B, T, D) - B sequences, T timesteps, D dimensions
        alpha_teacher_forcing: 0-1, how much to use true states
        traj_init_steps: Number of initial steps to use as "ground truth"

    Process:
    """
    B, T, D = batch.shape
    outputs = zeros_like(batch)

    # Step 1: Copy initial segment (ground truth)
    outputs[:, :traj_init_steps, :] = batch[:, :traj_init_steps, :]

    # Step 2: Predict remaining timesteps
    for t in range(traj_init_steps, T):
        # Teacher forcing: mix true and predicted states
        x_prev = alpha * batch[:, t-1, :] + (1-alpha) * outputs[:, t-1, :]

        # Path integration from t-1 to t
        x_next = x_prev + H(
            s=(t-1)*dt,
            t=t*dt,
            x_s=x_prev,
            x_t=None,  # To be computed
            inner_path='line',
            N=20  # integration points
        )

        outputs[:, t, :] = x_next

    return outputs
```

**The H() function in detail:**
```python
def H(s, t, x_s, x_t, inner_path='line', N=20):
    """
    Compute: ∫_s^t J(c(r)) · c'(r) dr

    where c(r) is a path from x_s to x_t
    """
    # Create linear path
    c = lambda r: x_s + (r - s)/(t - s) * (x_t - x_s)
    c_prime = lambda r: (x_t - x_s)/(t - s)

    # Numerical integration using Trapezoid rule
    grid_points = linspace(s, t, N)  # 20 points from s to t

    integral = 0
    for i in range(N-1):
        r1, r2 = grid_points[i], grid_points[i+1]

        # Integrand: J(c(r)) @ c'(r)
        f1 = compute_jacobians(c(r1)) @ c_prime(r1)
        f2 = compute_jacobians(c(r2)) @ c_prime(r2)

        # Trapezoid rule
        integral += (r2 - r1) / 2 * (f1 + f2)

    return integral
```

#### 4c. Compute Trajectory Loss
```python
# Compare predicted vs true
loss_trajectory = MSE(outputs, batch)
# Only compute loss after traj_init_steps
loss = MSE(outputs[:, 15:, :], batch[:, 15:, :])
```

### Step 5: Loop Closure Training
**Code:** `lightning_base.py:741`

```python
if self.loop_closure_training:
    train_rets['loop_closure'] = self.loop_closure_model_step(batch)
```

**Inside `loop_closure_model_step()` - Code:** `lightning_base.py:596-667`

```python
def loop_closure_model_step(batch):
    # Generate random loops
    loop_int = loop_closure(
        batch,
        jac_func=self.compute_jacobians,
        n_loops=None,  # defaults to batch_size
        n_loop_pts=20,
        loop_path='line',
        loop_closure_interp_pts=20
    )
    # loop_int shape: (batch_size, dims)
    # Should be ≈ 0 for each loop

    # Loss: penalize non-zero loop integrals
    loop_loss = (loop_int ** 2).mean()

    return loop_loss
```

### Step 6: Combine Losses
**Code:** `lightning_base.py:743-760`

```python
# Weights
alpha = (alpha_teacher_forcing - min_alpha) / (1 - min_alpha)
trajectory_weight = 1
loop_closure_weight = (1 - alpha) * final_loop_closure_weight +
                      alpha * loop_closure_weight
# As teacher forcing decreases (α → 0),
# loop closure weight increases

# Total loss
total_loss = (
    trajectory_weight * loss_trajectory +
    loop_closure_weight * loss_loop_closure +
    jac_penalty * jac_norm +              # Jacobian regularization (0 in your config)
    l1_penalty * l1_norm                  # L1 regularization (0 in your config)
)

# Your settings:
# trajectory_weight = 1.0
# loop_closure_weight starts at 0.001, increases as training progresses
```

### Step 7: Backpropagation
```python
# PyTorch Lightning handles this automatically
total_loss.backward()  # Compute gradients
optimizer.step()       # Update weights
```

**Optimizer - Code:** `lightning_base.py:1027-1063`
```python
optimizer = AdamW(
    model.parameters(),
    lr=1e-4,            # Learning rate
    weight_decay=1e-4   # L2 regularization
)
```

### Step 8: Logging
**Code:** `lightning_base.py:767-780`

```python
if (batch_idx + 1) % log_interval == 0:
    # Log to WandB
    log("trajectory train_loss", loss_trajectory)
    log("loop_closure train_loss", loss_loop_closure)
    log("total train loss", total_loss)
    log("train jac r2_score", r2_score(jac_true, jac_pred))
    log("alpha teacher forcing", alpha_teacher_forcing)
```

---

## Validation Step

**Code:** `lightning_base.py:815-866`

```python
def validation_step(batch):
    # Use validation settings
    alpha_validation = 0  # No teacher forcing
    obs_noise_scale = 0   # No noise

    # Predict trajectories
    val_rets = trajectory_model_step(
        batch,
        alpha_teacher_forcing=0,  # Pure prediction
        obs_noise_scale=0,
        traj_init_steps=15,       # Still use 15 initial points
        inner_N=20
    )

    # Also compute loop closure
    val_loop = loop_closure_model_step(batch)

    # Log validation metrics
    val_loss = val_rets['loss']
    log("mean val loss", val_loss)
    log("val loop closure loss", val_loop['loss'])

    return val_loss
```

---

## Summary: One Complete Training Iteration

```
┌─────────────────────────────────────────────┐
│ 1. GET BATCH                                │
│    Shape: (16, 25, 3)                       │
└─────────────┬───────────────────────────────┘
              │
┌─────────────▼───────────────────────────────┐
│ 2. COMPUTE JACOBIANS                        │
│    jacs = ∂(neural_net)/∂x                  │
│    Shape: (16, 25, 3, 3)                    │
└─────────────┬───────────────────────────────┘
              │
┌─────────────▼───────────────────────────────┐
│ 3. UPDATE TEACHER FORCING                   │
│    α_new = 0.999 * α_old + 0.001 * α_lyap   │
│    (Gradually reduce α from 1 to 0)         │
└─────────────┬───────────────────────────────┘
              │
┌─────────────▼───────────────────────────────┐
│ 4. TRAJECTORY PREDICTION                    │
│    ┌────────────────────────────────────┐   │
│    │ 4a. Add noise (1% of std)          │   │
│    └────────┬───────────────────────────┘   │
│    ┌────────▼───────────────────────────┐   │
│    │ 4b. Path integration               │   │
│    │  For t = 15..25:                   │   │
│    │    x_t = x_{t-1} +                 │   │
│    │         ∫ J(c(r))·c'(r) dr         │   │
│    └────────┬───────────────────────────┘   │
│    ┌────────▼───────────────────────────┐   │
│    │ 4c. Compute loss                   │   │
│    │  loss_traj = MSE(pred, true)       │   │
│    └────────────────────────────────────┘   │
└─────────────┬───────────────────────────────┘
              │
┌─────────────▼───────────────────────────────┐
│ 5. LOOP CLOSURE                             │
│    ┌────────────────────────────────────┐   │
│    │ 5a. Generate random loops          │   │
│    │  [C → G → B → E → C]               │   │
│    └────────┬───────────────────────────┘   │
│    ┌────────▼───────────────────────────┐   │
│    │ 5b. Integrate around loop          │   │
│    │  loop_int = ∮ J(x)·dx              │   │
│    └────────┬───────────────────────────┘   │
│    ┌────────▼───────────────────────────┐   │
│    │ 5c. Compute loss                   │   │
│    │  loss_loop = ||loop_int||²         │   │
│    └────────────────────────────────────┘   │
└─────────────┬───────────────────────────────┘
              │
┌─────────────▼───────────────────────────────┐
│ 6. COMBINE LOSSES                           │
│    total_loss = 1.0 * loss_traj +           │
│                 0.001 * loss_loop           │
└─────────────┬───────────────────────────────┘
              │
┌─────────────▼───────────────────────────────┐
│ 7. BACKPROPAGATION                          │
│    total_loss.backward()                    │
│    optimizer.step()                         │
└─────────────────────────────────────────────┘
```

---

## Key Insights

### 1. Why Path Integration?
- **Standard approach:** x_{t+1} = x_t + Δt · f(x_t) (Euler)
- **This approach:** x_t = x_s + ∫ J(path) · dpath
- **Benefit:** Accounts for Jacobian structure, more accurate

### 2. Why Loop Closure?
- Ensures learned dynamics are **path-independent**
- Prevents accumulation of errors
- Forces model to learn conservative field

### 3. Why Teacher Forcing Annealing?
- **Early training:** Model needs help (α=1, use true states)
- **Late training:** Model must be autonomous (α→0, use predictions)
- **Gradual transition:** Prevents instability

### 4. How It All Fits Together
```
Trajectory Loss     →  Learn correct state evolution
    +
Loop Closure Loss   →  Ensure path independence
    +
Jacobian Penalty    →  Match true Jacobian (if available)
    ↓
Complete, Consistent Model of Dynamics
```

---

## Configuration Summary (Your Run)

```python
# Data
n_periods = 12
pts_per_period = 100
num_ics = 32
obs_noise = 0.01  # 1% noise

# Model
hidden_dim = [256, 1024, 2048, 2048]
activation = 'silu'
total_parameters = 6.6M

# Training
batch_size = 16
max_epochs = 50  # (you reduced from 1000)
optimizer = AdamW(lr=1e-4, weight_decay=1e-4)

# Path Integration
traj_init_steps = 15      # Use first 15 points as ground truth
inner_N = 20              # 20 integration points per segment
inner_path = 'line'       # Linear interpolation
interp_pts = 4

# Loop Closure
loop_closure_weight = 0.001
n_loop_pts = 20
loop_closure_interp_pts = 20
loop_closure_int_method = 'Trapezoid'

# Teacher Forcing
alpha_teacher_forcing_initial = 1.0
gamma_teacher_forcing = 0.999  # Decay rate
teacher_forcing_annealing = True
min_alpha_teacher_forcing = 0.0
```

---

## References to Code

| Component | Function | File | Line |
|-----------|----------|------|------|
| Main training loop | `train_jacobians()` | `run_jacobians.py` | 12 |
| Training step | `training_step()` | `lightning_base.py` | 715 |
| Trajectory prediction | `trajectory_model_step()` | `lightning_base.py` | 469 |
| Path integration | `H()` | `jacobianODE.py` | 142 |
| Loop generation | `make_loops()` | `lightning_base.py` | 65 |
| Loop closure | `loop_closure()` | `lightning_base.py` | 115 |
| Jacobian computation | `compute_jacobians()` | `mlp.py` | 233 |
| Teacher forcing update | `update_alpha_teacher_forcing()` | `lightning_base.py` | 669 |
| Neural network | `MLP` | `mlp.py` | 44 |
| Lorenz system | `Lorenz` | `flows.py` | 48 |
