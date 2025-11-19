# How JacobianODE Works: Data Format & Training Process

## Overview
JacobianODE learns neural network models of dynamical systems by training on time series trajectories and their Jacobians (derivatives).

---

## 1. Data Generation Pipeline

### Step 1: Define a Dynamical System
Located in: `JacobianODE/dysts_sim/flows.py`

Example - **Lorenz System** (lines 48-63):
```python
class Lorenz(DynSys):
    @staticjit
    def _rhs(x, y, z, t, beta, rho, sigma):
        # Right-hand side: dx/dt = f(x)
        xdot = sigma * (y - x)
        ydot = rho * x - x * z - y
        zdot = x * y - beta * z
        return np.array([xdot, ydot, zdot])

    @staticjit
    def _jac(x, y, z, t, beta, rho, sigma):
        # Jacobian matrix: ∂f/∂x
        return np.array([
            [-sigma, sigma, 0.0],
            [rho - z, -1.0, -x],
            [y, x, -beta]
        ], dtype=np.float64)
```

**Key Points:**
- `_rhs`: Defines the ODE equations (how the system evolves)
- `_jac`: Defines the Jacobian (local linearization of dynamics)
- Parameters: `beta`, `rho`, `sigma` control the system's behavior

### Step 2: Generate Trajectories
Function: `make_trajectories()` in `jacobian_utils.py:149`

**What happens:**
1. **Initialize the system** with parameters (e.g., Lorenz with default parameters)
2. **Create multiple initial conditions** (default: 32 different starting points)
3. **Simulate forward in time** using numerical integration (ODE solver)
4. **Generate trajectories** over multiple periods (default: 12 periods)

**Configuration parameters** (from your script):
```python
n_periods = 12          # How many oscillation cycles to simulate
pts_per_period = 100    # Time points per period (sampling rate)
num_ics = 32            # Number of different initial conditions
```

**Output format:**
```
sol['values'] shape: (num_ics, time_steps, dimensions)
                     (32, 1200, 3) for Lorenz
```

Where:
- `num_ics = 32`: 32 different trajectories
- `time_steps = n_periods * pts_per_period = 12 * 100 = 1200`
- `dimensions = 3`: (x, y, z) for Lorenz system

---

## 2. Data Processing Pipeline

### Step 3: Postprocess Data
Function: `postprocess_data()` in `jacobian_utils.py`

**Operations:**
1. **Add observation noise** (simulates real-world measurement errors)
   - Your config: `obs_noise = 0.01` (1% noise)
2. **Optional filtering** (smooth noisy data)
   - Butterworth filter, etc.
3. **Optional normalization** (scale data to [-1, 1] or [0, 1])

**Example:**
```
Original trajectory: [x₁, x₂, ..., x₁₂₀₀]
After noise: [x₁ + ε₁, x₂ + ε₂, ..., x₁₂₀₀ + ε₁₂₀₀]
where εᵢ ~ N(0, 0.01 * std(x))
```

### Step 4: Create Training Sequences
Function: `create_dataloaders()` in `jacobian_utils.py:289`

**What happens:**
1. **Split trajectories** into train/validation/test sets
2. **Create sequences** of length `seq_length` (default: 25)
3. **Package into PyTorch DataLoader**

**Data structure:**
```python
class TimeSeriesDataset:
    sequence: torch.Tensor  # Shape: (n_sequences, seq_length, n_dims)

    # For Lorenz:
    # n_sequences: number of subsequences extracted
    # seq_length: 25 time steps
    # n_dims: 3 (x, y, z)
```

**Example sequence:**
```
Input sequence (25 time steps):
[[x₁, y₁, z₁],
 [x₂, y₂, z₂],
 ...
 [x₂₅, y₂₅, z₂₅]]

Shape: (25, 3)
```

---

## 3. Model Architecture

### Neural Network (MLP)
Located in: `JacobianODE/models/mlp.py`

**Default architecture** (from your run):
```
Input: 3 dimensions (x, y, z)
↓
Hidden Layer 1: 256 neurons + SiLU activation
↓
Hidden Layer 2: 1024 neurons + SiLU activation
↓
Hidden Layer 3: 2048 neurons + SiLU activation
↓
Hidden Layer 4: 2048 neurons + SiLU activation
↓
Output: 3 dimensions (dx/dt, dy/dt, dz/dt)

Total parameters: 6.6 Million
```

**What the model learns:**
```
Neural Network: f_θ(x) ≈ dx/dt

Given current state x, predict the derivative (rate of change)
```

---

## 4. Training Process

### Training Loop
Function: `train_model()` in `jacobian_utils.py:530`

**Each training iteration:**

1. **Get a batch** of sequences (batch_size = 16)
   ```
   Batch shape: (16, 25, 3)
   16 sequences, each 25 time steps, 3 dimensions
   ```

2. **Forward pass** - predict next states
   ```python
   for each state x_t in sequence:
       prediction = model(x_t)  # Predict dx/dt
       x_{t+1} = x_t + dt * prediction  # Integrate
   ```

3. **Compute losses:**

   a. **Trajectory Loss** (MSE):
   ```
   Loss = ||predicted_trajectory - true_trajectory||²
   ```

   b. **Loop Closure Loss** (ensures periodic orbits close properly):
   ```
   Loss = ||x_end - x_start||² for periodic trajectories
   Weight: 0.001 (from your config)
   ```

   c. **Jacobian Penalty** (optional, encourages accurate derivatives):
   ```
   Loss = ||∂model/∂x - true_jacobian||²
   Weight: 0.0 (disabled in your config)
   ```

4. **Backpropagation** - update model weights

5. **Validation** - test on held-out data

### Teacher Forcing
Your config uses **teacher forcing with annealing**:
- **Initially**: Model gets true previous state to predict next state (easier)
- **Gradually**: Model uses its own predictions (harder, more realistic)
- **Annealing rate**: `gamma = 0.999` (reduces teacher forcing each epoch)

---

## 5. Training Output

### What gets saved to `./output/`:

1. **Model checkpoints**
   - Best model weights based on validation loss

2. **WandB logs** (if configured)
   - Training/validation loss curves
   - Metrics: R² score, MSE, etc.

3. **Generated trajectories**
   - Visualizations of predicted vs. true trajectories

### Metrics you see during training:
```
train jac r2_score: 0.678
```
- R² score of 0.678 means model explains 67.8% of variance in Jacobian
- Closer to 1.0 is better

---

## 6. Data Format Summary

### Input to Training
```python
{
    'train_trajectories': Tensor of shape (n_train, seq_length, n_dims)
    'val_trajectories': Tensor of shape (n_val, seq_length, n_dims)
    'test_trajectories': Tensor of shape (n_test, seq_length, n_dims)
}

For your Lorenz example:
- n_dims = 3 (x, y, z coordinates)
- seq_length = 25 (25 consecutive time steps)
- n_train ≈ thousands (depends on how sequences are extracted)
```

### Each Training Sample
```python
# One sample from DataLoader
sample = {
    'sequence': torch.Tensor of shape (seq_length, n_dims)
}

# Example values for Lorenz:
sequence[0] = [x₁, y₁, z₁]  # State at time t₁
sequence[1] = [x₂, y₂, z₂]  # State at time t₂
...
sequence[24] = [x₂₅, y₂₅, z₂₅]  # State at time t₂₅
```

---

## 7. Example: One Training Step

**Input:**
```
Current state: x = [1.5, -2.3, 25.4]
```

**Model prediction:**
```
dx/dt = model(x) = [σ(y-x), ρx-xz-y, xy-βz]
                 ≈ [-3.8, 5.2, -14.1]
```

**Integration (Euler step):**
```
x_next = x + dt * (dx/dt)
       = [1.5, -2.3, 25.4] + 0.01 * [-3.8, 5.2, -14.1]
       = [1.462, -2.248, 25.259]
```

**Loss computation:**
```
True next state: x_true = [1.461, -2.251, 25.261]
Prediction error: ||x_next - x_true||² = 0.00012
```

---

## 8. How to Use Your Own Data

If you have time series data from experiments:

```python
# Your data format should be:
data = np.array(shape=(n_trajectories, time_steps, dimensions))

# Example: 10 trajectories, 1000 time steps each, 3 dimensions
data.shape = (10, 1000, 3)

# Then configure:
overrides = [
    "data=custom",
    "data.custom_data_path=my_data.npy",
    ...
]
```

---

## Key Takeaways

1. **Input**: Multiple trajectories from dynamical system
2. **Format**: `(n_trajectories, time_steps, dimensions)` tensors
3. **Model**: Neural network learns `f(x) = dx/dt`
4. **Training**: Minimize prediction error on trajectories
5. **Output**: Trained model that can predict future states

The beauty is that once trained, you can:
- Predict long-term dynamics
- Analyze system stability (via learned Jacobian)
- Discover hidden patterns in chaotic systems
