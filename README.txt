# APF-SAC: Artificial Potential Field Guided Soft Actor-Critic for 6-DoF Mobile Manipulator Obstacle Avoidance

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)
[![PyBullet](https://img.shields.io/badge/PyBullet-3.2+-orange.svg)](https://pybullet.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**An ASCF Path Planning Method for a Hybrid Robot in Coating Inspection Scenarios**

This repository implements a **three-stage progressive training framework** combining **Artificial Potential Field (APF) guidance** with **Soft Actor-Critic (SAC)** algorithm for real-time obstacle avoidance trajectory planning of 6-DoF mobile manipulators. The deployment phase integrates **Control Barrier Function Quadratic Programming (CBF-QP)** as a safety filter to guarantee collision-free operation at 100Hz.

---

## 🎯 Key Features

### Training Phase (Pure SAC)
- **Algorithm**: Stable Baselines3 SAC with automatic entropy adjustment
- **Guidance Mechanism**: Distance-dynamic APF guidance (adaptive potential field)
- **Curriculum Learning**: Three-stage progressive training strategy
  - **First 30%**: Fixed scenarios (static obstacle positions)
  - **Middle 30%**: Randomized scenarios (random obstacle initialization)
  - **Final 40%**: Complex scenarios (dual dynamic obstacles)
- **State Space**: 29-dimensional (6 joint positions + APF gradient information + end-effector pose)
- **Action Space**: 6-dimensional continuous (joint velocity increments Δq̇)

### Deployment Phase (SAC + CBF-QP)
- **Safety Filter**: Full multi-point CBF-QP constraint optimization running at **100Hz**
- **Protected Points**: End-effector + Elbow + Wrist (3 critical collision points)
- **Solver**: CVXOPT QP optimizer with warm-starting
- **Real-time Performance**: Average solving time **< 5ms** per control loop
- **Safety Radius**: 0.2m (configurable padding distance)

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Training Pipeline                        │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  Fixed Env   │───→│  Random Env  │───→│ Complex Env  │  │
│  │   (30%)      │    │   (30%)      │    │  (40%)       │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                   │                   │            │
│         └───────────────────┴───────────────────┘            │
│                         │                                    │
│              ┌────────────▼────────────┐                    │
│              │   APF-Guided SAC Agent   │                    │
│              │  (Actor-Critic Network)  │                    │
│              └──────────────────────────┘                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Deployment Pipeline                        │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   SAC    │───→│  CBF-QP Filter│───→│  PyBullet    │      │
│  │  Policy  │    │ (Safety Check)│    │   Physics    │      │
│  │ (Action) │    │ (Constraint)  │    │   Engine     │      │
│  └──────────┘    └──────────────┘    └──────────────┘      │
│                                                      │      │
│              ┌──────────────────────────────────────┘      │
│              │ Multi-point Protection (EE, Elbow, Wrist)   │
│              └────────────────────────────────────────────┘
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```bash
├── APF_SAC_train.py          # Main training script with curriculum learning
├── env.py                    # Training environment (PyBullet + APF)
├── env_test.py               # Testing environment with CBF constraints
├── reward.py                 # Reward function shaping (collision + goal + smoothness)
├── test_apf_sac.py           # Basic testing script (SAC only)
├── test_apf_sac_cbf.py       # Safety-critical testing (SAC + CBF-QP filter)
├── deploy_filter.py          # CBF safety filter wrapper
├── cbf_qp.py                 # CBF constraint formulation and QP solver
└── utils/                    # Helper functions (optional)
    └── config.py             # Hyperparameter configuration
```

---

## ⚠️ Important Note

> **Before running `train` or `test`**: 
> Modify the **URDF path** and **model paths** in `env.py` and `env_test.py` to your **absolute local paths**:
> ```python
> # In env.py / env_test.py, change:
> self.urdf_path = "/absolute/path/to/your/openarm_v10.urdf"
> ```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create conda environment (recommended)
conda create -n apf_sac python=3.8
conda activate apf_sac

# Install core dependencies
pip install stable-baselines3==2.0.0
pip install pybullet==3.2.5
pip install gymnasium==0.28.1
pip install numpy scipy matplotlib
pip install loguru  # Logging utility

# Install CBF-QP solver (critical for deployment)
pip install cvxopt==1.3.0
```

### 2. Training the Model

**Basic training with default parameters:**
```bash
python APF_SAC_train.py \
    --mode train \
    --episodes 6000 \
    --save_dir ./models \
    --log_dir ./logs
```

**Custom hyperparameters:**
```bash
python APF_SAC_train.py \
    --mode train \
    --episodes 10000 \
    --omega 4.0 \
    --safety_radius 0.25 \
    --curriculum True \
    --save_dir ./checkpoints
```

### 3. Testing & Deployment

**Standard testing (SAC policy only):**
```bash
python test_apf_sac.py \
    --model_path ./models/sac_final.zip \
    --episodes 100 \
    --render True
```

**Safety-critical testing (SAC + CBF-QP filter):**
```bash
python test_apf_sac_cbf.py \
    --model_path ./models/sac_final.zip \
    --cbf_radius 0.2 \
    --kappa1 1.25 \
    --kappa2 1.25 \
    --test_episodes 1000 \
    --mode complex  # complex, random, or fixed
```

---

## ⚙️ Configuration Details

### Training Parameters (`Config` Class)

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `CRITIC_LR` | `3e-4` | Critic network learning rate |
| `ACTOR_LR` | `3e-4` | Actor network learning rate |
| `BUFFER_SIZE` | `200000` | Experience replay buffer size |
| `BATCH_SIZE` | `256` | Mini-batch size for training |
| `ENT_COEF` | `"auto_0.125"` | Automatic entropy coefficient tuning |
| `OMEGA` | `4.0` | APF guidance gain upper bound (λ_max) |
| `GAMMA` | `0.99` | Discount factor for reward |
| `TAU` | `0.005` | Soft update coefficient for target networks |

### CBF-QP Safety Parameters

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `safety_radius` | `0.2` | Safety padding radius $r_{pad}$ (meters) |
| `kappa1` ($\hat{\kappa}_1$) | `1.25` | CBF class-$\mathcal{K}$ function parameter 1 |
| `kappa2` ($\hat{\kappa}_2$) | `1.25` | CBF class-$\mathcal{K}$ function parameter 2 |
| `control_bounds` | `[-1.0, 1.0]` | Joint velocity increment limits (rad/s) |
| `protected_points` | `['ee', 'elbow', 'wrist']` | Critical collision check points |

---

## 📊 Performance Metrics

- **Training Convergence**: ~5000 episodes for complex scenarios
- **Inference Frequency**: 100Hz (10ms control loop)
- **CBF-QP Solving Time**: <5ms average (cvxopt with warm-start)
- **Success Rate**: >95% in cluttered environments (obstacle density >0.3/m³)

---

## 📝 Citation

If you use this code in your research, please cite the original paper:

```bibtex
@article{apf_sac_manipulator,
  title={An ASCF Path Planning Method for a Hybrid Robot in Coating Inspection Scenarios},
  author={Junhao Hu},
  journal={Control and Decision (CCDC)},
  year={2025},
  publisher={Chinese Association of Automation}
}
```

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Acknowledgments

- **Stable Baselines3** for the SAC implementation
- **PyBullet** for physics simulation
- **CVXOPT** for real-time QP optimization
- The CBF formulation follows the control barrier function theory from [Ames et al., 2019]
```
