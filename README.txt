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
- **Real-time Performance**: Average solving time **&lt; 5ms** per control loop
- **Safety Radius**: 0.2m (configurable padding distance)

---

## 🏗️ System Architecture

