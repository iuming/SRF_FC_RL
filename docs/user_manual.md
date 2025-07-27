# SRF FC RL User Manual

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start Guide](#quick-start-guide)
4. [Detailed Usage](#detailed-usage)
5. [Real-Time Control Interface](#real-time-control-interface)
6. [Configuration](#configuration)
7. [Understanding the Environment](#understanding-the-environment)
8. [Training Models](#training-models)
9. [Evaluating Models](#evaluating-models)
10. [Troubleshooting](#troubleshooting)
11. [Advanced Features](#advanced-features)
12. [Best Practices](#best-practices)
13. [FAQ](#faq)

## Introduction

The Superconducting RadioFrequency cavity Frequency Control by Reinforcement Learning (SRF FC RL) system is a sophisticated machine learning platform designed to control RF cavity systems using Proximal Policy Optimization (PPO). This system learns to minimize frequency detuning through intelligent piezo-based frequency control.

### Key Features

- **Reinforcement Learning Control**: PPO-based intelligent control system
- **Real-Time Interface**: Live monitoring and manual control capabilities
- **Physics-Based Simulation**: Accurate RF cavity dynamics modeling
- **Multiple Control Modes**: Automatic, manual, and off control modes
- **Comprehensive Visualization**: Real-time plotting and data analysis
- **Configurable Parameters**: Easily adaptable to different cavity systems

### System Requirements

- **Operating System**: Windows 10/11, Linux, or macOS
- **Python**: 3.8 or higher
- **Memory**: Minimum 4GB RAM (8GB recommended)
- **Storage**: At least 2GB available space
- **Hardware**: CPU with multiple cores recommended for training

## Installation

### Step 1: Prerequisites

Ensure you have Python 3.8+ installed on your system. You can download it from [python.org](https://www.python.org/).

### Step 2: Install Dependencies

1. **Clone or download the repository** to your local machine.

2. **Navigate to the project directory**:
   ```bash
   cd SRF_FC_RL
   ```

3. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Step 3: Install LLRF Libraries

The system requires the LLRF (Low Level RF) libraries for RF simulation:

```bash
# Install llrflibs according to your system requirements
# Please refer to the specific installation instructions for your platform
```

### Step 4: Verify Installation

Test the installation by running:
```bash
python main.py env-test
```

If the installation is successful, you should see environment initialization messages without errors.

## Quick Start Guide

### For Windows Users

The easiest way to get started is using the provided batch files:

1. **Training a Model**:
   - Double-click `train.bat`
   - The training process will start automatically

2. **Testing a Model**:
   - Double-click `test.bat`
   - This will evaluate the best trained model

3. **Real-Time Control (Command Line)**:
   - Double-click `realtime.bat`
   - Use keyboard commands to control the system

4. **Real-Time Control (GUI)**:
   - Double-click `realtime_gui.bat`
   - Use the graphical interface for control

### For Linux/macOS Users

Use the command line interface:

```bash
# Test the environment
python main.py env-test

# Train a new model
python main.py train

# Test a trained model
python main.py test

# Real-time control (command line)
python main.py realtime

# Real-time control (GUI)
python main.py realtime-gui
```

## Detailed Usage

### Command Line Interface

The `main.py` script provides several commands:

#### Environment Testing
```bash
python main.py env-test
```
This command verifies that:
- The environment initializes correctly
- Observations and rewards are valid
- Actions are handled properly
- No NaN or infinite values are generated

#### Training
```bash
python main.py train
```
Features:
- PPO algorithm with optimized hyperparameters
- Vectorized environments for parallel training
- Early stopping based on reward threshold
- Automatic model saving and evaluation
- Tensorboard logging for monitoring progress

Training will create:
- `best_model/`: Contains the best performing model
- `models/`: Contains all saved models during training
- `logs/`: Training logs and statistics
- `ppo_rf_cavity_tensorboard/`: Tensorboard logs for visualization

#### Testing
```bash
python main.py test
```
Features:
- Model performance evaluation over multiple episodes
- Detailed demonstration with data logging
- Comprehensive visualization of control performance
- Performance analysis and statistics

#### Real-Time Control
```bash
python main.py realtime      # Command line interface
python main.py realtime-gui  # Graphical interface
```

### Advanced Script Usage

For more control, you can run scripts directly:

```bash
cd scripts

# Training
python train_rf_cavity.py

# Testing and evaluation
python test_rf_cavity.py

# Real-time control
python realtime_simple.py    # Command line
python realtime_gui.py       # GUI

# Environment testing
python test_environment.py
```

## Real-Time Control Interface

The system provides two real-time control interfaces for live monitoring and manual intervention.

### Command Line Interface

Launch with:
```bash
python main.py realtime
```

#### Available Commands

- **`a`** - Enable automatic control (requires loaded model)
- **`m`** - Enable manual control mode
- **`o`** - Turn off all control
- **`p`** - Pause/resume simulation
- **`r`** - Reset simulation
- **`s`** - Show detailed status
- **`q`** - Quit

#### Usage Example

1. Start the interface: `python main.py realtime`
2. The system will display current cavity parameters
3. Press `a` to enable automatic control
4. Monitor the performance in real-time
5. Press `s` to see detailed statistics
6. Press `q` to quit

### GUI Interface

Launch with:
```bash
python main.py realtime-gui
```

#### GUI Components

1. **Control Panel** (Left side):
   - Simulation controls (Start/Pause/Reset/Stop)
   - Control mode selection (Auto/Manual/Off)
   - Manual action slider
   - Model loading and status

2. **Status Display** (Right side):
   - Current parameter values
   - Performance metrics
   - System status log
   - Model information

3. **Real-Time Plots** (Bottom):
   - Cavity voltage amplitude
   - Reflected voltage amplitude
   - Cavity voltage phase
   - Frequency detuning (primary objective)
   - Control actions
   - Rewards

#### GUI Features

- **Real-time plotting**: Live visualization of all system parameters
- **Interactive control**: Manual action slider with immediate feedback
- **Model management**: Load and switch between trained models
- **Data export**: Save simulation data to CSV format
- **Status monitoring**: Comprehensive system status display

#### Using the GUI

1. **Starting**: Click "Start Simulation" to begin
2. **Control Mode**: Select from Auto/Manual/Off
3. **Manual Control**: Use the action slider when in manual mode
4. **Loading Models**: Use "Load Model" to switch between trained models
5. **Data Export**: Click "Export Data" to save current session data
6. **Monitoring**: Watch real-time plots and status information

## Configuration

The system is highly configurable through `configs/config.py`. This file contains three main configuration sections:

### Environment Configuration (`ENV_CONFIG`)

```python
ENV_CONFIG = {
    # Episode settings
    'max_steps': 2048 * 16,           # Maximum steps per episode
    
    # RF System Parameters
    'sampling_time': 1e-6,            # Simulation sampling time
    'fill_time': 510,                 # Cavity fill time
    'flat_time': 1300,                # Flat top time
    
    # RF Source
    'source_frequency': -460,         # Source frequency offset (Hz)
    'source_amplitude': 1,            # Source amplitude
    
    # Cavity parameters
    'cavity_frequency': 1.3e9,        # Cavity resonant frequency (Hz)
    'coupling_beta': 1e4,             # Coupling coefficient
    'cavity_roQ': 1036,               # R/Q ratio
    'loaded_q': 3e6,                  # Loaded quality factor
    'beam_current': 0.008,            # Beam current (A)
    
    # Mechanical modes
    'mechanical_modes': {
        'f': [280, 341, 460, 487, 618],  # Frequencies (Hz)
        'Q': [40, 20, 50, 80, 100],      # Quality factors
        'K': [2, 0.8, 2, 0.6, 0.2]      # Coupling coefficients
    },
}
```

### Training Configuration (`TRAINING_CONFIG`)

```python
TRAINING_CONFIG = {
    # PPO Parameters
    'learning_rate': 1e-4,            # Learning rate
    'n_steps': 32768,                 # Steps per environment per update
    'batch_size': 512,                # Batch size for training
    'n_epochs': 20,                   # Number of epochs per update
    'gamma': 0.99,                    # Discount factor
    'gae_lambda': 0.95,               # GAE lambda parameter
    'clip_range': 0.2,                # PPO clipping parameter
    'ent_coef': 0.001,                # Entropy coefficient
    
    # Environment settings
    'n_envs': 4,                      # Number of parallel environments
    'total_timesteps': 1_000_000,     # Total training timesteps
    
    # Network architecture
    'net_arch': [256, 256],           # Hidden layer sizes
}
```

### Evaluation Configuration (`EVAL_CONFIG`)

```python
EVAL_CONFIG = {
    'n_eval_episodes': 10,            # Number of evaluation episodes
    'deterministic': True,            # Use deterministic policy
    'max_demo_steps': 32768,          # Maximum steps for demonstration
    'sample_rate_threshold': 2000,    # Sampling threshold for plotting
}
```

### Customizing Configuration

To modify the configuration:

1. Open `configs/config.py`
2. Modify the desired parameters
3. Save the file
4. Restart your training or testing session

**Important**: Some parameters require careful consideration:
- Changing cavity parameters affects the physics simulation
- Training parameters impact learning performance
- Always test configuration changes with `python main.py env-test`

## Understanding the Environment

### Observation Space

The environment provides a 4-dimensional continuous observation space:

1. **Cavity Voltage Amplitude** (MV): The amplitude of the voltage in the cavity
2. **Reflected Voltage Amplitude** (MV): The amplitude of the reflected voltage
3. **Cavity Voltage Phase** (degrees): The phase of the cavity voltage
4. **Frequency Detuning** (kHz): The difference between cavity and desired frequency

### Action Space

The action space is 1-dimensional continuous:
- **Piezo Control Signal**: Range [-2.0, 2.0], controls frequency via piezo actuators

### Reward Function

The reward is designed to encourage frequency detuning minimization:
- **Reward = -|frequency_detuning|**
- Higher rewards (closer to 0) indicate better control
- The agent learns to minimize the absolute frequency detuning

### Physics Simulation

The environment simulates:
- **RF Source**: Signal generation with configurable parameters
- **I/Q Modulator**: Handles pulsed/CW operation modes
- **RF Amplifier**: Signal amplification with gain control
- **Cavity Dynamics**: Including mechanical modes and beam loading
- **Piezo Control**: Frequency control through piezo actuators

## Training Models

### Basic Training

Start training with default parameters:
```bash
python main.py train
```

### Monitoring Training Progress

#### Tensorboard
Monitor training in real-time:
```bash
tensorboard --logdir=ppo_rf_cavity_tensorboard/
```
Open your browser to `http://localhost:6006` to view:
- Learning curves
- Policy statistics
- Environment metrics

#### Console Output
During training, you'll see:
- Episode rewards
- Training statistics
- Model save notifications
- Early stopping status

### Training Features

- **Early Stopping**: Training stops when the model achieves satisfactory performance
- **Model Checkpointing**: Best models are automatically saved
- **Vectorized Environments**: Parallel training for faster learning
- **Comprehensive Logging**: Detailed logs for analysis

### Training Tips

1. **CPU vs GPU**: This implementation is optimized for CPU training
2. **Patience**: Initial training may take time to show progress
3. **Monitoring**: Use Tensorboard to monitor training progress
4. **Hyperparameter Tuning**: Modify `TRAINING_CONFIG` for optimization

## Evaluating Models

### Basic Evaluation

Test the best trained model:
```bash
python main.py test
```

### Evaluation Output

The evaluation process provides:
- **Performance Metrics**: Mean reward, detuning statistics
- **Visualization**: Plots of control performance
- **Data Logging**: Detailed episode data
- **Statistical Analysis**: Performance over multiple episodes

### Understanding Results

#### Performance Metrics
- **Mean Absolute Detuning**: Primary performance indicator (lower is better)
- **Control Effort**: Magnitude and variation of actions
- **Stability**: Rate of detuning changes
- **Convergence Time**: Time to reach stable operation

#### Expected Performance
Well-trained models typically achieve:
- **< 1 kHz mean absolute detuning**
- **Smooth control actions**
- **Quick recovery from disturbances**

### Advanced Evaluation

For detailed analysis, run:
```bash
cd scripts
python test_rf_cavity.py
```

This provides additional features:
- Extended evaluation episodes
- Detailed performance analysis
- Comprehensive visualization
- Data export capabilities

## Troubleshooting

### Common Issues

#### Installation Problems

**Import Errors**:
```
ModuleNotFoundError: No module named 'gymnasium'
```
**Solution**: Install dependencies with `pip install -r requirements.txt`

**LLRF Library Issues**:
```
ImportError: cannot import name 'llrflibs'
```
**Solution**: Install the LLRF libraries according to your system requirements

#### Runtime Errors

**NaN/Inf Values**:
```
Warning: NaN or Inf detected in observation/reward
```
**Solution**: The environment includes safety checks. If persistent, check RF simulation parameters in `config.py`

**Memory Issues**:
```
OutOfMemoryError: Unable to allocate memory
```
**Solutions**:
- Reduce `n_envs` in `TRAINING_CONFIG`
- Decrease `max_steps` in `ENV_CONFIG`
- Use smaller network architecture

#### Training Problems

**Training Instability**:
- Use CPU device (already configured)
- Reduce learning rate
- Adjust batch size
- Check reward scaling

**Slow Convergence**:
- Increase `total_timesteps`
- Adjust `learning_rate`
- Modify network architecture
- Check environment configuration

### Performance Optimization

#### CPU Optimization
- The system is optimized for CPU training
- Use multiple cores by setting appropriate `n_envs`
- Monitor CPU usage during training

#### Memory Management
- Adjust `batch_size` based on available memory
- Reduce `max_steps` if memory is limited
- Use appropriate `n_envs` for your system

### Getting Help

If you encounter issues not covered here:

1. **Check the logs**: Look in the `logs/` directory for detailed error messages
2. **Verify installation**: Run `python main.py env-test` to check setup
3. **Review configuration**: Ensure `config.py` parameters are appropriate
4. **Check dependencies**: Verify all required packages are installed

## Advanced Features

### Custom Environment Configuration

You can create custom configurations for different cavity systems:

```python
# Create a custom config
CUSTOM_CONFIG = ENV_CONFIG.copy()
CUSTOM_CONFIG.update({
    'cavity_frequency': 2.6e9,  # Different frequency
    'loaded_q': 5e6,            # Different Q factor
    # ... other parameters
})
```

### Model Comparison

To compare different models:

1. Train multiple models with different hyperparameters
2. Use the evaluation scripts to test each model
3. Compare performance metrics
4. Select the best performing model

### Data Analysis

The system generates extensive data for analysis:

- **Training logs**: In `logs/` directory
- **Tensorboard logs**: In `ppo_rf_cavity_tensorboard/`
- **Evaluation data**: Generated during testing
- **Real-time data**: Exported from GUI interface

### Custom Training

For advanced users, you can customize training:

```python
# Modify training configuration
CUSTOM_TRAINING = TRAINING_CONFIG.copy()
CUSTOM_TRAINING.update({
    'learning_rate': 5e-5,      # Lower learning rate
    'total_timesteps': 2_000_000,  # More training steps
    'n_envs': 8,                # More parallel environments
})
```

## Best Practices

### Training Best Practices

1. **Start with default parameters**: The provided configuration is well-tested
2. **Monitor training**: Use Tensorboard to track progress
3. **Be patient**: RL training can take time to show results
4. **Save checkpoints**: The system automatically saves the best models
5. **Test regularly**: Use evaluation to check model performance

### Deployment Best Practices

1. **Test thoroughly**: Evaluate models extensively before deployment
2. **Monitor performance**: Use real-time interfaces to monitor operation
3. **Have backups**: Keep multiple trained models available
4. **Document changes**: Record any configuration modifications
5. **Regular updates**: Retrain models with new data when available

### Configuration Best Practices

1. **Understand physics**: Cavity parameters should reflect real systems
2. **Test changes**: Always run `env-test` after configuration changes
3. **Document settings**: Keep records of successful configurations
4. **Gradual changes**: Make incremental modifications when tuning
5. **Validate results**: Ensure configuration changes improve performance

## FAQ

### General Questions

**Q: What is the purpose of this system?**
A: The SRF FC RL system uses reinforcement learning to control superconducting RF cavity frequency, minimizing detuning through intelligent piezo-based control.

**Q: What algorithm does it use?**
A: The system uses Proximal Policy Optimization (PPO), a state-of-the-art reinforcement learning algorithm for continuous control problems.

**Q: Can I use this for different cavity systems?**
A: Yes, the system is configurable through `config.py` and can be adapted to different cavity configurations.

### Technical Questions

**Q: Why does training use CPU instead of GPU?**
A: For this MLP (Multi-Layer Perceptron) policy with the current problem size, CPU training provides better stability and sufficient performance.

**Q: How long does training take?**
A: Training time depends on your hardware and configuration. Typically, 1M timesteps take 30-60 minutes on a modern CPU.

**Q: What does "frequency detuning" mean?**
A: Frequency detuning is the difference between the actual cavity frequency and the desired frequency. The goal is to minimize this difference.

### Usage Questions

**Q: Can I run the system without training a new model?**
A: Yes, if you have a pre-trained model in the `best_model/` directory, you can directly use testing and real-time control features.

**Q: How do I know if my model is performing well?**
A: Look for low mean absolute detuning (< 1 kHz), smooth control actions, and stable operation in the evaluation results.

**Q: Can I export data from the real-time interface?**
A: Yes, the GUI interface provides data export functionality to save simulation data in CSV format.

### Troubleshooting Questions

**Q: What if I get NaN or Inf values?**
A: The system includes comprehensive safety checks. If this persists, check your configuration parameters, especially cavity and RF settings.

**Q: Training seems stuck, what should I do?**
A: Monitor the Tensorboard logs. If there's no improvement for extended periods, try adjusting the learning rate or other hyperparameters.

**Q: The GUI doesn't start, what's wrong?**
A: Ensure you have all GUI dependencies installed (tkinter, matplotlib). On some systems, you may need to install these separately.

---

*This user manual is part of the SRF FC RL project. For additional support or questions, please refer to the project repository or contact the development team.*
