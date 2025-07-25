"""
Configuration File for RF Cavity Control System

Filename: config.py
Author: Ming Liu
Email: ming.liu@example.com
GitHub: https://github.com/iuming
Created: 2025-07-25
Version: 1.0.0

Description:
    This module contains all configuration parameters for the RF cavity control
    system, including environment settings, training hyperparameters, and
    evaluation configurations. The centralized configuration approach allows
    for easy parameter tuning and experimental setup.

Features:
    - Environment configuration (RF system parameters)
    - Training configuration (PPO hyperparameters)
    - Evaluation configuration (testing parameters)
    - Real-time control settings
    - Modular parameter organization

Dependencies:
    - None (pure configuration)

Changelog:
    v1.0.0 (2025-07-25):
        - Initial configuration structure
        - Added comprehensive environment parameters
        - Defined PPO training hyperparameters
        - Created evaluation and real-time control configs
        - Organized parameters by functional categories

License:
    This code is part of the ML_Learning repository.
    
Usage:
    from config import ENV_CONFIG, TRAINING_CONFIG, EVAL_CONFIG
    
    env = RFCavityControlEnv(config=ENV_CONFIG)
    model = PPO(**TRAINING_CONFIG)
"""

# Environment Configuration
ENV_CONFIG = {
    # Episode settings
    'max_steps': 2048 * 16,
    
    # RF System Parameters
    'sampling_time': 1e-6,
    'fill_time': 510,
    'flat_time': 1300,
    
    # RF Source
    'source_frequency': -460,
    'source_amplitude': 1,
    
    # Pulsed operation
    'pulsed_mode': True,
    'buffer_size': 2048 * 8,
    
    # Amplifier
    'amplifier_gain_db': 20 * 2.3025850929940457,  # 20 * log10(12e6)
    
    # Cavity parameters
    'mechanical_modes': {
        'f': [280, 341, 460, 487, 618],  # Frequencies (Hz)
        'Q': [40, 20, 50, 80, 100],      # Quality factors
        'K': [2, 0.8, 2, 0.6, 0.2]      # Coupling coefficients
    },
    'cavity_frequency': 1.3e9,           # Cavity resonant frequency (Hz)
    'coupling_beta': 1e4,                # Coupling coefficient
    'cavity_roQ': 1036,                  # R/Q ratio
    'loaded_q': 3e6,                     # Loaded quality factor
    'beam_current': 0.008,               # Beam current (A)
    
    # Simulation settings
    'simulation_length': 2048 * 500,
    'pulse_length': 2048 * 20,
}

# Training Configuration
TRAINING_CONFIG = {
    # PPO Parameters
    'algorithm': 'PPO',
    'policy': 'MlpPolicy',
    'learning_rate': 1e-4,
    'n_steps': 32768,
    'batch_size': 512,
    'n_epochs': 20,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef': 0.001,
    'device': 'cpu',  # Use CPU for better stability
    
    # Environment settings
    'n_envs': 4,
    'total_timesteps': 1_000_000,
    
    # Network architecture
    'net_arch': [256, 256],
    
    # Callbacks
    'eval_freq': 50000,
    'early_stop_threshold': -0.1,
    'early_stop_check_freq': 10000,
    
    # Logging
    'tensorboard_log': './ppo_rf_cavity_tensorboard/',
    'log_path': './logs/',
    'best_model_path': './best_model/',
}

# Evaluation Configuration
EVAL_CONFIG = {
    'n_eval_episodes': 10,
    'deterministic': True,
    'render': False,
    'max_demo_steps': 32768,
    'sample_rate_threshold': 2000,  # Sample data if more than this many points
}
