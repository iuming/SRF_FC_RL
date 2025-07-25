"""
RF Cavity Control System

A reinforcement learning system for controlling RF cavity frequency
using PPO (Proximal Policy Optimization) algorithm.

Author: Ming Liu
Email: ming.liu@example.com
GitHub: https://github.com/iuming
Created: 2025-07-25
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Ming Liu"
__email__ = "ming.liu@example.com"

# Import main components
from src import RFCavityControlEnv, RealTimeControlWrapper
from configs import ENV_CONFIG, TRAINING_CONFIG, EVAL_CONFIG

__all__ = [
    'RFCavityControlEnv',
    'RealTimeControlWrapper', 
    'ENV_CONFIG',
    'TRAINING_CONFIG',
    'EVAL_CONFIG'
]
