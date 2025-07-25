"""
RF Cavity Control Training Script

Filename: train_rf_cavity.py
Author: Ming Liu
Email: ming.liu@example.com
GitHub: https://github.com/iuming
Created: 2025-07-25
Version: 1.0.0

Description:
    This script trains a Proximal Policy Optimization (PPO) agent to control
    RF cavity amplitude and phase using reinforcement learning. The training
    process uses Stable-Baselines3 with optimized hyperparameters for stable
    convergence on CPU-based training.

Features:
    - PPO algorithm with optimized hyperparameters
    - CPU-only training for numerical stability
    - Tensorboard logging for monitoring
    - Model checkpointing and saving
    - Comprehensive error handling
    - Progress monitoring with callback system

Dependencies:
    - stable_baselines3
    - gymnasium
    - numpy
    - tensorboard (for logging)
    - matplotlib (for visualization)
    - llrflibs (for RF cavity simulation)

Changelog:
    v1.0.0 (2025-07-25):
        - Initial training script implementation
        - Added PPO with CPU optimization
        - Implemented comprehensive logging
        - Added model saving and checkpointing
        - Fixed NaN handling issues
        - Optimized hyperparameters for stability

License:
    This code is part of the ML_Learning repository.
    
Usage:
    python train_rf_cavity.py
    
    # Monitor training progress:
    tensorboard --logdir ./logs/
    
    # Adjust hyperparameters in configs/config.py
"""

import gymnasium as gym
import numpy as np
import os
import sys
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'configs'))

from rf_cavity_env import RFCavityControlEnv
from config import ENV_CONFIG, TRAINING_CONFIG


class EarlyStoppingCallback(BaseCallback):
    """
    Callback for early stopping based on reward threshold
    """
    def __init__(self, reward_threshold: float, check_freq: int, verbose=0):
        super().__init__(verbose)
        self.reward_threshold = reward_threshold
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Get recent episode rewards
            rewards = [ep_info["r"] for ep_info in self.model.ep_info_buffer]
            if len(rewards) > 0:
                mean_reward = np.mean(rewards)
                if self.verbose >= 1:
                    print(f"Step {self.n_calls}: Mean reward = {mean_reward:.4f}")
                
                if mean_reward > self.reward_threshold:
                    print(f"Early stopping triggered! Mean reward {mean_reward:.2f} > {self.reward_threshold}")
                    return False  # Stop training
        return True


def create_env(env_config=None):
    """Create a single environment instance"""
    if env_config is None:
        env_config = ENV_CONFIG
    
    env = RFCavityControlEnv(
        render_mode=None,
        max_steps=env_config['max_steps'],
        config=env_config
    )
    return Monitor(env)


def setup_directories():
    """Create necessary directories"""
    dirs = [
        TRAINING_CONFIG['best_model_path'],
        TRAINING_CONFIG['log_path'],
        './models/',
        './results/'
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)


def main():
    """Main training function"""
    print("="*60)
    print("RF Cavity Control - Reinforcement Learning Training")
    print("="*60)
    
    # Setup directories
    setup_directories()
    
    # Register environment
    gym.register(
        id='RFCavityControl-v1', 
        entry_point=lambda: create_env()
    )
    
    print(f"Environment registered: RFCavityControl-v1")
    print(f"Max steps per episode: {ENV_CONFIG['max_steps']}")
    print(f"Number of parallel environments: {TRAINING_CONFIG['n_envs']}")
    print(f"Device: {TRAINING_CONFIG['device']}")
    print()
    
    # Create vectorized environment
    vec_env = make_vec_env(
        lambda: create_env(),
        n_envs=TRAINING_CONFIG['n_envs']
    )
    
    # Setup callbacks
    eval_callback = EvalCallback(
        create_env(),
        best_model_save_path=TRAINING_CONFIG['best_model_path'],
        log_path=TRAINING_CONFIG['log_path'],
        eval_freq=TRAINING_CONFIG['eval_freq'],
        deterministic=True,
        verbose=1,
        n_eval_episodes=5  # Reduced for faster evaluation
    )
    
    early_stop_callback = EarlyStoppingCallback(
        reward_threshold=TRAINING_CONFIG['early_stop_threshold'],
        check_freq=TRAINING_CONFIG['early_stop_check_freq'],
        verbose=1
    )
    
    # Initialize PPO model
    print("Initializing PPO model...")
    model = PPO(
        TRAINING_CONFIG['policy'],
        vec_env,
        policy_kwargs={"net_arch": TRAINING_CONFIG['net_arch']},
        verbose=1,
        learning_rate=TRAINING_CONFIG['learning_rate'],
        n_steps=TRAINING_CONFIG['n_steps'],
        batch_size=TRAINING_CONFIG['batch_size'],
        n_epochs=TRAINING_CONFIG['n_epochs'],
        gamma=TRAINING_CONFIG['gamma'],
        gae_lambda=TRAINING_CONFIG['gae_lambda'],
        clip_range=TRAINING_CONFIG['clip_range'],
        ent_coef=TRAINING_CONFIG['ent_coef'],
        tensorboard_log=TRAINING_CONFIG['tensorboard_log'],
        device=TRAINING_CONFIG['device']
    )
    
    print(f"Model initialized with {model.policy}")
    print(f"Network architecture: {TRAINING_CONFIG['net_arch']}")
    print(f"Learning rate: {TRAINING_CONFIG['learning_rate']}")
    print()
    
    # Start training
    print("Starting training...")
    start_time = datetime.now()
    
    try:
        model.learn(
            total_timesteps=TRAINING_CONFIG['total_timesteps'],
            callback=[eval_callback, early_stop_callback],
            tb_log_name=f"ppo_rf_cavity_{start_time.strftime('%Y%m%d_%H%M%S')}"
        )
        
        print("Training completed successfully!")
        
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise
    
    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_path = f"./models/ppo_rf_cavity_final_{timestamp}"
    model.save(final_model_path)
    
    end_time = datetime.now()
    training_duration = end_time - start_time
    
    print()
    print("="*60)
    print("Training Summary")
    print("="*60)
    print(f"Start time: {start_time}")
    print(f"End time: {end_time}")
    print(f"Duration: {training_duration}")
    print(f"Total timesteps: {TRAINING_CONFIG['total_timesteps']:,}")
    print(f"Final model saved: {final_model_path}.zip")
    print(f"Best model saved: {TRAINING_CONFIG['best_model_path']}")
    print(f"Tensorboard logs: {TRAINING_CONFIG['tensorboard_log']}")
    
    # Close environment
    vec_env.close()


if __name__ == "__main__":
    main()
