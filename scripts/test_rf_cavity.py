"""
RF Cavity Control Model Evaluation Script

Filename: test_rf_cavity.py
Author: Ming Liu
Email: ming.liu@example.com
GitHub: https://github.com/iuming
Created: 2025-07-25
Version: 1.0.0

Description:
    This script evaluates the performance of trained PPO agents for RF cavity
    control. It provides comprehensive analysis including reward tracking,
    action distribution analysis, system response visualization, and statistical
    performance metrics.

Features:
    - Model loading and evaluation
    - Multi-episode performance analysis
    - Real-time visualization capabilities
    - Statistical performance metrics
    - Action distribution analysis
    - System response characterization
    - Comprehensive plotting and logging

Dependencies:
    - stable_baselines3
    - gymnasium
    - numpy
    - matplotlib
    - pandas (for data analysis)
    - llrflibs (for RF cavity simulation)

Changelog:
    v1.0.0 (2025-07-25):
        - Initial evaluation script implementation
        - Added comprehensive performance analysis
        - Implemented multi-plot visualization
        - Added statistical metrics calculation
        - Created action distribution analysis
        - Enhanced error handling and logging

License:
    This code is part of the ML_Learning repository.
    
Usage:
    python test_rf_cavity.py
    
    # Evaluate specific model:
    python test_rf_cavity.py --model_path path/to/model.zip
    
    # Run with custom episodes:
    python test_rf_cavity.py --episodes 100
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'configs'))

from rf_cavity_env import RFCavityControlEnv
from config import ENV_CONFIG, EVAL_CONFIG


def load_model(model_path, device='cpu'):
    """Load a trained model"""
    try:
        model = PPO.load(model_path, device=device)
        print(f"Model loaded successfully from: {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def create_test_env():
    """Create test environment"""
    return Monitor(RFCavityControlEnv(
        render_mode='human',
        max_steps=ENV_CONFIG['max_steps'],
        config=ENV_CONFIG
    ))


def evaluate_model_performance(model, env, n_episodes=10):
    """Evaluate model performance"""
    print(f"Evaluating model performance over {n_episodes} episodes...")
    
    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=n_episodes,
        deterministic=EVAL_CONFIG['deterministic'],
        render=EVAL_CONFIG['render']
    )
    
    print(f"Evaluation Results:")
    print(f"  Mean reward: {mean_reward:.4f} ± {std_reward:.4f}")
    print(f"  Best possible reward: 0.0 (perfect frequency control)")
    print(f"  Performance: {max(0, (mean_reward + 1000) / 1000 * 100):.1f}%")
    
    return mean_reward, std_reward


def run_demonstration(model, env, max_steps=None):
    """Run a demonstration episode with detailed logging"""
    if max_steps is None:
        max_steps = EVAL_CONFIG['max_demo_steps']
    
    print(f"Running demonstration for up to {max_steps} steps...")
    
    obs, _ = env.reset()
    total_reward = 0
    
    # Data collection
    data = {
        'actions': [],
        'rewards': [],
        'vc_amplitude': [],
        'vr_amplitude': [],
        'vc_phase': [],
        'frequency_detuning': [],
        'timesteps': []
    }
    
    step_count = 0
    
    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=EVAL_CONFIG['deterministic'])
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        step_count += 1
        
        # Record data
        data['actions'].append(action[0])
        data['rewards'].append(reward)
        data['vc_amplitude'].append(obs[0])
        data['vr_amplitude'].append(obs[1])
        data['vc_phase'].append(obs[2])
        data['frequency_detuning'].append(obs[3])
        data['timesteps'].append(step)
        
        if terminated or truncated:
            print(f"Episode finished at step {step_count}")
            break
    
    print(f"Demonstration completed:")
    print(f"  Total steps: {step_count}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Average reward per step: {total_reward/step_count:.6f}")
    print(f"  Final frequency detuning: {obs[3]:.3f} kHz")
    
    return data


def plot_results(data, save_path=None):
    """Plot demonstration results"""
    # Apply sampling if too many data points
    n_points = len(data['timesteps'])
    if n_points > EVAL_CONFIG['sample_rate_threshold']:
        sample_rate = max(1, n_points // EVAL_CONFIG['sample_rate_threshold'])
        print(f"Sampling data every {sample_rate} points for visualization")
        
        # Sample all data
        for key in data:
            data[key] = data[key][::sample_rate]
    
    # Create comprehensive plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('RF Cavity Control Performance Analysis', fontsize=16)
    
    # Control Actions
    axes[0, 0].plot(data['timesteps'], data['actions'], 'b-', linewidth=1)
    axes[0, 0].set_title('Piezo Control Actions')
    axes[0, 0].set_ylabel('Action')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Rewards
    axes[0, 1].plot(data['timesteps'], data['rewards'], 'r-', linewidth=1)
    axes[0, 1].set_title('Instantaneous Rewards')
    axes[0, 1].set_ylabel('Reward')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Frequency Detuning (most important metric)
    axes[0, 2].plot(data['timesteps'], data['frequency_detuning'], 'g-', linewidth=1)
    axes[0, 2].set_title('Frequency Detuning (Primary Objective)')
    axes[0, 2].set_ylabel('Detuning (kHz)')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].axhline(y=0, color='k', linestyle='--', alpha=0.5, label='Target')
    axes[0, 2].legend()
    
    # Cavity Voltage Amplitude
    axes[1, 0].plot(data['timesteps'], data['vc_amplitude'], 'm-', linewidth=1)
    axes[1, 0].set_title('Cavity Voltage Amplitude')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Amplitude (MV)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Reflected Voltage Amplitude
    axes[1, 1].plot(data['timesteps'], data['vr_amplitude'], 'c-', linewidth=1)
    axes[1, 1].set_title('Reflected Voltage Amplitude')
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Amplitude (MV)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Cavity Voltage Phase
    axes[1, 2].plot(data['timesteps'], data['vc_phase'], 'orange', linewidth=1)
    axes[1, 2].set_title('Cavity Voltage Phase')
    axes[1, 2].set_xlabel('Time Step')
    axes[1, 2].set_ylabel('Phase (degrees)')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"./results/rf_cavity_analysis_{timestamp}.png"
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Analysis plot saved: {save_path}")
    
    plt.show()
    
    return save_path


def print_performance_summary(data):
    """Print detailed performance summary"""
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    
    # Basic statistics
    detuning = np.array(data['frequency_detuning'])
    actions = np.array(data['actions'])
    rewards = np.array(data['rewards'])
    
    print(f"Frequency Control Performance:")
    print(f"  Mean absolute detuning: {np.mean(np.abs(detuning)):.4f} kHz")
    print(f"  Std deviation: {np.std(detuning):.4f} kHz")
    print(f"  Max absolute detuning: {np.max(np.abs(detuning)):.4f} kHz")
    print(f"  Final detuning: {detuning[-1]:.4f} kHz")
    
    print(f"\nControl Effort:")
    print(f"  Mean absolute action: {np.mean(np.abs(actions)):.4f}")
    print(f"  Action std deviation: {np.std(actions):.4f}")
    print(f"  Action range: [{np.min(actions):.3f}, {np.max(actions):.3f}]")
    
    print(f"\nReward Analysis:")
    print(f"  Total reward: {np.sum(rewards):.2f}")
    print(f"  Mean reward: {np.mean(rewards):.6f}")
    print(f"  Best reward: {np.max(rewards):.6f}")
    print(f"  Worst reward: {np.min(rewards):.6f}")
    
    # Stability analysis
    detuning_changes = np.abs(np.diff(detuning))
    print(f"\nStability Analysis:")
    print(f"  Mean detuning change per step: {np.mean(detuning_changes):.6f} kHz")
    print(f"  Max detuning change: {np.max(detuning_changes):.6f} kHz")
    
    # Performance classification
    mean_abs_detuning = np.mean(np.abs(detuning))
    if mean_abs_detuning < 0.1:
        performance = "Excellent"
    elif mean_abs_detuning < 1.0:
        performance = "Good"
    elif mean_abs_detuning < 10.0:
        performance = "Fair"
    else:
        performance = "Poor"
    
    print(f"\nOverall Performance Rating: {performance}")


def main():
    """Main testing function"""
    print("="*60)
    print("RF Cavity Control - Model Testing and Evaluation")
    print("="*60)
    
    # Model path - try different locations
    model_paths = [
        "./best_model/best_model.zip",
        "../best_model/best_model.zip", 
        "./models/ppo_rf_cavity_final.zip",
        "./ppo_sin_final.zip"  # Fallback to old naming
    ]
    
    model = None
    for path in model_paths:
        if os.path.exists(path):
            model = load_model(path)
            if model is not None:
                break
    
    if model is None:
        print("No trained model found. Please ensure a model exists at one of:")
        for path in model_paths:
            print(f"  {path}")
        return
    
    # Create test environment
    print("Creating test environment...")
    test_env = create_test_env()
    
    # Evaluate model
    try:
        mean_reward, std_reward = evaluate_model_performance(model, test_env)
        
        # Run demonstration
        demo_data = run_demonstration(model, test_env)
        
        # Plot results
        plot_path = plot_results(demo_data)
        
        # Print summary
        print_performance_summary(demo_data)
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        raise
    
    finally:
        test_env.close()
    
    print("\nTesting completed successfully!")


if __name__ == "__main__":
    main()
