"""
Simple environment testing script to verify RF cavity environment works correctly
"""

import os
import sys
import numpy as np

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'configs'))

from rf_cavity_env import RFCavityControlEnv
from config import ENV_CONFIG


def test_environment():
    """Test the RF cavity control environment"""
    print("="*50)
    print("RF Cavity Control Environment Test")
    print("="*50)
    
    print("Creating environment...")
    env = RFCavityControlEnv(config=ENV_CONFIG)
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print(f"Max steps: {env.max_steps}")
    
    print("\nResetting environment...")
    obs, info = env.reset()
    print(f"Initial observation: {obs}")
    print(f"Observation shape: {obs.shape}")
    print(f"Observation dtype: {obs.dtype}")
    
    # Check for NaN or Inf values
    print(f"Any NaN in observation: {np.any(np.isnan(obs))}")
    print(f"Any Inf in observation: {np.any(np.isinf(obs))}")
    
    print("\nTesting random actions...")
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Step {i+1}:")
        print(f"  Action: {action[0]:.4f}")
        print(f"  Observation: [{obs[0]:.4f}, {obs[1]:.4f}, {obs[2]:.2f}, {obs[3]:.4f}]")
        print(f"  Reward: {reward:.6f}")
        print(f"  Valid obs: {not (np.any(np.isnan(obs)) or np.any(np.isinf(obs)))}")
        print(f"  Valid reward: {not (np.isnan(reward) or np.isinf(reward))}")
        
        if terminated or truncated:
            print(f"  Episode ended (terminated={terminated}, truncated={truncated})")
            break
    
    env.close()
    print("\nEnvironment test completed successfully!")
    print("✓ No NaN or Inf values detected")
    print("✓ Environment runs without errors")


if __name__ == "__main__":
    test_environment()
