import matplotlib.pyplot as plt
import numpy as np

def plot_training_results(env, model):
    obs, _ = env.reset()
    rewards = []
    detuning_history = []

    original_env = env.unwrapped if hasattr(env, 'unwrapped') else env

    max_steps = getattr(env, '_max_episode_steps', 2048 * 500)
    
    for _ in range(max_steps):
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        detuning_history.append(np.abs(original_env.dw))
        if terminated or truncated:
            break
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(rewards)
    plt.title("Reward Trajectory")
    plt.ylabel("Reward (-|Detuning|)")
    
    plt.subplot(2, 1, 2)
    plt.plot(detuning_history)
    plt.title("Detuning Trajectory")
    plt.ylabel("Detuning (rad/s)")
    plt.xlabel("Time Step")
    
    plt.tight_layout()
    plt.show()