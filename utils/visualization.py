import matplotlib.pyplot as plt
import numpy as np
import os

def plot_training_results(env, model):
    obs, _ = env.reset()
    rewards = []
    detuning_history = []
    actions = []

    original_env = env.unwrapped if hasattr(env, 'unwrapped') else env

    max_steps = getattr(env, '_max_episode_steps', 2048 * 500)
    
    for _ in range(max_steps):
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        detuning_history.append(original_env.dw / (2 * np.pi))
        actions.append(action[0])
        if terminated or truncated:
            break
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(rewards)
    plt.title("Reward Trajectory")
    plt.ylabel("Reward (-|Detuning|)")
    
    plt.subplot(3, 1, 2)
    plt.plot(detuning_history)
    plt.title("Detuning Trajectory")
    plt.ylabel("Detuning (Hz)")

    plt.subplot(3, 1, 3)
    plt.plot(actions)
    plt.title("Action Trajectory")
    plt.ylabel("Action")
    plt.xlabel("Time Step")
    
    plt.tight_layout()
    

    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    plt.savefig(f"{results_dir}/training_results_{model.__class__.__name__}_timesteps-{model.num_timesteps}_lr-{model.learning_rate}_nsteps-{model.n_steps}_batch-{model.batch_size}_epochs-{model.n_epochs}_gamma-{model.gamma}.png")

    plt.show()

