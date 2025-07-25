import gymnasium as gym
from stable_baselines3 import PPO
from RFEnvironment import RFEnvironment
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
import os
import numpy as np

# 加载训练好的模型
model_path = "logs/best_model"
model = PPO.load(model_path)

# 创建环境
env = RFEnvironment()

# 评估模型
num_eval_episodes = 1
episode_rewards = []
sig_vc_real_all = []
sig_vc_imag_all = []
sig_dw_all = []

for _ in range(num_eval_episodes):
    obs, _ = env.reset()
    episode_reward = 0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
        sig_vc_real_all.extend(env.sig_vc_real)
        sig_vc_imag_all.extend(env.sig_vc_imag)
        sig_dw_all.extend(env.sig_dw)
        done = terminated or truncated
    episode_rewards.append(episode_reward)

mean_reward = sum(episode_rewards) / num_eval_episodes
std_reward = (sum((x - mean_reward) ** 2 for x in episode_rewards) / num_eval_episodes) ** 0.5

print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# 绘制奖励分布
plt.figure(figsize=(10, 6))
plt.bar(range(num_eval_episodes), episode_rewards)
plt.axhline(y=mean_reward, color='r', linestyle='--', label=f'Mean Reward: {mean_reward:.2f}')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Episode Rewards during Evaluation')
plt.legend()

# 获取训练参数
# 这里 total_timesteps 不能从模型获取，需手动指定或通过其他方式记录
total_timesteps = 100000
# 从模型获取策略类型
policy_type = type(model.policy).__name__
learning_rate = model.learning_rate
n_steps = model.n_steps
batch_size = model.batch_size
n_epochs = model.n_epochs
gamma = model.gamma
ent_coef = model.ent_coef

# 生成图片文件名
image_filename = f'evaluation_total_timesteps_{total_timesteps}_policy_{policy_type}_lr_{learning_rate}_n_steps_{n_steps}_batch_size_{batch_size}_n_epochs_{n_epochs}_gamma_{gamma}_ent_coef_{ent_coef}.png'

# 创建保存图片的目录
image_dir = "evaluation_images"
os.makedirs(image_dir, exist_ok=True)

# 保存奖励分布图片
reward_image_path = os.path.join(image_dir, image_filename)
plt.savefig(reward_image_path)
print(f"Reward evaluation plot saved to {reward_image_path}")

# 显示奖励分布图片
plt.show()

# 进行一定步数的交互
num_steps = 1000000
for _ in range(num_steps):
    # 随机选择一个动作
    # action = env.action_space.sample()
    action, _ = model.predict(obs, deterministic=True)

    # 执行动作
    observation, reward, terminated, truncated, info = env.step(action)

    # 检查是否终止或截断
    if terminated or truncated:
        # 重置环境
        observation, info = env.reset()

# 渲染环境
env.render()

# 生成腔体相关量图片文件名
cavity_image_filename = f'cavity_evaluation_total_timesteps_{total_timesteps}_policy_{policy_type}_lr_{learning_rate}_n_steps_{n_steps}_batch_size_{batch_size}_n_epochs_{n_epochs}_gamma_{gamma}_ent_coef_{ent_coef}.png'
cavity_image_path = os.path.join(image_dir, cavity_image_filename)
plt.savefig(cavity_image_path)
print(f"Cavity evaluation plot saved to {cavity_image_path}")

# 显示腔体相关量图片
plt.show()

# 关闭环境
env.close()
