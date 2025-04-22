import sys
import os
import chardet

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import argparse
import gymnasium as gym
import numpy as np
import yaml
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from envs.rf_environment import RFEnvironment
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from utils.visualization import plot_training_results

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config", type=str, default="config/config.yaml", help="配置文件路径")
#     return parser.parse_args()

def main(args):
    # args = parse_args()
    
    # # 加载配置
    # with open(args.config, "r") as f:
    #     config = yaml.safe_load(f)

    with open(args.config_path, 'rb') as f:
        raw_data = f.read()
        encoding_result = chardet.detect(raw_data)
        encoding = encoding_result['encoding']
        print(f"Detected encoding: {encoding} (Confidence: {encoding_result['confidence']:.2f})")

    # 使用检测到的编码读取文件
    with open(args.config_path, 'r', encoding=encoding, errors='replace') as f:
        config = yaml.safe_load(f)
    
    # 创建环境
    env = RFEnvironment(config["env"])
    env = gym.wrappers.TimeLimit(env, max_episode_steps=config["env"]["max_episode_steps"])

    # 加载模型
    model = PPO.load(config["evaluate"]["model_path"], env=env)
    
    # 评估模型
    episode_rewards, episode_lengths = evaluate_policy(
    model, env, n_eval_episodes=config["evaluate"]["num_episodes"], return_episode_rewards=True
    )
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    print(f"平均奖励: {mean_reward.mean():.2f} ± {std_reward.std():.2f}")
    
    # 可视化结果
    plot_training_results(env, model)
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='config/config.yaml')
    args = parser.parse_args()
    main(args)