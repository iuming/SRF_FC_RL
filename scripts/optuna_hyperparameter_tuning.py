import sys
import os
import chardet

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import argparse
import gymnasium as gym
import numpy as np
import yaml
import optuna
from stable_baselines3 import PPO
from envs.rf_environment import RFEnvironment
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

class CustomTensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # 检查action是否存在
        if 'action' not in self.locals:
            return True

        action = self.locals['action']
        original_env = self.training_env.get_attr('unwrapped')[0]

        # 记录失谐量（dw）的绝对值
        self.logger.record('env/dw_abs', np.abs(original_env.dw))

        # 记录动作值分布（处理向量化环境）
        if len(action.shape) == 2:  # 向量化环境返回 (n_envs, action_dim)
            action = action[0]  # 取第一个环境的动作
        self.logger.record('action/mean', np.mean(action))
        self.logger.record('action/std', np.std(action))

        return True

def make_env(config):
    def _init():
        env = RFEnvironment(config["env"])
        env = gym.wrappers.TimeLimit(env, max_episode_steps=config["env"]["max_episode_steps"])
        return env
    return _init

def objective(trial, config):
    # 定义超参数搜索空间
    ent_coef = trial.suggest_loguniform('ent_coef', 0.0001, 0.1)
    learning_rate = trial.suggest_loguniform('learning_rate', 0.0001, 0.01)
    n_steps = trial.suggest_int('n_steps', 128, 2048, step=128)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512])
    n_epochs = trial.suggest_int('n_epochs', 1, 10)
    gamma = trial.suggest_uniform('gamma', 0.9, 0.999)

    # 创建并行环境
    env = SubprocVecEnv([make_env(config) for _ in range(config["train"]["n_envs"])])

    # 初始化模型
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        ent_coef=ent_coef,
        tensorboard_log=config["logging"]["tensorboard_dir"],
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        seed=config["train"]["seed"],
        device="cuda"
    )

    # 执行训练
    model.learn(total_timesteps=config["train"]["total_timesteps"],
                callback=CustomTensorboardCallback(),
                tb_log_name="ppo_rf_piezo_compensation"
                )

    # 简单示例：评估模型，这里可以根据具体需求修改评估方式
    episode_rewards = []
    obs = env.reset()
    for _ in range(10):  # 运行 10 个回合进行评估
        episode_reward = 0
        while True:
            action, _ = model.predict(obs)
            try:
                # 尝试按 5 个值解包
                obs, reward, terminated, truncated, _ = env.step(action)
                done = np.logical_or(terminated, truncated)
            except ValueError:
                # 如果解包失败，按 4 个值解包
                obs, reward, done, _ = env.step(action)

            # 累加奖励
            episode_reward += np.sum(reward)

            # 检查是否所有子环境都完成
            if np.all(done):
                break

        episode_rewards.append(episode_reward / config["train"]["n_envs"])

    env.close()

    # 返回评估指标，这里使用平均回合奖励
    return np.mean(episode_rewards)

def main(args):
    # 检测配置文件编码
    with open(args.config_path, 'rb') as f:
        raw_data = f.read()
        encoding_result = chardet.detect(raw_data)
        encoding = encoding_result['encoding']
        print(f"Detected encoding: {encoding} (Confidence: {encoding_result['confidence']:.2f})")

    # 使用检测到的编码读取文件
    with open(args.config_path, 'r', encoding=encoding, errors='replace') as f:
        config = yaml.safe_load(f)

    # 创建 Optuna 研究对象
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, config), n_trials=10)  # 进行 10 次试验

    # 输出最佳超参数
    print("Best trial:")
    best_trial = study.best_trial
    print("  Value:", best_trial.value)
    print("  Params:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='config/config.yaml')
    args = parser.parse_args()
    main(args)    