import sys
import os
import chardet

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import argparse
import gymnasium as gym
import numpy as np
import yaml
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
    # env = RFEnvironment(config["env"])
    # env = gym.wrappers.TimeLimit(env, max_episode_steps=config["env"]["max_episode_steps"])

    # 创建并行环境
    def make_env():
        def _init():
            env = RFEnvironment(config["env"])
            env = gym.wrappers.TimeLimit(env, max_episode_steps=config["env"]["max_episode_steps"])
            return env
        return _init

    env = SubprocVecEnv([make_env() for _ in range(config["train"]["n_envs"])])
    
    # 初始化模型
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        ent_coef=config["train"]["ent_coef"],
        tensorboard_log=config["logging"]["tensorboard_dir"],
        learning_rate=config["train"]["learning_rate"],
        n_steps=config["train"]["n_steps"],
        batch_size=config["train"]["batch_size"],
        n_epochs=config["train"]["n_epochs"],
        gamma=config["train"]["gamma"],
        seed=config["train"]["seed"],
        device=config["train"]["device"] 
    )
    
    # 执行训练
    model.learn(total_timesteps=config["train"]["total_timesteps"],
                callback=CustomTensorboardCallback(),
                tb_log_name="ppo_rf_piezo_compensation"
            )
    
    # 保存模型
    model.save(config["evaluate"]["model_path"])
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='config/config.yaml')
    args = parser.parse_args()
    main(args)