import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from RFEnvironment import RFEnvironment

# 创建保存模型的目录
log_dir = "logs/"
os.makedirs(log_dir, exist_ok=True)

# 创建环境
env = RFEnvironment()

# 创建评估环境并使用 Monitor 包装
eval_env = Monitor(RFEnvironment())

# 定义回调函数，用于保存最优模型
eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir,
                             log_path=log_dir, eval_freq=1000,
                             deterministic=True, render=False)

# 创建 PPO 模型，指定使用 CPU 设备
model = PPO('MlpPolicy', env, verbose=1, device='cpu')

# 训练模型
model.learn(total_timesteps=100_000, callback=eval_callback)

# 保存最终模型
model.save(os.path.join(log_dir, "final_model"))

# 关闭环境
env.close()
eval_env.close()
