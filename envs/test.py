import gymnasium as gym
from RFEnvironment import RFEnvironment


def test_rf_environment():
    # 创建环境实例
    env = RFEnvironment()

    # 重置环境
    observation, info = env.reset()

    # 进行一定步数的交互
    num_steps = 1000000
    for _ in range(num_steps):
        # 随机选择一个动作
        # action = env.action_space.sample()
        action = [1.0]

        # 执行动作
        observation, reward, terminated, truncated, info = env.step(action)

        # 检查是否终止或截断
        if terminated or truncated:
            # 重置环境
            observation, info = env.reset()

    # 渲染环境
    env.render()

    # 关闭环境
    env.close()


if __name__ == "__main__":
    test_rf_environment()
