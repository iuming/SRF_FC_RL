# RF环境强化学习控制

## 项目结构
```
Gymnasium_rf/
├── config/       # 配置文件
├── envs/         # 环境实现
├── models/       # 训练模型
├── scripts/      # 训练/评估脚本
├── utils/        # 工具模块
├── README.md     # 项目说明
└── requirements.txt # 依赖清单
```

## 安装指南
1. 创建虚拟环境：
```bash
conda create -n RL python==3.13.2
conda activate RL
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 训练模型
```bash
python scripts/train.py --config config/config.yaml
```

## 评估模型
```bash
python scripts/evaluate.py --config config/config.yaml
```

## 注意事项
1. 确保安装CUDA环境以启用GPU加速
2. 配置文件中的参数可根据实际需求调整
3. 可视化结果会在评估完成后自动弹出
