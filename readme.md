# 2048游戏并行DQN AI

一个使用PyTorch和Pygame实现的2048游戏，带有并行深度Q网络(DQN)人工智能训练功能。

## 功能特性

- 完整的2048游戏实现
- 并行DQN智能体训练
- 多环境并行训练
- GPU加速支持
- 交互式游戏界面
- 实时训练状态监控

## 系统要求

- Python 3.12
- PyTorch 1.9+
- Pygame 2.0+
- NumPy
- CUDA（可选，用于GPU加速）

## 安装步骤

### 1. 克隆项目
项目仓库地址:
- https: https://github.com/lhx77/game2048.git
- ssh: git@github.com:lhx77/game2048.git

### 2. 安装依赖
bash
pip install -r requirements.txt
## 项目结构
2048-dqn-game/
├── game2048.py
├── training.py
├── README.md
├── requirements.txt
└── models/
## 使用方法

### 1. 启动游戏
bash
python game2048.py
### 2. 游戏控制

#### 键盘控制
- **方向键**：上、下、左、右移动
- **R键**：重置游戏
- **ESC键**：退出游戏

#### 鼠标控制
- **AI Control**：切换AI控制模式
- **Start Train**：开始/停止AI训练
- **Reset Game**：重置当前游戏

### 3. AI训练

1. 点击**Start Train**按钮开始训练
2. 训练将在后台线程中进行
3. 模型会自动保存到`models_parallel/`目录
4. 点击**AI Control**让训练好的AI控制游戏

## 技术架构

### 1. 游戏引擎
- Pygame图形界面
- 4×4网格2048游戏逻辑
- 实时分数和步数统计

### 2. AI系统
- 并行DQN智能体(ParallelDQNAgent)
- 经验回放缓冲区
- 目标网络同步
- 多环境并行采样