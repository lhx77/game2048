# training.py
import pygame
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os
import math

class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None

        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)

class DQNNetwork(nn.Module):
    """Dueling DQN for better value estimation"""
    def __init__(self, input_size=256, n_actions=4, hidden_size=512):
        super(DQNNetwork, self).__init__()

        # Feature extraction
        self.feature_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, n_actions)
        )

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        features = self.feature_layer(x)
        
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine value and advantages
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values

def get_one_hot(grid):
    """Convert 4x4 grid to 4x4x16 one-hot representation"""
    one_hot = np.zeros((4, 4, 16), dtype=np.float32)
    grid = np.array(grid).reshape(4, 4)
    for i in range(4):
        for j in range(4):
            val = grid[i, j]
            if val == 0:
                one_hot[i, j, 0] = 1
            else:
                # Map 2^k to index k (2->1, 4->2, ..., 2048->11, 4096->12)
                power = int(math.log2(val))
                idx = min(power, 15)
                one_hot[i, j, idx] = 1
    return one_hot.flatten()

def get_symmetries(grid, action, reward, next_grid, done):
    """Generate 8 symmetries for a given transition"""
    symmetries = []
    
    # Grid is 4x4
    curr_grid = np.array(grid).reshape(4, 4)
    curr_next = np.array(next_grid).reshape(4, 4)
    
    # Action mapping for rotations and flips
    # 0: Up, 1: Down, 2: Left, 3: Right
    
    def rotate_action(a, k):
        # Rotate action k times 90 deg clockwise
        for _ in range(k):
            if a == 0: a = 3 # Up -> Right
            elif a == 3: a = 1 # Right -> Down
            elif a == 1: a = 2 # Down -> Left
            elif a == 2: a = 0 # Left -> Up
        return a

    def flip_action_h(a):
        if a == 2: return 3 # Left -> Right
        if a == 3: return 2 # Right -> Left
        return a

    # 4 rotations
    for k in range(4):
        g = np.rot90(curr_grid, k, axes=(1, 0)) # Clockwise rotation
        ng = np.rot90(curr_next, k, axes=(1, 0))
        a = rotate_action(action, k)
        
        symmetries.append((get_one_hot(g), a, reward, get_one_hot(ng), done))
        
        # Horizontal flip of the rotated grid
        g_flip = np.fliplr(g)
        ng_flip = np.fliplr(ng)
        a_flip = flip_action_h(a)
        
        symmetries.append((get_one_hot(g_flip), a_flip, reward, get_one_hot(ng_flip), done))
        
    return symmetries

class DQNAgent:
    """强化训练以达到2048的DQN智能体"""
    def __init__(self, state_size=256, n_actions=4, device='cpu'):
        self.device = device
        self.state_size = state_size
        self.n_actions = n_actions

        # 优化参数
        self.gamma = 0.99
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.epsilon_decay = 0.9997 # 稍微减慢衰减，让前期探索更充分
        self.batch_size = 128 # 进一步减小 Batch Size 以适配弱 CPU
        self.learning_rate = 1e-4
        self.target_update = 200
        self.memory_size = 200000
        self.update_every = 16 # 统一更新频率

        # 网络
        self.policy_net = DQNNetwork(state_size, n_actions, hidden_size=512).to(device)
        self.target_net = DQNNetwork(state_size, n_actions, hidden_size=512).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

        # 记忆
        self.memory = ReplayBuffer(self.memory_size)

        # 训练状态
        self.epsilon = self.epsilon_start
        self.training_steps = 0
        self.update_count = 0

        # 统计
        self.episode_rewards = []
        self.episode_scores = []
        self.episode_max_tiles = []
        self.episode_lengths = []
        self.losses = []

        # 里程碑记录
        self.best_max_tile = 0
        self.consecutive_improvements = 0
        self.episodes_since_last_improvement = 0

        # 名称
        self.name = "DuelingDQN-2048"

        # 训练阶段
        self.training_stage = 1  # 1: 基础, 2: 中级, 3: 高级
        self.stage_thresholds = {1: 256, 2: 512, 3: 1024}

    def select_action(self, state, legal_actions=None, eval_mode=False):
        """使用epsilon-greedy策略选择动作，并支持动作掩码"""
        if legal_actions is None:
            legal_actions = [0, 1, 2, 3]
            
        if not legal_actions:
            return 0 # 应该不会发生，因为 done 会先触发

        if not eval_mode and random.random() < self.epsilon:
            return random.choice(legal_actions)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)[0]
            
            # 动作掩码：将非法动作的Q值设为极小值
            masked_q = torch.full_like(q_values, -1e9)
            for a in legal_actions:
                masked_q[a] = q_values[a]
                
            return masked_q.argmax().item()

    def compute_td_loss(self, batch):
        """计算时序差分损失"""
        states, actions, rewards, next_states, dones = batch

        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)

        # 当前Q值
        current_q = self.policy_net(states_tensor).gather(1, actions_tensor.unsqueeze(1))

        # 双Q学习
        with torch.no_grad():
            next_actions = self.policy_net(next_states_tensor).argmax(1, keepdim=True)
            next_q = self.target_net(next_states_tensor).gather(1, next_actions)
            target_q = rewards_tensor.unsqueeze(1) + (1 - dones_tensor.unsqueeze(1)) * self.gamma * next_q

        # 计算损失
        loss = nn.MSELoss()(current_q.squeeze(), target_q.squeeze())

        return loss

    def update(self):
        """更新网络"""
        batch = self.memory.sample(self.batch_size)
        if batch is None:
            return 0.0

        # 数据增强：对Batch中的状态进行随机变换（旋转/翻转）
        # 注意：这里需要相应地调整动作。对于2048，这是一个复杂的操作。
        # 简单起见，我们先不在这里做动作变换，而是直接计算Loss。
        # 如果要做完整的对称性，需要在存储经验时或者采样后对(s, a, r, s', d)整体做变换。

        # 计算损失
        loss = self.compute_td_loss(batch)

        # 优化
        self.optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        # 衰减epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # 更新目标网络
        self.update_count += 1
        if self.update_count % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def adapt_training_params(self, recent_performance):
        """根据表现调整训练参数"""
        if len(recent_performance) < 20:
            return

        avg_max = np.mean(recent_performance[-20:])

        # 根据表现调整训练阶段
        if avg_max >= 1024 and self.training_stage < 3:
            self.training_stage = 3
            print(f"  Advanced to training stage 3 (target: 2048)")
        elif avg_max >= 512 and self.training_stage < 2:
            self.training_stage = 2
            print(f"  Advanced to training stage 2 (target: 1024)")
        elif avg_max >= 256 and self.training_stage < 1:
            self.training_stage = 1
            print(f"  Advanced to training stage 1 (target: 512)")

    def save(self, path):
        """保存模型"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_steps': self.training_steps,
            'best_max_tile': self.best_max_tile,
            'training_stage': self.training_stage,
            'stats': {
                'rewards': self.episode_rewards,
                'scores': self.episode_scores,
                'max_tiles': self.episode_max_tiles,
                'lengths': self.episode_lengths,
                'losses': self.losses
            },
            'name': self.name
        }, path)
        print(f"Model saved to {path}")

    def load(self, path):
        """加载模型"""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.training_steps = checkpoint['training_steps']
            self.best_max_tile = checkpoint.get('best_max_tile', 0)
            self.training_stage = checkpoint.get('training_stage', 1)

            if 'stats' in checkpoint:
                stats = checkpoint['stats']
                self.episode_rewards = stats.get('rewards', [])
                self.episode_scores = stats.get('scores', [])
                self.episode_max_tiles = stats.get('max_tiles', [])
                self.episode_lengths = stats.get('lengths', [])
                self.losses = stats.get('losses', [])

            if 'name' in checkpoint:
                self.name = checkpoint['name']

            print(f"Model loaded from {path}")
            return True
        else:
            print(f"No model found at {path}")
            return False

    def agent_name(self):
        """获取智能体名称"""
        return self.name

    def set_name(self, name):
        """设置智能体名称"""
        self.name = name

class TrainingGame:
    """专门为2048优化的训练游戏"""
    def __init__(self):
        self.grid_size = 4
        self.grid = [[0] * 4 for _ in range(4)]
        self.score = 0
        self.game_over = False
        self.win = False
        self.steps = 0
        self.consecutive_no_merge = 0

        # 添加两个初始方块
        self.add_random_tile()
        self.add_random_tile()

        # 初始化last_max_tile为当前最大方块
        self.last_max_tile = self.get_max_tile()

    def get_observation(self):
        """获取一热编码观察状态"""
        return get_one_hot(self.grid)

    def get_enhanced_observation(self):
        """增强的状态观察"""
        obs = np.array(self.grid, dtype=np.float32)

        # 基础特征
        obs_without_zero = np.where(obs == 0, 1, obs)
        obs_log = np.log2(obs_without_zero)
        obs_norm = obs_log / 11.0

        # 空格特征
        empty_mask = (obs == 0).astype(np.float32)

        # 潜在合并特征
        merge_potential = self._get_merge_potential()

        # 单调性特征
        monotonic = self._get_monotonicity()

        # 合并所有特征
        features = np.concatenate([
            obs_norm.flatten(),
            empty_mask.flatten(),
            merge_potential,
            monotonic
        ])

        return features

    def _get_merge_potential(self):
        """计算潜在合并机会"""
        potential = np.zeros(16, dtype=np.float32)

        idx = 0
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                val = self.grid[i][j]
                if val == 0:
                    potential[idx] = 0
                else:
                    # 检查四个方向的潜在合并
                    merge_count = 0
                    if j < 3 and self.grid[i][j+1] == val:
                        merge_count += 1
                    if i < 3 and self.grid[i+1][j] == val:
                        merge_count += 1
                    if j > 0 and self.grid[i][j-1] == val:
                        merge_count += 1
                    if i > 0 and self.grid[i-1][j] == val:
                        merge_count += 1
                    potential[idx] = merge_count
                idx += 1

        return potential

    def _get_monotonicity(self):
        """计算单调性"""
        features = []

        # 行单调性
        for i in range(self.grid_size):
            row = [self.grid[i][j] for j in range(self.grid_size) if self.grid[i][j] != 0]
            if len(row) >= 2:
                # 检查是否递减
                decreasing = all(row[k] >= row[k+1] for k in range(len(row)-1))
                features.append(1.0 if decreasing else 0.0)
            else:
                features.append(0.0)

        return np.array(features, dtype=np.float32)

    def add_random_tile(self):
        """添加随机方块"""
        empty_cells = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.grid[i][j] == 0:
                    empty_cells.append((i, j))

        if empty_cells:
            i, j = random.choice(empty_cells)
            self.grid[i][j] = 2 if random.random() < 0.9 else 4
            return True
        return False

    def compress(self, line):
        """压缩行"""
        return [x for x in line if x != 0]

    def merge(self, line):
        """合并行"""
        score = 0
        result = []
        i = 0
        while i < len(line):
            if i < len(line) - 1 and line[i] == line[i + 1]:
                merged = line[i] * 2
                result.append(merged)
                score += merged
                i += 2
            else:
                result.append(line[i])
                i += 1
        return result, score

    def move(self, direction):
        """移动"""
        moved = False
        total_score = 0
        new_grid = [[0] * self.grid_size for _ in range(self.grid_size)]

        if direction == 0:  # 上
            for col in range(self.grid_size):
                column = [self.grid[row][col] for row in range(self.grid_size)]
                compressed = self.compress(column)
                merged, score = self.merge(compressed)
                total_score += score
                for row in range(len(merged)):
                    new_grid[row][col] = merged[row]
                    if column[row] != merged[row]:
                        moved = True

        elif direction == 1:  # 下
            for col in range(self.grid_size):
                column = [self.grid[row][col] for row in range(self.grid_size)]
                compressed = self.compress(column[::-1])
                merged, score = self.merge(compressed)
                total_score += score
                merged = merged[::-1] + [0] * (self.grid_size - len(merged))
                for row in range(self.grid_size):
                    new_grid[self.grid_size - 1 - row][col] = merged[row]
                    if column[row] != merged[self.grid_size - 1 - row]:
                        moved = True

        elif direction == 2:  # 左
            for row in range(self.grid_size):
                compressed = self.compress(self.grid[row])
                merged, score = self.merge(compressed)
                total_score += score
                new_grid[row] = merged + [0] * (self.grid_size - len(merged))
                if self.grid[row] != new_grid[row]:
                    moved = True

        elif direction == 3:  # 右
            for row in range(self.grid_size):
                compressed = self.compress(self.grid[row][::-1])
                merged, score = self.merge(compressed)
                total_score += score
                merged = merged[::-1] + [0] * (self.grid_size - len(merged))
                new_grid[row] = merged
                if self.grid[row] != new_grid[row]:
                    moved = True

        self.grid = new_grid
        return moved, total_score

    def can_move(self):
        """检查是否可以移动"""
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.grid[i][j] == 0:
                    return True
                val = self.grid[i][j]
                if j < self.grid_size - 1 and self.grid[i][j+1] == val:
                    return True
                if i < self.grid_size - 1 and self.grid[i+1][j] == val:
                    return True
        return False

    def get_legal_actions(self):
        """获取当前合法的动作列表"""
        legal_actions = []
        for a in range(4):
            # 模拟移动，看是否有变化
            # 这里为了效率，可以实现一个轻量级的 check_move
            if self.check_move(a):
                legal_actions.append(a)
        return legal_actions

    def check_move(self, direction):
        """检查某个方向是否可以移动"""
        if direction == 0:  # 上
            for col in range(self.grid_size):
                for row in range(self.grid_size - 1):
                    if self.grid[row][col] == 0 and self.grid[row+1][col] != 0:
                        return True
                    if self.grid[row][col] != 0 and self.grid[row][col] == self.grid[row+1][col]:
                        return True
        elif direction == 1:  # 下
            for col in range(self.grid_size):
                for row in range(self.grid_size - 1, 0, -1):
                    if self.grid[row][col] == 0 and self.grid[row-1][col] != 0:
                        return True
                    if self.grid[row][col] != 0 and self.grid[row][col] == self.grid[row-1][col]:
                        return True
        elif direction == 2:  # 左
            for row in range(self.grid_size):
                for col in range(self.grid_size - 1):
                    if self.grid[row][col] == 0 and self.grid[row][col+1] != 0:
                        return True
                    if self.grid[row][col] != 0 and self.grid[row][col] == self.grid[row][col+1]:
                        return True
        elif direction == 3:  # 右
            for row in range(self.grid_size):
                for col in range(self.grid_size - 1, 0, -1):
                    if self.grid[row][col] == 0 and self.grid[row][col-1] != 0:
                        return True
                    if self.grid[row][col] != 0 and self.grid[row][col] == self.grid[row][col-1]:
                        return True
        return False

    def get_max_tile(self):
        """获取最大方块"""
        return max(max(row) for row in self.grid)

    def count_empty_cells(self):
        """统计空格数量"""
        return sum(1 for row in self.grid for val in row if val == 0)

    def calculate_smoothness(self):
        """计算平滑度"""
        smoothness = 0.0
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                val = self.grid[i][j]
                if val == 0:
                    continue

                # 检查右边
                if j < 3 and self.grid[i][j+1] != 0:
                    right_val = self.grid[i][j+1]
                    smoothness -= abs(math.log2(val) - math.log2(right_val))

                # 检查下边
                if i < 3 and self.grid[i+1][j] != 0:
                    down_val = self.grid[i+1][j]
                    smoothness -= abs(math.log2(val) - math.log2(down_val))

        return smoothness

    def calculate_reward(self, old_max, new_max, moved, game_over, win, score_gain, empty_cells, action):
        """专门为达到2048设计的奖励函数 - 强惩罚差异版本"""
        reward = 0.0

        # 1. 合并奖励（最重要）- 大幅奖励合并
        if score_gain > 0:
            # 合并的基础奖励
            base_merge_reward = math.log2(score_gain) * 20.0  # 大幅提高合并奖励
            reward += base_merge_reward

            # 根据合并大小给予指数级奖励
            if score_gain >= 1024:  # 512+512
                reward += 200.0
            elif score_gain >= 512:  # 256+256
                reward += 100.0
            elif score_gain >= 256:  # 128+128
                reward += 50.0
            elif score_gain >= 128:  # 64+64
                reward += 25.0
            elif score_gain >= 64:  # 32+32
                reward += 12.0
            elif score_gain >= 32:  # 16+16
                reward += 6.0
            elif score_gain >= 16:  # 8+8
                reward += 3.0
            elif score_gain >= 8:  # 4+4
                reward += 1.5
            else:  # 2+2
                reward += 0.5

        # 2. 最大方块奖励
        if new_max > old_max:
            if old_max > 0:
                increase = math.log2(new_max) - math.log2(old_max)
            else:
                increase = math.log2(new_max)

            reward += increase * 50.0  # 大幅提高最大方块增长奖励

            # 里程碑奖励
            if new_max >= 2048:
                reward += 1000.0
            elif new_max >= 1024:
                reward += 400.0
            elif new_max >= 512:
                reward += 200.0
            elif new_max >= 256:
                reward += 100.0
            elif new_max >= 128:
                reward += 40.0
            elif new_max >= 64:
                reward += 20.0
            elif new_max >= 32:
                reward += 10.0
            elif new_max >= 16:
                reward += 5.0
            elif new_max >= 8:
                reward += 2.0
            elif new_max >= 4:
                reward += 1.0

        # 3. 块与块之间的差值惩罚（关键改进）- 大幅增加惩罚系数
        smoothness_penalty = self._calculate_enhanced_smoothness_penalty()
        reward += smoothness_penalty  # 系数增加到1.0，直接使用惩罚值

        # 4. 空格奖励 - 鼓励保持一定空格
        if empty_cells > 0:
            empty_reward = math.log(empty_cells + 1) * 1.2
            reward += empty_reward
        else:
            reward -= 2.0  # 没有空格的惩罚

        # 5. 单调性奖励 - 增加奖励
        monotonic_bonus = self._calculate_enhanced_monotonicity()
        reward += monotonic_bonus * 0.8

        # 6. 角落奖励 - 鼓励最大方块在角落
        corner_bonus = self._calculate_corner_bonus()
        reward += corner_bonus * 1.0

        # 7. 移动有效性奖励/惩罚
        if moved:
            reward += 0.5
            self.consecutive_no_merge = 0
        else:
            reward -= 1.0  # 大幅增加无效移动惩罚
            self.consecutive_no_merge += 1

            # 连续无效移动大幅增加惩罚
            if self.consecutive_no_merge > 1:
                penalty = 2.0 * self.consecutive_no_merge
                reward -= penalty

        # 8. 游戏结束惩罚/奖励
        if game_over:
            if new_max >= 2048:
                reward += 2000.0
            else:
                # 根据最终最大方块给予不同惩罚
                if new_max >= 1024:
                    reward -= 20.0
                elif new_max >= 512:
                    reward -= 40.0
                elif new_max >= 256:
                    reward -= 80.0
                elif new_max >= 128:
                    reward -= 120.0
                else:
                    reward -= 200.0

        if win:
            reward += 5000.0  # 达到2048的极大奖励

        return reward

    def _calculate_enhanced_smoothness_penalty(self):
        """计算增强的平滑度惩罚 - 大幅惩罚相邻块差异"""
        smoothness_penalty = 0.0
        total_penalty_factor = 0.0

        # 计算行平滑度惩罚
        for i in range(self.grid_size):
            for j in range(self.grid_size - 1):
                val1 = self.grid[i][j]
                val2 = self.grid[i][j + 1]
                if val1 > 0 and val2 > 0:
                    # 计算相邻块的数值差异
                    if val1 > 0 and val2 > 0:
                        diff = abs(math.log2(val1) - math.log2(val2))
                        # 差异越大，惩罚越大，使用平方惩罚
                        penalty = -diff * diff * 0.3
                        smoothness_penalty += penalty
                        total_penalty_factor += 1.0
                    elif val1 > 0 or val2 > 0:
                        # 一个为空，一个非空，给予中等惩罚
                        smoothness_penalty -= 0.5
                        total_penalty_factor += 1.0

        # 计算列平滑度惩罚
        for j in range(self.grid_size):
            for i in range(self.grid_size - 1):
                val1 = self.grid[i][j]
                val2 = self.grid[i + 1][j]
                if val1 > 0 and val2 > 0:
                    if val1 > 0 and val2 > 0:
                        diff = abs(math.log2(val1) - math.log2(val2))
                        # 差异越大，惩罚越大，使用平方惩罚
                        penalty = -diff * diff * 0.3
                        smoothness_penalty += penalty
                        total_penalty_factor += 1.0
                    elif val1 > 0 or val2 > 0:
                        # 一个为空，一个非空，给予中等惩罚
                        smoothness_penalty -= 0.5
                        total_penalty_factor += 1.0

        # 计算棋盘整体梯度惩罚
        gradient_penalty = self._calculate_gradient_penalty()
        smoothness_penalty += gradient_penalty * 1.5

        return smoothness_penalty

    def _calculate_gradient_penalty(self):
        """计算棋盘梯度惩罚 - 惩罚不理想的梯度方向"""
        penalty = 0.0

        # 理想情况：棋盘应该呈蛇形排列，从左上到右下数值递增
        # 计算每行和每列的梯度一致性

        # 行梯度：从左到右应该递减（或递增）
        for i in range(self.grid_size):
            row_values = []
            for j in range(self.grid_size):
                if self.grid[i][j] > 0:
                    row_values.append(self.grid[i][j])

            if len(row_values) >= 2:
                # 检查是否严格递减
                decreasing = all(row_values[k] >= row_values[k+1] for k in range(len(row_values)-1))
                # 检查是否严格递增
                increasing = all(row_values[k] <= row_values[k+1] for k in range(len(row_values)-1))

                if not decreasing and not increasing:
                    # 既不严格递减也不严格递增，给予惩罚
                    penalty -= 1.0
                elif decreasing:
                    # 严格递减，给予奖励
                    penalty += 0.5
                # 严格递增给予较小奖励

        # 列梯度：从上到下应该递减
        for j in range(self.grid_size):
            col_values = []
            for i in range(self.grid_size):
                if self.grid[i][j] > 0:
                    col_values.append(self.grid[i][j])

            if len(col_values) >= 2:
                # 检查是否严格递减
                decreasing = all(col_values[k] >= col_values[k+1] for k in range(len(col_values)-1))

                if not decreasing:
                    # 不严格递减，给予惩罚
                    penalty -= 1.0
                else:
                    # 严格递减，给予奖励
                    penalty += 0.5

        return penalty

    def _calculate_enhanced_monotonicity(self):
        """计算增强的单调性奖励"""
        bonus = 0.0
        max_tile = self.get_max_tile()

        # 行单调性
        for i in range(self.grid_size):
            row = [self.grid[i][j] for j in range(self.grid_size) if self.grid[i][j] != 0]
            if len(row) >= 2:
                # 检查是否严格递减（推荐策略：大方块在右）
                is_strict_decreasing = all(row[k] > row[k+1] for k in range(len(row)-1))
                is_decreasing = all(row[k] >= row[k+1] for k in range(len(row)-1))

                if is_strict_decreasing:
                    bonus += 2.0
                elif is_decreasing:
                    bonus += 1.0
                else:
                    # 检查是否递增
                    is_increasing = all(row[k] <= row[k+1] for k in range(len(row)-1))
                    if is_increasing:
                        bonus += 0.2
                    else:
                        # 既不递增也不递减，惩罚
                        bonus -= 0.5

        # 列单调性
        for j in range(self.grid_size):
            col = [self.grid[i][j] for i in range(self.grid_size) if self.grid[i][j] != 0]
            if len(col) >= 2:
                # 检查是否严格递减（推荐策略：大方块在下）
                is_strict_decreasing = all(col[k] > col[k+1] for k in range(len(col)-1))
                is_decreasing = all(col[k] >= col[k+1] for k in range(len(col)-1))

                if is_strict_decreasing:
                    bonus += 2.0
                elif is_decreasing:
                    bonus += 1.0
                else:
                    # 检查是否递增
                    is_increasing = all(col[k] <= col[k+1] for k in range(len(col)-1))
                    if is_increasing:
                        bonus += 0.2
                    else:
                        # 既不递增也不递减，惩罚
                        bonus -= 0.5

        return bonus

    def _calculate_corner_bonus(self):
        """计算角落奖励 - 鼓励最大方块在角落"""
        max_tile = self.get_max_tile()
        if max_tile < 16:  # 小方块不强调角落位置
            return 0.0

        # 找到最大方块的位置
        max_positions = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.grid[i][j] == max_tile:
                    max_positions.append((i, j))

        if not max_positions:
            return 0.0

        # 计算到四个角落的最小距离
        corners = [(0, 0), (0, 3), (3, 0), (3, 3)]
        min_distance = float('inf')
        for (i, j) in max_positions:
            for (ci, cj) in corners:
                distance = abs(i - ci) + abs(j - cj)
                if distance < min_distance:
                    min_distance = distance

        # 距离越近，奖励越高
        corner_bonus = 1.0 / (1.0 + min_distance)
        return corner_bonus

    def step(self, action):
        """执行一步"""
        if self.game_over:
            return None, 0.0, True

        self.steps += 1
        self.last_grid = [row[:] for row in self.grid] # 保存移动前的网格
        old_max = self.last_max_tile  # 保存移动前的最大方块

        moved, score_gain = self.move(action)
        self.score += score_gain

        if moved:
            self.add_random_tile()

        new_max = self.get_max_tile()
        self.last_max_tile = new_max  # 更新为新的最大方块

        if new_max >= 2048 and not self.win:
            self.win = True

        if not self.can_move():
            self.game_over = True

        empty_cells = self.count_empty_cells()
        reward = self.calculate_reward(
            old_max, new_max, moved, self.game_over,
            self.win, score_gain, empty_cells, action
        )

        return self.get_observation(), reward, self.game_over

    def reset(self):
        """重置游戏"""
        self.grid = [[0] * self.grid_size for _ in range(self.grid_size)]
        self.score = 0
        self.game_over = False
        self.win = False
        self.steps = 0
        self.consecutive_no_merge = 0

        # 添加两个初始方块
        self.add_random_tile()
        self.add_random_tile()

        # 初始化last_max_tile为当前最大方块
        self.last_max_tile = self.get_max_tile()
        self.last_grid = [row[:] for row in self.grid]
        return self.get_observation()

def train_agent(episodes=3000, save_dir='models', device='cpu'):
    """训练达到2048的强化训练"""
    os.makedirs(save_dir, exist_ok=True)

    print(f"Starting 2048 training for {episodes} episodes...")
    print(f"Goal: Reach 2048 tile consistently")
    print(f"Using device: {device}")
    print(f"Save directory: {save_dir}")

    # 创建智能体
    agent = DQNAgent(device=device)

    # 训练统计
    episode_start_time = time.time()
    recent_max_tiles = []
    best_max_tile = 0
    plateau_counter = 0

    for episode in range(1, episodes + 1):
        # 创建游戏
        game = TrainingGame()

        # 重置
        state = game.reset()

        episode_reward = 0
        episode_score = 0
        episode_max_tile = 0
        episode_length = 0

        while True:
            # 自适应探索
            if episode % 50 == 0 and len(recent_max_tiles) > 0:
                agent.adapt_training_params(recent_max_tiles)

            # 选择动作 - 在训练初期更倾向于探索
            legal_actions = game.get_legal_actions()
            if episode < 100:  # 前100轮增加探索
                action = random.choice(legal_actions) if legal_actions else 0
            else:
                action = agent.select_action(state, legal_actions=legal_actions, eval_mode=False)

            # 执行动作
            next_state, reward, done = game.step(action)

            # 更新统计
            episode_score = game.score
            current_max = game.get_max_tile()
            episode_max_tile = max(episode_max_tile, current_max)

            # 存储经验 - 使用对称性增强
            if hasattr(game, 'last_grid'):
                s_list = get_symmetries(game.last_grid, action, reward, game.grid, done)
                for s, a, r, ns, d in s_list:
                    agent.memory.push(s, a, r, ns, d)
            else:
                agent.memory.push(state, action, reward, next_state, done)

            # 定期更新网络
            agent.training_steps += 1
            if agent.training_steps % agent.update_every == 0:
                loss = agent.update()
                if loss > 0:
                    agent.losses.append(loss)

            # 更新状态
            state = next_state
            episode_reward += reward
            episode_length += 1

            if done:
                break

        # 记录统计
        agent.episode_rewards.append(episode_reward)
        agent.episode_scores.append(episode_score)
        agent.episode_max_tiles.append(episode_max_tile)
        agent.episode_lengths.append(episode_length)

        # 更新最近表现
        recent_max_tiles.append(episode_max_tile)
        if len(recent_max_tiles) > 100:
            recent_max_tiles.pop(0)

        # 检查突破
        if episode_max_tile > best_max_tile:
            best_max_tile = episode_max_tile
            agent.best_max_tile = best_max_tile
            plateau_counter = 0

            # 保存里程碑模型
            if episode_max_tile >= 128:
                agent.save(f"{save_dir}/milestone_{episode_max_tile}_ep{episode}.pth")
                print(f"  MILESTONE! Max tile: {episode_max_tile} (Episode {episode})")

                # 达到2048时提前结束训练
                if episode_max_tile >= 2048:
                    print(f"  GOAL ACHIEVED! 2048 reached at episode {episode}")
                    break
        else:
            plateau_counter += 1

        # 如果长时间无进展，重置探索
        if plateau_counter > 200 and episode > 500:
            agent.epsilon = min(0.5, agent.epsilon + 0.2)  # 大幅增加探索
            plateau_counter = 0
            print(f"  Plateau detected, increasing exploration: epsilon={agent.epsilon:.3f}")

        # 保存最佳模型
        if episode % 200 == 0:
            avg_recent = np.mean(recent_max_tiles[-50:]) if len(recent_max_tiles) >= 50 else 0
            if avg_recent >= 128:
                agent.save(f"{save_dir}/checkpoint_ep{episode}_avg{avg_recent:.0f}.pth")

        # 打印进度
        if episode % 20 == 0 or episode_max_tile >= 64:
            avg_reward = np.mean(agent.episode_rewards[-20:]) if len(agent.episode_rewards) >= 20 else episode_reward
            avg_score = np.mean(agent.episode_scores[-20:]) if len(agent.episode_scores) >= 20 else episode_score
            avg_max = np.mean(agent.episode_max_tiles[-20:]) if len(agent.episode_max_tiles) >= 20 else episode_max_tile

            print(f"Ep {episode:4d}/{episodes} | "
                  f"R:{episode_reward:7.1f} | "
                  f"S:{episode_score:6.0f} | "
                  f"Max:{episode_max_tile:4d} | "
                  f"AvgM:{avg_max:5.1f} | "
                  f"Best:{best_max_tile:4d} | "
                  f"Stage:{agent.training_stage} | "
                  f"Eps:{agent.epsilon:.3f} | "
                  f"Mem:{len(agent.memory):5d}")

    # 最终保存
    agent.save(f"{save_dir}/2048_dqn_final.pth")

    training_time = time.time() - episode_start_time
    print(f"\nTraining completed in {training_time/60:.1f} minutes")
    print(f"Best max tile achieved: {best_max_tile}")

    # 打印统计摘要
    if len(agent.episode_max_tiles) > 0:
        tiles = np.array(agent.episode_max_tiles)
        print(f"\nMax tile distribution:")
        for tile in [64, 128, 256, 512, 1024, 2048]:
            count = np.sum(tiles >= tile)
            percentage = count / len(tiles) * 100
            print(f"  ≥{tile:4d}: {count:4d} episodes ({percentage:.1f}%)")

    return agent

def evaluate_agent(model_path, num_games=20, device='cpu'):
    """评估智能体表现"""
    print(f"\nEvaluating agent from {model_path}")

    agent = DQNAgent(device=device)
    if not agent.load(model_path):
        print("Failed to load model")
        return

    scores = []
    max_tiles = []
    steps_list = []

    for game_num in range(num_games):
        game = TrainingGame()
        state = game.reset()

        steps = 0
        while not game.game_over:
            action = agent.select_action(state, eval_mode=True)
            state, _, _ = game.step(action)
            steps += 1

        scores.append(game.score)
        max_tiles.append(game.get_max_tile())
        steps_list.append(steps)

        if (game_num + 1) % 5 == 0:
            print(f"  Game {game_num+1:2d}/{num_games}: Score={game.score:6d}, Max={game.get_max_tile():4d}, Steps={steps:3d}")

    # 打印评估结果
    print(f"\nEvaluation Results ({num_games} games):")
    print(f"Average Score: {np.mean(scores):.0f} ± {np.std(scores):.0f}")
    print(f"Best Score: {max(scores)}")
    print(f"Average Max Tile: {np.mean(max_tiles):.1f}")
    print(f"Best Max Tile: {max(max_tiles)}")
    print(f"Average Steps: {np.mean(steps_list):.1f}")

    # 最大方块统计
    print(f"\nMax Tile Distribution:")
    for tile in [64, 128, 256, 512, 1024, 2048]:
        count = sum(1 for t in max_tiles if t >= tile)
        percentage = count / num_games * 100
        print(f"  ≥{tile:4d}: {count:2d}/{num_games} ({percentage:.1f}%)")

    return scores, max_tiles

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='2048 DQN Agent Training')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'eval', 'play'],
                        help='Mode: train, eval, or play')
    parser.add_argument('--episodes', type=int, default=2000,
                        help='Number of training episodes')
    parser.add_argument('--model', type=str, default='models/2048_dqn_final.pth',
                        help='Model path for eval/play')
    parser.add_argument('--save_dir', type=str, default='models_2048',
                        help='Directory to save models')
    parser.add_argument('--device', type=str, default=None,
                        choices=['cpu', 'cuda'],
                        help='Device to use (cpu or cuda)')
    parser.add_argument('--eval_games', type=int, default=20,
                        help='Number of games for evaluation')

    args = parser.parse_args()

    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print(f"Using device: {device}")

    if args.mode == 'train':
        # 训练模式
        agent = train_agent(
            episodes=args.episodes,
            save_dir=args.save_dir,
            device=device
        )

    elif args.mode == 'eval':
        # 评估模式
        evaluate_agent(
            model_path=args.model,
            num_games=args.eval_games,
            device=device
        )

    elif args.mode == 'play':
        # 游玩模式
        print("Play mode - Watch AI play")

        agent = DQNAgent(device=device)
        if os.path.exists(args.model):
            agent.load(args.model)
        else:
            print(f"Model not found: {args.model}")
            return

        # 使用游戏界面
        from game_2048 import Game2048
        game = Game2048(ai_agent=agent)

        print("\nControls:")
        print("  - Click 'AI Control' to start AI")
        print("  - Arrow keys: Manual control")
        print("  - R: Reset game")
        print("  - ESC: Quit")

        game.run()

if __name__ == "__main__":
    main()