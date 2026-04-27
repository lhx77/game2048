import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import math
import os
import time
from collections import deque


# --- 数据预处理 ---
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
            if a == 0:
                a = 3  # Up -> Right
            elif a == 3:
                a = 1  # Right -> Down
            elif a == 1:
                a = 2  # Down -> Left
            elif a == 2:
                a = 0  # Left -> Up
        return a

    def flip_action_h(a):
        if a == 2: return 3  # Left -> Right
        if a == 3: return 2  # Right -> Left
        return a

    # 4 rotations
    for k in range(4):
        g = np.rot90(curr_grid, k, axes=(1, 0))  # Clockwise rotation
        ng = np.rot90(curr_next, k, axes=(1, 0))
        a = rotate_action(action, k)

        symmetries.append((get_one_hot(g), a, reward, get_one_hot(ng), done))

        # Horizontal flip of the rotated grid
        g_flip = np.fliplr(g)
        ng_flip = np.fliplr(ng)
        a_flip = flip_action_h(a)

        symmetries.append((get_one_hot(g_flip), a_flip, reward, get_one_hot(ng_flip), done))

    return symmetries


# --- 神经网络架构 ---
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
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values


# --- 经验回放 ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None

        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)


# --- DQN 智能体 ---
class DQNAgent:
    def __init__(self, state_size=256, n_actions=4, device='cpu'):
        self.device = device
        self.state_size = state_size
        self.n_actions = n_actions

        # 优化参数
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.epsilon_decay = 0.999
        self.batch_size = 256
        self.learning_rate = 2e-4
        self.target_update = 200
        self.memory_size = 200000
        self.update_every = 8
        self.training_steps = 0

        # 网络
        self.policy_net = DQNNetwork(state_size, n_actions, hidden_size=512).to(device)
        self.target_net = DQNNetwork(state_size, n_actions, hidden_size=512).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.memory = ReplayBuffer(self.memory_size)

        # 统计
        self.episode_rewards = []
        self.episode_scores = []
        self.episode_max_tiles = []
        self.episode_lengths = []
        self.losses = []

        # 状态记录
        self.last_action = -1
        self.action_repetition_count = 0

    def select_action(self, state, legal_actions=None, eval_mode=False):
        if legal_actions is None:
            legal_actions = [0, 1, 2, 3]

        if not legal_actions:
            return 0

        if not eval_mode and random.random() < self.epsilon:
            return random.choice(legal_actions)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)[0]

            legal_q_values = q_values[legal_actions]
            max_q = torch.max(legal_q_values)
            best_actions = [legal_actions[i] for i, q in enumerate(legal_q_values) if q >= max_q - 1e-5]

            return random.choice(best_actions)

    def update(self):
        batch = self.memory.sample(self.batch_size)
        if batch is None:
            return 0.0

        states, actions, rewards, next_states, dones = batch

        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)

        current_q = self.policy_net(states_tensor).gather(1, actions_tensor.unsqueeze(1))

        with torch.no_grad():
            next_actions = self.policy_net(next_states_tensor).argmax(1, keepdim=True)
            next_q = self.target_net(next_states_tensor).gather(1, next_actions)
            target_q = rewards_tensor.unsqueeze(1) + (1 - dones_tensor.unsqueeze(1)) * self.gamma * next_q

        loss = nn.MSELoss()(current_q.squeeze(), target_q.squeeze())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.training_steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return loss.item()

    def adapt_training_params(self, recent_max_tiles):
        """根据表现自适应调整学习率等参数"""
        if not recent_max_tiles: return

        avg_max = sum(recent_max_tiles) / len(recent_max_tiles)

        # 如果平均表现较好，稍微降低学习率以稳定收敛
        if avg_max >= 1024:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = 5e-5
        elif avg_max >= 512:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = 8e-5

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())


# --- 训练游戏环境 ---
class TrainingGame:
    def __init__(self):
        self.grid_size = 4
        self.reset()

    def reset(self):
        self.grid = [[0] * self.grid_size for _ in range(self.grid_size)]
        self.score = 0
        self.game_over = False
        self.win = False
        self.steps = 0
        self.consecutive_no_merge = 0
        self.consecutive_invalid_moves = 0
        self.last_action = -1
        self.action_repetition_count = 0

        self.add_random_tile()
        self.add_random_tile()

        self.last_max_tile = self.get_max_tile()
        self.last_grid = [row[:] for row in self.grid]
        return self.get_observation()

    def _simulate_move(self, grid, direction):
        new_grid = [row[:] for row in grid]
        moved = False
        score = 0

        if direction == 0:  # Up
            for j in range(self.grid_size):
                col = [new_grid[i][j] for i in range(self.grid_size)]
                merged, s = self._merge_line(col)
                score += s
                for i in range(self.grid_size):
                    if new_grid[i][j] != merged[i]:
                        moved = True
                    new_grid[i][j] = merged[i]
        elif direction == 1:  # Down
            for j in range(self.grid_size):
                col = [new_grid[i][j] for i in range(self.grid_size)][::-1]
                merged, s = self._merge_line(col)
                score += s
                merged = merged[::-1]
                for i in range(self.grid_size):
                    if new_grid[i][j] != merged[i]:
                        moved = True
                    new_grid[i][j] = merged[i]
        elif direction == 2:  # Left
            for i in range(self.grid_size):
                line = new_grid[i]
                merged, s = self._merge_line(line)
                score += s
                if new_grid[i] != merged:
                    moved = True
                new_grid[i] = merged
        elif direction == 3:  # Right
            for i in range(self.grid_size):
                line = new_grid[i][::-1]
                merged, s = self._merge_line(line)
                score += s
                merged = merged[::-1]
                if new_grid[i] != merged:
                    moved = True
                new_grid[i] = merged

        return new_grid, moved, score

    def _merge_line(self, line):
        non_zero = [x for x in line if x != 0]
        merged = []
        score = 0
        i = 0
        while i < len(non_zero):
            if i < len(non_zero) - 1 and non_zero[i] == non_zero[i + 1]:
                merged.append(non_zero[i] * 2)
                score += non_zero[i] * 2
                i += 2
            else:
                merged.append(non_zero[i])
                i += 1
        return merged + [0] * (self.grid_size - len(merged)), score

    def check_move(self, direction):
        _, moved, _ = self._simulate_move(self.grid, direction)
        return moved

    def move(self, direction):
        new_grid, moved, score = self._simulate_move(self.grid, direction)
        self.grid = new_grid
        return moved, score

    def step(self, action):
        if self.game_over:
            return self.get_observation(), 0.0, True

        is_legal = self.check_move(action)
        if not is_legal:
            self.consecutive_invalid_moves += 1
            reward = -5.0 * (1.2 ** self.consecutive_invalid_moves)
            if self.consecutive_invalid_moves >= 5:
                self.game_over = True
            return self.get_observation(), reward, self.game_over

        self.consecutive_invalid_moves = 0
        self.steps += 1
        self.last_grid = [row[:] for row in self.grid]
        old_max = self.last_max_tile

        moved, score_gain = self.move(action)
        self.score += score_gain
        self.add_random_tile()

        new_max = self.get_max_tile()
        self.last_max_tile = new_max

        if not self.can_move():
            self.game_over = True

        reward = self.calculate_reward(old_max, new_max, score_gain, action)
        return self.get_observation(), reward, self.game_over

    def get_observation(self):
        return get_one_hot(self.grid)

    def get_max_tile(self):
        return max(max(row) for row in self.grid)

    def can_move(self):
        for i in range(4):
            if self.check_move(i): return True
        return False

    def add_random_tile(self):
        empty = [(i, j) for i in range(4) for j in range(4) if self.grid[i][j] == 0]
        if empty:
            i, j = random.choice(empty)
            self.grid[i][j] = 2 if random.random() < 0.9 else 4

    def calculate_reward(self, old_max, new_max, score_gain, action):
        reward = 0.0

        # 1. 合并奖励
        if score_gain > 0:
            reward += math.log2(score_gain) * 5.0
            self.consecutive_no_merge = 0
        else:
            self.consecutive_no_merge += 1
            if self.consecutive_no_merge > 3:
                reward -= 2.0

        # 2. 角落策略
        # 找实际最大值
        actual_max = 0
        for row in self.grid:
            for val in row:
                actual_max = max(actual_max, val)

        # 3. 空格惩罚
        empty_count = sum(1 for row in self.grid for x in row if x == 0)

        if empty_count < 4:
            penalty = -(8 - 2 ** empty_count)
            reward += penalty
        else:
            # 有足够空格，正常奖励
            reward += empty_count * 0.5

        # 4. 同一值多个惩罚 - 使用Sigmoid平滑函数
        duplicate_penalty = 0.0

        # 统计每个值的数量
        value_count = {}
        for i in range(4):
            for j in range(4):
                val = self.grid[i][j]
                if val > 0:
                    value_count[val] = value_count.get(val, 0) + 1

        # 计算惩罚
        for val, count in value_count.items():
            if count >= 2 and val > 0:
                x = math.log2(val)
                k = 1.0      # 陡峭度，值越大过渡越尖锐
                x0 = 4.0     # 中心点，值16开始有显著惩罚
                sigmoid = 1.0 / (1.0 + math.exp(-k * (x - x0)))
                value_factor = 0.1 + 0.9 * sigmoid
                if val <= 4:  # 2和4
                    value_factor *= 0
                elif val <= 8:  # 8
                    value_factor *= 0.2
                base_penalty = 0.4
                count_factor = (count - 1)
                penalty = -(base_penalty * value_factor * count_factor)
                duplicate_penalty += penalty

        reward += duplicate_penalty
        return reward

    def get_legal_actions(self):
        return [a for a in range(4) if self.check_move(a)]


# --- 训练入口 ---
def train_agent(episodes=3000, save_dir='models', device='cpu'):
    os.makedirs(save_dir, exist_ok=True)
    agent = DQNAgent(device=device)

    for episode in range(1, episodes + 1):
        game = TrainingGame()
        state = game.reset()

        while True:
            legal_actions = game.get_legal_actions()
            if not legal_actions: break

            action = agent.select_action(state, legal_actions=legal_actions)
            next_state, reward, done = game.step(action)

            s_list = get_symmetries(game.last_grid, action, reward, game.grid, done)
            for s, a, r, ns, d in s_list:
                agent.memory.push(s, a, r, ns, d)

            agent.training_steps += 1
            if agent.training_steps % agent.update_every == 0:
                agent.update()

            state = next_state
            if done: break

        if episode % 100 == 0:
            print(f"Episode {episode} completed. Max Tile: {game.get_max_tile()}")
            agent.save(f"{save_dir}/2048_ep{episode}.pth")


if __name__ == "__main__":
    train_agent()
