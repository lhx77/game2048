# training.py
import pygame
import time
import argparse
import matplotlib.pyplot as plt
from game2048 import Game2048
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os

class ReplayBuffer:
    """Experience replay buffer"""
    def __init__(self, capacity=10000):
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
    """Deep Q-Network for 2048"""
    def __init__(self, input_size=16, n_actions=4, hidden_size=128):
        super(DQNNetwork, self).__init__()

        # 输入应该是展平的4x4网格 = 16
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, n_actions)
        )

    def forward(self, x):
        # 确保输入被展平
        if x.dim() > 2:
            batch_size = x.size(0)
            x = x.view(batch_size, -1)
        return self.network(x)

class DQNAgent:
    """Deep Q-Learning Agent for 2048"""
    def __init__(self, state_size=16, n_actions=4, device='cpu'):
        self.device = device
        self.state_size = state_size
        self.n_actions = n_actions

        # DQN parameters
        self.gamma = 0.99
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.learning_rate = 1e-3
        self.target_update = 10
        self.memory_size = 10000

        # Networks
        self.policy_net = DQNNetwork(state_size, n_actions).to(device)
        self.target_net = DQNNetwork(state_size, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

        # Memory
        self.memory = ReplayBuffer(self.memory_size)

        # Training state
        self.epsilon = self.epsilon_start
        self.training_steps = 0
        self.update_count = 0

        # Statistics
        self.episode_rewards = []
        self.episode_scores = []
        self.episode_max_tiles = []
        self.episode_lengths = []
        self.losses = []

        # Name for display
        self.name = "DQNAgent"

    def select_action(self, state, eval_mode=False):
        """Select action using epsilon-greedy policy"""
        if not eval_mode and random.random() < self.epsilon:
            return random.randrange(self.n_actions)

        with torch.no_grad():
            # 展平状态：从(4,4)到(16,)
            state_flattened = state.flatten()
            state_tensor = torch.FloatTensor(state_flattened).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def compute_td_loss(self, batch):
        """Compute temporal difference loss"""
        states, actions, rewards, next_states, dones = batch

        # 展平状态：从(batch, 4, 4)到(batch, 16)
        batch_size = states.shape[0]
        states_flat = states.reshape(batch_size, -1)
        next_states_flat = next_states.reshape(batch_size, -1)

        # Convert to tensors
        states_tensor = torch.FloatTensor(states_flat).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states_flat).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)

        # Get current Q values
        current_q = self.policy_net(states_tensor).gather(1, actions_tensor.unsqueeze(1))

        # Get next Q values from target network
        with torch.no_grad():
            next_q = self.target_net(next_states_tensor).max(1)[0]
            target_q = rewards_tensor + (1 - dones_tensor) * self.gamma * next_q

        # Compute loss
        loss = nn.MSELoss()(current_q.squeeze(), target_q)

        return loss

    def update(self):
        """Update the network"""
        # Sample from memory
        batch = self.memory.sample(self.batch_size)
        if batch is None:
            return 0.0

        # Compute loss
        loss = self.compute_td_loss(batch)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def save(self, path):
        """Save the model"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_steps': self.training_steps,
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
        """Load the model"""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.training_steps = checkpoint['training_steps']

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

    def get_status(self):
        """Get agent status"""
        return {
            'epsilon': self.epsilon,
            'memory_size': len(self.memory),
            'training_steps': self.training_steps,
            'episodes': len(self.episode_rewards)
        }

    def agent_name(self):
        """Get agent name for display"""
        return self.name

    def set_name(self, name):
        """Set agent name"""
        self.name = name

# Simple AI agents for testing
class SimpleAIAgent:
    """Simple rule-based AI agent"""
    def __init__(self):
        self.memory = []
        self.name = "SimpleAIAgent"

    def select_action(self, state, eval_mode=False):
        """Simple strategy: prefer right and down moves"""
        # This is a very basic strategy
        return random.choice([1, 3])  # Prefer down and right

    def update(self):
        """No update for simple agent"""
        return 0.0

    def agent_name(self):
        return self.name

class RandomAIAgent:
    """Completely random AI agent (baseline)"""
    def __init__(self):
        self.memory = []
        self.name = "RandomAIAgent"

    def select_action(self, state, eval_mode=False):
        """Random action"""
        return random.randrange(4)  # 0-3: up, down, left, right

    def update(self):
        """No update for random agent"""
        return 0.0

    def agent_name(self):
        return self.name

class TrainingGame:
    def __init__(self):
        self.grid_size = 4
        self.grid = [[0] * 4 for _ in range(4)]
        self.score = 0
        self.game_over = False
        self.win = False
        self.steps = 0
        self.add_random_tile()
        self.add_random_tile()

    def add_random_tile(self):
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

    def get_observation(self):
        obs = np.array(self.grid, dtype=np.float32)
        obs = np.where(obs == 0, 1, obs)
        obs = np.log2(obs)
        obs = obs / 11.0
        return obs

    def compress(self, line):
        return [x for x in line if x != 0]

    def merge(self, line):
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
        moved = False
        total_score = 0
        new_grid = [[0] * self.grid_size for _ in range(self.grid_size)]

        if direction == 0:  # Up
            for col in range(self.grid_size):
                column = [self.grid[row][col] for row in range(self.grid_size)]
                compressed = self.compress(column)
                merged, score = self.merge(compressed)
                total_score += score
                for row in range(len(merged)):
                    new_grid[row][col] = merged[row]
                    if column[row] != merged[row]:
                        moved = True

        elif direction == 1:  # Down
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

        elif direction == 2:  # Left
            for row in range(self.grid_size):
                compressed = self.compress(self.grid[row])
                merged, score = self.merge(compressed)
                total_score += score
                new_grid[row] = merged + [0] * (self.grid_size - len(merged))
                if self.grid[row] != new_grid[row]:
                    moved = True

        elif direction == 3:  # Right
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

    def get_max_tile(self):
        return max(max(row) for row in self.grid)

    def count_empty_cells(self):
        return sum(1 for row in self.grid for val in row if val == 0)

    def calculate_reward(self, score_gain, moved, max_tile, game_over, win):
        reward = 0.0

        # 1. 主要目标：当前最大方块的对数值（主要奖励）
        if max_tile >= 2:  # 至少有方块2
            # 使用log2，更大的方块获得指数级更高奖励
            # 例如：2->1, 4->2, 8->3, 16->4, 32->5, 64->6, 128->7, 256->8, 512->9, 1024->10, 2048->11
            reward += np.log2(max_tile) * 4.0

        # 2. 方块合并奖励（间接有利于增加最大方块）
        # 每次合并增加奖励，但比最大方块权重低
        reward += score_gain * 0.001  # 非常低的分数权重，只作为合并信号

        # 3. 无效移动惩罚（轻微）
        if not moved:
            reward -= 0.2

        # 4. 游戏结束惩罚
        if game_over:
            reward -= 30.0

        # 5. 达到2048的胜利奖励
        if win:
            reward += 20.0

        return reward

    def step(self, action):
        if self.game_over:
            return None, 0.0, True

        self.steps += 1
        old_max_tile = self.get_max_tile()

        moved, score_gain = self.move(action)
        self.score += score_gain

        if moved:
            self.add_random_tile()

        max_tile = self.get_max_tile()
        if max_tile >= 2048 and not self.win:
            self.win = True

        if not self.can_move():
            self.game_over = True

        reward = self.calculate_reward(
            score_gain, moved, max_tile,
            self.game_over, self.win
        )

        return self.get_observation(), reward, self.game_over

    def reset(self):
        self.grid = [[0] * self.grid_size for _ in range(self.grid_size)]
        self.score = 0
        self.game_over = False
        self.win = False
        self.steps = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.get_observation()


def train_agent(episodes=1000, render=False, save_dir='models', device='cpu'):
    """Train the DQN agent"""
    os.makedirs(save_dir, exist_ok=True)

    print(f"Starting training for {episodes} episodes...")
    print(f"Using device: {device}")
    print(f"Save directory: {save_dir}")

    # Create agent
    agent = DQNAgent(device=device)

    # Training statistics
    episode_start_time = time.time()

    for episode in range(1, episodes + 1):
        # Create game instance
        if render and episode % 100 == 0:
            # Render every 100 episodes
            game = Game2048(agent)
        else:
            # Create a lightweight game instance for training
            game = TrainingGame()

        # Reset game
        state = game.reset() if hasattr(game, 'reset') else game.get_observation()

        episode_reward = 0
        episode_score = 0
        episode_max_tile = 0
        episode_length = 0

        while True:
            # Select and execute action
            action = agent.select_action(state, eval_mode=False)

            if hasattr(game, 'step') and callable(game.step):
                # Using lightweight training game
                next_state, reward, done = game.step(action)
                if next_state is None:
                    break

                episode_score = game.score
                episode_max_tile = game.get_max_tile()
            else:
                # Using full Game2048 instance
                moved = game.step(action)
                if not moved:
                    reward = -1.0
                else:
                    # Calculate reward
                    max_tile = game.get_max_tile()
                    empty_cells = game.count_empty_cells()
                    reward = game.score * 0.01 + empty_cells * 0.1

                next_state = game.get_observation()
                done = game.game_over
                episode_score = game.score
                episode_max_tile = game.get_max_tile()

            # Store transition
            agent.memory.push(state, action, reward, next_state, done)

            # Update network
            loss = agent.update()
            if loss > 0:
                agent.losses.append(loss)

            # Update statistics
            state = next_state
            episode_reward += reward
            episode_length += 1

            if done:
                break

        # Record episode statistics
        agent.episode_rewards.append(episode_reward)
        agent.episode_scores.append(episode_score)
        agent.episode_max_tiles.append(episode_max_tile)
        agent.episode_lengths.append(episode_length)

        # Save model periodically
        if episode % 100 == 0:
            agent.save(f"{save_dir}/2048_dqn_episode_{episode}.pth")

        # Print progress
        if episode % 10 == 0:
            avg_reward = np.mean(agent.episode_rewards[-10:]) if len(agent.episode_rewards) >= 10 else episode_reward
            avg_score = np.mean(agent.episode_scores[-10:]) if len(agent.episode_scores) >= 10 else episode_score
            avg_max_tile = np.mean(agent.episode_max_tiles[-10:]) if len(agent.episode_max_tiles) >= 10 else episode_max_tile

            print(f"Episode {episode:4d}/{episodes} | "
                  f"Reward: {episode_reward:7.2f} | "
                  f"Score: {episode_score:6.0f} | "
                  f"Max: {episode_max_tile:4d} | "
                  f"Steps: {episode_length:3d} | "
                  f"Avg R: {avg_reward:6.2f} | "
                  f"Avg S: {avg_score:6.0f} | "
                  f"Avg M: {avg_max_tile:5.1f} | "
                  f"Eps: {agent.epsilon:.3f} | "
                  f"Mem: {len(agent.memory):5d}")

    # Save final model
    agent.save(f"{save_dir}/2048_dqn_final.pth")

    # Plot training statistics
    plot_training_stats(agent, save_dir)

    print(f"\nTraining completed in {time.time() - episode_start_time:.1f} seconds")

    return agent

def plot_training_stats(agent, save_dir):
    """Plot training statistics"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Episode rewards
    axes[0, 0].plot(agent.episode_rewards, alpha=0.6)
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')

    # Moving average of rewards
    window = 50
    if len(agent.episode_rewards) >= window:
        moving_avg = np.convolve(agent.episode_rewards, np.ones(window)/window, mode='valid')
        axes[0, 0].plot(range(window-1, len(agent.episode_rewards)), moving_avg, 'r-', linewidth=2)

    # Episode scores
    axes[0, 1].plot(agent.episode_scores, alpha=0.6)
    axes[0, 1].set_title('Episode Scores')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Score')

    # Max tiles (log scale)
    axes[0, 2].plot(agent.episode_max_tiles, alpha=0.6)
    axes[0, 2].set_title('Max Tile per Episode')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Max Tile')
    axes[0, 2].set_yscale('log', base=2)

    # Episode lengths
    axes[1, 0].plot(agent.episode_lengths, alpha=0.6)
    axes[1, 0].set_title('Episode Lengths')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Steps')

    # Losses
    if agent.losses:
        axes[1, 1].plot(agent.losses, alpha=0.6)
        axes[1, 1].set_title('Training Loss')
        axes[1, 1].set_xlabel('Update')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_yscale('log')

    # Max tile distribution
    if agent.episode_max_tiles:
        max_tiles = np.array(agent.episode_max_tiles)
        unique_tiles = np.unique(max_tiles)
        tile_counts = [np.sum(max_tiles == tile) for tile in unique_tiles]

        axes[1, 2].bar(unique_tiles, tile_counts, alpha=0.6)
        axes[1, 2].set_title('Max Tile Distribution')
        axes[1, 2].set_xlabel('Max Tile')
        axes[1, 2].set_ylabel('Count')
        axes[1, 2].set_xticks(unique_tiles)
        axes[1, 2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_stats.png", dpi=300, bbox_inches='tight')
    plt.show()

def test_agent(model_path, episodes=10, render=True):
    """Test a trained agent"""
    print(f"\nTesting agent from {model_path}")

    # Create agent
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = DQNAgent(device=device)

    # Load model
    if not agent.load(model_path):
        print("Failed to load model, exiting...")
        return

    # Test statistics
    test_scores = []
    test_max_tiles = []
    test_steps = []

    for episode in range(episodes):
        # Create game
        game = Game2048(agent) if render else None

        if render:
            # Reset game
            game.reset_game()
            state = game.get_observation()

            episode_score = 0
            episode_max_tile = 0
            episode_steps = 0

            while not game.game_over:
                # Select action (eval mode, no exploration)
                action = agent.select_action(state, eval_mode=True)

                # Execute action
                game.step(action)

                # Update state
                state = game.get_observation()
                episode_steps += 1
                episode_score = game.score
                episode_max_tile = game.get_max_tile()

                # Draw
                game.draw()

            test_scores.append(episode_score)
            test_max_tiles.append(episode_max_tile)
            test_steps.append(episode_steps)

            print(f"Test {episode+1:2d}/{episodes}: "
                  f"Score: {episode_score:6.0f}, "
                  f"Max: {episode_max_tile:4d}, "
                  f"Steps: {episode_steps:3d}")

            if render:
                pygame.time.wait(1000)  # Wait 1 second between games

        else:
            # Quick test without rendering
            class QuickGame:
                def __init__(self):
                    self.grid_size = 4
                    self.reset()

                def reset(self):
                    self.grid = [[0] * 4 for _ in range(4)]
                    self.score = 0
                    self.game_over = False
                    self.add_random_tile()
                    self.add_random_tile()

                def add_random_tile(self):
                    empty_cells = []
                    for i in range(self.grid_size):
                        for j in range(self.grid_size):
                            if self.grid[i][j] == 0:
                                empty_cells.append((i, j))

                    if empty_cells:
                        i, j = random.choice(empty_cells)
                        self.grid[i][j] = 2 if random.random() < 0.9 else 4

                def get_observation(self):
                    obs = np.array(self.grid, dtype=np.float32)
                    obs = np.where(obs == 0, 1, obs)
                    obs = np.log2(obs)
                    obs = obs / 11.0
                    return obs

                def step(self, action):
                    # Simplified step for testing
                    # This is a placeholder - in reality you'd need the full game logic
                    self.game_over = random.random() < 0.1  # 10% chance to end
                    self.score += random.randint(0, 100)
                    return not self.game_over

            game = QuickGame()
            game.reset()
            state = game.get_observation()

            episode_score = 0
            episode_steps = 0

            while not game.game_over:
                action = agent.select_action(state, eval_mode=True)
                moved = game.step(action)
                state = game.get_observation()
                episode_steps += 1
                episode_score = game.score

            test_scores.append(episode_score)
            test_steps.append(episode_steps)

    # Print test results
    if test_scores:
        print(f"\nTest Results ({episodes} episodes):")
        print(f"Average Score: {np.mean(test_scores):.1f} ± {np.std(test_scores):.1f}")
        print(f"Best Score: {max(test_scores)}")
        if test_max_tiles:
            print(f"Average Max Tile: {np.mean(test_max_tiles):.1f}")
            print(f"Best Max Tile: {max(test_max_tiles)}")
        print(f"Average Steps: {np.mean(test_steps):.1f}")

    if render and game is not None:
        pygame.quit()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train and test 2048 DQN agent')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test', 'demo'],
                        help='Mode: train, test, or demo')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of episodes to train/test')
    parser.add_argument('--render', action='store_true',
                        help='Render the game during training/testing')
    parser.add_argument('--model', type=str, default='models/2048_dqn_final.pth',
                        help='Model path for testing/demo')
    parser.add_argument('--save_dir', type=str, default='models',
                        help='Directory to save models')
    parser.add_argument('--device', type=str, default=None,
                        choices=['cpu', 'cuda'],
                        help='Device to use (cpu or cuda)')

    args = parser.parse_args()

    # Set device
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print(f"Using device: {device}")

    if args.mode == 'train':
        # Training mode
        agent = train_agent(
            episodes=args.episodes,
            render=args.render,
            save_dir=args.save_dir,
            device=device
        )

    elif args.mode == 'test':
        # Testing mode
        test_agent(
            model_path=args.model,
            episodes=args.episodes,
            render=args.render
        )

    elif args.mode == 'demo':
        # Demo mode - interactive testing
        print("Demo mode - Manual testing with AI agent")

        # Create and load agent
        agent = DQNAgent(device=device)
        if os.path.exists(args.model):
            agent.load(args.model)
        else:
            print(f"Model not found: {args.model}")
            return

        # Create game with the agent
        game = Game2048(ai_agent=agent)

        # Set initial mode
        game.auto_play_mode = True
        game.buttons['auto_play']['active'] = True
        game.buttons['auto_play']['color'] = game.AUTO_PLAY_COLOR

        print("\nDemo started!")
        print("Controls:")
        print("  - Arrow keys: Manual control")
        print("  - R: Reset game")
        print("  - ESC: Quit")
        print("  - Click buttons to change modes")
        print("\nClick 'Auto Play' button to see AI in action!")

        # Run the game
        game.run()

if __name__ == "__main__":
    main()