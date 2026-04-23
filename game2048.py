# game_2048.py
import pygame
import random
import numpy as np
import sys
import threading
import time
import math
from typing import Dict, Optional

class Game2048:
    """2048 Game with AI Control"""
    def __init__(self, ai_agent=None):
        pygame.init()

        # Game parameters
        self.grid_size = 4
        self.cell_size = 100
        self.grid_padding = 10
        self.window_width = 500
        self.window_height = 650
        self.window = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("2048")

        # Fonts
        self.clock = pygame.time.Clock()
        self.title_font = pygame.font.Font(None, 48)
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        self.button_font = pygame.font.Font(None, 28)

        # Colors
        self.BACKGROUND = (187, 173, 160)
        self.BUTTON_COLOR = (143, 122, 102)
        self.BUTTON_HOVER = (165, 142, 120)
        self.BUTTON_TEXT = (249, 246, 242)
        self.EMPTY = (205, 193, 180)
        self.TEXT_COLOR = (119, 110, 101)
        self.TEXT_COLOR_LIGHT = (249, 246, 242)
        self.GREEN = (76, 175, 80)
        self.ORANGE = (255, 152, 0)
        self.BLUE = (33, 150, 243)
        self.GRAY = (158, 158, 158)

        # Tile colors
        self.TILE_COLORS = {
            0: (205, 193, 180), 2: (238, 228, 218), 4: (237, 224, 200),
            8: (242, 177, 121), 16: (245, 149, 99), 32: (246, 124, 95),
            64: (246, 94, 59), 128: (237, 207, 114), 256: (237, 204, 97),
            512: (237, 200, 80), 1024: (237, 197, 63), 2048: (237, 194, 46)
        }

        # AI settings
        self.ai_agent = ai_agent

        # Mode settings
        self.ai_mode = False  # AI control mode
        self.training_mode = False  # Training mode
        self.training_thread = None
        self.training_stats = {}  # Training statistics

        # Game state
        self.ai_move_delay = 300
        self.last_ai_move_time = 0

        # Button layout
        button_y_start = 55
        button_width = 120
        button_height = 40

        self.buttons = {
            'ai_toggle': {
                'rect': pygame.Rect(20, button_y_start, button_width, button_height),
                'text': 'AI Control',
                'color': self.BUTTON_COLOR,
                'active_color': self.GREEN
            },
            'train': {
                'rect': pygame.Rect(185, button_y_start, button_width, button_height),
                'text': 'Start Train',
                'color': self.BUTTON_COLOR,
                'active_color': self.ORANGE
            },
            'reset': {
                'rect': pygame.Rect(350, button_y_start, button_width, button_height),
                'text': 'Reset Game',
                'color': self.BUTTON_COLOR
            }
        }

        self.reset_game()

    def reset_game(self):
        """Reset the current game"""
        self.grid = [[0] * self.grid_size for _ in range(self.grid_size)]
        self.score = 0
        self.game_over = False
        self.win = False
        self.steps = 0

        # Add two initial tiles
        self.add_random_tile()
        self.add_random_tile()

    def add_random_tile(self):
        """Add a random tile to empty cell"""
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
        """Get one-hot state for AI"""
        one_hot = np.zeros((4, 4, 16), dtype=np.float32)
        for i in range(4):
            for j in range(4):
                val = self.grid[i][j]
                if val == 0:
                    one_hot[i, j, 0] = 1
                else:
                    power = int(math.log2(val))
                    idx = min(power, 15)
                    one_hot[i, j, idx] = 1
        return one_hot.flatten()

    def compress(self, line):
        """Remove zeros from line"""
        return [x for x in line if x != 0]

    def merge(self, line):
        """Merge same numbers in line, return (new line, score)"""
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

    def check_move(self, direction):
        """Check if a move is possible in a specific direction"""
        if direction == 0:  # Up
            for col in range(self.grid_size):
                for row in range(self.grid_size - 1):
                    if self.grid[row][col] == 0 and self.grid[row+1][col] != 0:
                        return True
                    if self.grid[row][col] != 0 and self.grid[row][col] == self.grid[row+1][col]:
                        return True
        elif direction == 1:  # Down
            for col in range(self.grid_size):
                for row in range(self.grid_size - 1, 0, -1):
                    if self.grid[row][col] == 0 and self.grid[row-1][col] != 0:
                        return True
                    if self.grid[row][col] != 0 and self.grid[row][col] == self.grid[row-1][col]:
                        return True
        elif direction == 2:  # Left
            for row in range(self.grid_size):
                for col in range(self.grid_size - 1):
                    if self.grid[row][col] == 0 and self.grid[row][col+1] != 0:
                        return True
                    if self.grid[row][col] != 0 and self.grid[row][col] == self.grid[row][col+1]:
                        return True
        elif direction == 3:  # Right
            for row in range(self.grid_size):
                for col in range(self.grid_size - 1, 0, -1):
                    if self.grid[row][col] == 0 and self.grid[row][col-1] != 0:
                        return True
                    if self.grid[row][col] != 0 and self.grid[row][col] == self.grid[row][col-1]:
                        return True
        return False

    def get_legal_actions(self):
        """Get list of legal actions"""
        return [a for a in range(4) if self.check_move(a)]

    def move(self, direction):
        """Move grid, return (moved, score)"""
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
        """Check if any move is possible"""
        # Check for empty cells
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.grid[i][j] == 0:
                    return True

        # Check for adjacent same numbers
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                val = self.grid[i][j]
                if j < self.grid_size - 1 and self.grid[i][j+1] == val:
                    return True
                if i < self.grid_size - 1 and self.grid[i+1][j] == val:
                    return True

        return False

    def step(self, action):
        """Execute one action"""
        if self.game_over:
            return False

        self.steps += 1

        # Execute move
        moved, score_gain = self.move(action)

        # Update score
        self.score += score_gain

        # Add new tile if moved
        if moved:
            self.add_random_tile()

        # Check game state
        max_tile = self.get_max_tile()
        if max_tile >= 2048 and not self.win:
            self.win = True

        # Check if game is over
        if not self.can_move():
            self.game_over = True

        return moved

    def get_max_tile(self):
        """Get current max tile value"""
        return max(max(row) for row in self.grid)

    def count_empty_cells(self):
        """Count empty cells"""
        return sum(1 for row in self.grid for val in row if val == 0)

    def draw_button(self, button_info, mouse_pos):
        """Draw a button"""
        rect = button_info['rect']
        color = button_info['color']

        # Use active color for active modes
        if button_info.get('active', False):
            color = button_info.get('active_color', self.BUTTON_COLOR)

        # Hover effect
        if rect.collidepoint(mouse_pos):
            if button_info.get('active', False):
                # Darken active color on hover
                color = tuple(max(0, c - 20) for c in color)
            else:
                color = self.BUTTON_HOVER

        # Draw button
        pygame.draw.rect(self.window, color, rect, 0, 5)
        pygame.draw.rect(self.window, (0, 0, 0), rect, 2, 5)

        # Draw button text
        text = self.button_font.render(button_info['text'], True, self.BUTTON_TEXT)
        text_rect = text.get_rect(center=rect.center)
        self.window.blit(text, text_rect)

        return button_info

    def draw(self):
        """Draw the game interface"""
        # Draw background
        self.window.fill(self.BACKGROUND)

        # Draw title area
        title_bg = pygame.Rect(0, 0, self.window_width, 100)
        pygame.draw.rect(self.window, (173, 157, 143), title_bg)

        # Draw title
        title = self.title_font.render("2048", True, self.TEXT_COLOR)
        self.window.blit(title, (self.window_width//2 - title.get_width()//2, 20))

        # Draw score and status area
        status_bg = pygame.Rect(0, 100, self.window_width, 60)
        pygame.draw.rect(self.window, (187, 173, 160), status_bg)

        # Draw score
        score_text = self.font.render(f"Score: {self.score}", True, self.TEXT_COLOR)
        self.window.blit(score_text, (20, 110))

        # Draw steps
        steps_text = self.font.render(f"Steps: {self.steps}", True, self.TEXT_COLOR)
        self.window.blit(steps_text, (180, 110))

        # Draw max tile
        max_tile = self.get_max_tile()
        max_text = self.font.render(f"Max: {max_tile}", True, self.TEXT_COLOR)
        self.window.blit(max_text, (340, 110))

        # Draw mode indicator
        mode_y = 160
        if self.ai_mode:
            mode_color = self.GREEN
            mode_text = f"AI Control Mode"
            if hasattr(self.ai_agent, 'agent_name'):
                mode_text += f": {self.ai_agent.agent_name()}"
        elif self.training_mode:
            mode_color = self.ORANGE
            mode_text = "Training Mode"

            # Show training stats if available
            if self.training_stats:
                episodes = self.training_stats.get('episodes', 0)
                avg_max = self.training_stats.get('avg_max_tile', 0)
                last_max = self.training_stats.get('last_max_tile', 0)
                avg_score = self.training_stats.get('avg_score', 0)

                # 显示最大方块训练进度
                mode_text += f" (Ep:{episodes}, Max:{last_max}, AvgMax:{avg_max:.0f})"
        else:
            mode_color = self.GRAY
            mode_text = "Manual Control"

        mode_surface = self.small_font.render(mode_text, True, mode_color)
        self.window.blit(mode_surface, (self.window_width//2 - mode_surface.get_width()//2, mode_y))

        # Calculate game grid position
        grid_y_start = 190
        grid_total_size = self.grid_size * (self.cell_size + self.grid_padding) + self.grid_padding
        grid_x_start = (self.window_width - grid_total_size) // 2

        # Draw grid background
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x = grid_x_start + j * (self.cell_size + self.grid_padding)
                y = grid_y_start + i * (self.cell_size + self.grid_padding)
                pygame.draw.rect(self.window, self.EMPTY, (x, y, self.cell_size, self.cell_size), 0, 5)

        # Draw tiles
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                value = self.grid[i][j]
                if value > 0:
                    x = grid_x_start + j * (self.cell_size + self.grid_padding)
                    y = grid_y_start + i * (self.cell_size + self.grid_padding)

                    # Tile color
                    color = self.TILE_COLORS.get(value, (60, 58, 50))
                    pygame.draw.rect(self.window, color, (x, y, self.cell_size, self.cell_size), 0, 5)

                    # Tile number
                    text_color = self.TEXT_COLOR if value < 8 else self.TEXT_COLOR_LIGHT

                    # Select font size based on number
                    if value < 10:
                        font_size = 55
                    elif value < 100:
                        font_size = 50
                    elif value < 1000:
                        font_size = 45
                    else:
                        font_size = 40

                    font = pygame.font.Font(None, font_size)
                    text = font.render(str(value), True, text_color)
                    text_rect = text.get_rect(center=(x + self.cell_size//2, y + self.cell_size//2))
                    self.window.blit(text, text_rect)

        # Draw control panel background
        control_bg = pygame.Rect(0, grid_y_start + grid_total_size + 20, self.window_width, 100)
        pygame.draw.rect(self.window, (200, 190, 180), control_bg, 0, 10)

        # Draw control instructions
        control_text = self.small_font.render("Controls: Arrow Keys to Move | R to Reset | ESC to Quit", True, self.TEXT_COLOR)
        self.window.blit(control_text, (self.window_width//2 - control_text.get_width()//2, grid_y_start + grid_total_size + 40))

        # Draw buttons
        mouse_pos = pygame.mouse.get_pos()
        for button_id, button_info in self.buttons.items():
            updated_info = self.draw_button(button_info, mouse_pos)
            self.buttons[button_id] = updated_info

        # Game over overlay
        if self.game_over:
            overlay = pygame.Surface((grid_total_size, grid_total_size))
            overlay.set_alpha(200)
            overlay.fill((0, 0, 0))
            self.window.blit(overlay, (grid_x_start, grid_y_start))

            if self.win:
                text = self.font.render("You Win!", True, (255, 255, 255))
            else:
                text = self.font.render("Game Over", True, (255, 255, 255))

            text_rect = text.get_rect(center=(self.window_width//2, grid_y_start + grid_total_size//2))
            self.window.blit(text, text_rect)

            restart_text = self.small_font.render("Click Reset Game to Continue", True, (255, 255, 255))
            restart_rect = restart_text.get_rect(center=(self.window_width//2, grid_y_start + grid_total_size//2 + 40))
            self.window.blit(restart_text, restart_rect)

        pygame.display.flip()

    def handle_button_click(self, pos):
        """Handle button clicks"""
        for button_id, button_info in self.buttons.items():
            if button_info['rect'].collidepoint(pos):
                if button_id == 'ai_toggle':
                    self.toggle_ai_mode()
                elif button_id == 'train':
                    self.toggle_training_mode()
                elif button_id == 'reset':
                    self.reset_game()
                return True
        return False

    def toggle_ai_mode(self):
        """Toggle AI control mode"""
        if self.ai_agent is None:
            print("Warning: No AI agent set, AI mode unavailable")
            return

        self.ai_mode = not self.ai_mode
        self.buttons['ai_toggle']['active'] = self.ai_mode

        if self.ai_mode:
            self.buttons['ai_toggle']['text'] = 'AI On'
        else:
            self.buttons['ai_toggle']['text'] = 'AI Control'

    def toggle_training_mode(self):
        """Toggle training mode"""
        if self.ai_agent is None:
            print("Initializing AI agent for training...")
            from training import DQNAgent
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.ai_agent = DQNAgent(device=device)

        self.training_mode = not self.training_mode
        self.buttons['train']['active'] = self.training_mode

        if self.training_mode:
            self.buttons['train']['text'] = 'Training...'
            # Start training in a separate thread
            self.start_training()
        else:
            self.buttons['train']['text'] = 'Start Train'
            # Stop training
            self.stop_training()

    def start_training(self):
        """Start training in a separate thread"""
        if self.training_thread is not None and self.training_thread.is_alive():
            return

        def training_worker():
            episodes = 0
            max_tiles = []  # 记录最大方块历史
            avg_score = 0   # 初始化平均分数
            avg_max_tile = 0 # 初始化平均最大方块
            from training import TrainingGame, get_symmetries

            while self.training_mode:
                # Create a new game for training
                game = TrainingGame()
                state = game.reset()
                episode_max_tile = 0
                episode_score = 0

                while self.training_mode:
                    # Select action with masking
                    legal_actions = game.get_legal_actions()
                    if not legal_actions:
                        break
                        
                    action = self.ai_agent.select_action(state, legal_actions=legal_actions, eval_mode=False)

                    # Execute action
                    next_state, reward, done = game.step(action)

                    # 每 10 步更新一次 UI 统计数据，减少主线程压力
                    if game.steps % 10 == 0:
                        current_max = game.get_max_tile()
                        current_score = game.score
                        self.training_stats = {
                            'episodes': episodes + 1,
                            'avg_score': avg_score,
                            'avg_max_tile': avg_max_tile,
                            'last_max_tile': current_max,
                            'last_score': current_score
                        }
                    
                    # 记录本局最高
                    episode_max_tile = max(episode_max_tile, game.get_max_tile())
                    episode_score = game.score

                    # Store experience
                    if hasattr(self.ai_agent, 'memory'):
                        s_list = get_symmetries(game.last_grid, action, reward, game.grid, done)
                        for s, a, r, ns, d in s_list:
                            self.ai_agent.memory.push(s, a, r, ns, d)

                    # Update agent (针对 CPU 大幅降低频率)
                    self.ai_agent.training_steps += 1
                    update_every = 16 # 每 16 步更新一次
                    
                    if self.ai_agent.training_steps % update_every == 0:
                        if hasattr(self.ai_agent, 'update'):
                            try:
                                self.ai_agent.update()
                                # 更新完后强制睡眠，让出 CPU 给 UI 线程
                                time.sleep(0.05)
                            except Exception as e:
                                print(f"Training Update Error: {e}")
                                pass

                    # Update state
                    state = next_state

                    if done:
                        break
                    
                    # 释放 GIL 锁
                    time.sleep(0.001)
                
                if not self.training_mode:
                    break

                # Update training stats
                episodes += 1
                max_tiles.append(episode_max_tile)

                # Keep only last 100 max tiles for average
                if len(max_tiles) > 100:
                    max_tiles.pop(0)

                avg_max_tile = np.mean(max_tiles) if max_tiles else 0
                avg_score = episode_score  # 或者用历史平均分数

                # Update UI stats (显示最大方块统计)
                self.training_stats = {
                    'episodes': episodes,
                    'avg_score': avg_score,
                    'avg_max_tile': avg_max_tile,
                    'last_max_tile': episode_max_tile,
                    'last_score': episode_score
                }

                # 每 50 轮自动保存一次，增加路径检查
                if episodes > 0 and episodes % 50 == 0:
                    try:
                        import os
                        if not os.path.exists('models'):
                            os.makedirs('models')
                        self.ai_agent.save("models/2048_trained.pth")
                        print(f"Auto-saved model at episode {episodes}")
                    except Exception as e:
                        print(f"Save error: {e}")

                # 增加睡眠时间，确保 UI 线程有充足的刷新机会
                time.sleep(0.1)

        self.training_thread = threading.Thread(target=training_worker, daemon=True)
        self.training_thread.start()

    def stop_training(self):
        """Stop training"""
        self.training_mode = False
        if self.training_thread is not None:
            self.training_thread.join(timeout=1.0)
        self.training_thread = None

    def ai_move(self):
        """Make AI move (only in AI control mode)"""
        if not self.ai_mode or self.game_over or self.ai_agent is None:
            return False

        current_time = pygame.time.get_ticks()
        if current_time - self.last_ai_move_time < self.ai_move_delay:
            return False

        # Get current state
        state = self.get_observation()

        # AI selects action with masking
        legal_actions = self.get_legal_actions()
        if not legal_actions:
            return False
            
        action = self.ai_agent.select_action(state, legal_actions=legal_actions, eval_mode=True)

        # Execute action
        moved = self.step(action)

        self.last_ai_move_time = current_time
        return moved

    def set_ai_agent(self, agent):
        """Set AI agent"""
        self.ai_agent = agent

    def run(self):
        """Run the main game loop"""
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        self.reset_game()
                    elif not self.ai_mode and not self.game_over:
                        if event.key == pygame.K_UP:
                            self.step(0)
                        elif event.key == pygame.K_DOWN:
                            self.step(1)
                        elif event.key == pygame.K_LEFT:
                            self.step(2)
                        elif event.key == pygame.K_RIGHT:
                            self.step(3)

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        self.handle_button_click(event.pos)

            # AI control (only in AI mode, not in training mode)
            if self.ai_mode and not self.game_over:
                self.ai_move()

            # Draw
            self.draw()
            self.clock.tick(60)

        # Clean up
        self.stop_training()
        pygame.quit()
        sys.exit()

# Simple AI agent for testing
class SimpleAIAgent:
    def __init__(self):
        self.memory = []

    def select_action(self, state, eval_mode=False):
        # Simple strategy: try to merge tiles
        return random.choice([0, 1, 2, 3])  # Random action for now

    def agent_name(self):
        return 'SimpleAIAgent'

if __name__ == "__main__":
    # 创建一个AI智能体
    import torch

    # 选择设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 导入DQNAgent
    from training import DQNAgent

    # 创建DQN智能体
    ai_agent = DQNAgent(state_size=256, n_actions=4, device=device)

    # Create and run the game
    game = Game2048(ai_agent=ai_agent)
    try:
        game.run()
    finally:
        # 游戏退出时自动保存
        if hasattr(ai_agent, 'training_steps') and ai_agent.training_steps > 0:
            ai_agent.save("models/2048_trained.pth")