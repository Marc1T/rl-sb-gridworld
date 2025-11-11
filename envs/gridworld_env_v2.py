import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 8}

    def __init__(self, size=5, render_mode=None):
        super(GridWorldEnv, self).__init__()
        self.size = size
        self.window_size = 500
        self.render_mode = render_mode

        # Définir les actions (haut, bas, gauche, droite)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=size - 1, shape=(2,), dtype=np.int32)

        # Initialisation
        self.agent_pos = np.array([0, 0])
        self.goal_pos = np.array([size - 1, size - 1])

        # Interface Pygame
        self.window = None
        self.clock = None
        self.cell_size = self.window_size // self.size

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = np.array([0, 0])
        obs = self.agent_pos.copy()
        info = {}
        return obs, info

    def step(self, action):
        # Déplacement
        if action == 0:  # Haut
            self.agent_pos[1] = max(self.agent_pos[1] - 1, 0)
        elif action == 1:  # Bas
            self.agent_pos[1] = min(self.agent_pos[1] + 1, self.size - 1)
        elif action == 2:  # Gauche
            self.agent_pos[0] = max(self.agent_pos[0] - 1, 0)
        elif action == 3:  # Droite
            self.agent_pos[0] = min(self.agent_pos[0] + 1, self.size - 1)

        done = np.array_equal(self.agent_pos, self.goal_pos)
        reward = 1 if done else -0.01

        if self.render_mode == "human":
            self.render()

        return self.agent_pos.copy(), reward, done, False, {}

    def render(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.set_caption("🏁 GridWorld - Stable Baselines3")
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()

        if self.render_mode == "human":
            self.window.fill((255, 255, 255))  # fond blanc

            # Dessiner la grille
            for x in range(self.size):
                for y in range(self.size):
                    rect = pygame.Rect(
                        x * self.cell_size,
                        y * self.cell_size,
                        self.cell_size,
                        self.cell_size,
                    )
                    pygame.draw.rect(self.window, (200, 200, 200), rect, 1)

            # Dessiner la cible
            gx, gy = self.goal_pos
            pygame.draw.rect(
                self.window,
                (50, 205, 50),  # vert
                pygame.Rect(gx * self.cell_size, gy * self.cell_size, self.cell_size, self.cell_size),
            )

            # Dessiner l’agent
            ax, ay = self.agent_pos
            pygame.draw.circle(
                self.window,
                (30, 144, 255),  # bleu
                (ax * self.cell_size + self.cell_size // 2, ay * self.cell_size + self.cell_size // 2),
                self.cell_size // 3,
            )

            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

        elif self.render_mode == "rgb_array":
            surface = pygame.Surface((self.window_size, self.window_size))
            surface.fill((255, 255, 255))
            # (Tu peux étendre ce mode plus tard pour générer des vidéos)
            return np.transpose(np.array(pygame.surfarray.pixels3d(surface)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
