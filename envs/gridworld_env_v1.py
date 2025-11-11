import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

class GridWorldEnv(gym.Env):
    """
    Environnement GridWorld simple et extensible :
    - L’agent doit atteindre la case but (G)
    - Possibilité d’ajouter des obstacles
    - Rendu graphique avec pygame
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 5}

    def __init__(self, grid_size=5, render_mode=None):
        super().__init__()
        self.grid_size = grid_size
        self.render_mode = render_mode
        self.window_size = 500  # taille par défaut pour le rendu
        
        # Espace d’observation : position (x, y)
        self.observation_space = spaces.Box(low=0, high=grid_size-1, shape=(2,), dtype=np.int32)

        # Espace d’action : 4 directions
        self.action_space = spaces.Discrete(4)  # 0: Haut, 1: Bas, 2: Gauche, 3: Droite

        # Initialisation
        self.agent_pos = np.array([0, 0])
        self.goal_pos = np.array([grid_size - 1, grid_size - 1])
        self.obstacles = []  # tu pourras en ajouter ici

        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = np.array([0, 0])
        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        # Déplacement de l’agent
        if action == 0 and self.agent_pos[1] > 0:
            self.agent_pos[1] -= 1
        elif action == 1 and self.agent_pos[1] < self.grid_size - 1:
            self.agent_pos[1] += 1
        elif action == 2 and self.agent_pos[0] > 0:
            self.agent_pos[0] -= 1
        elif action == 3 and self.agent_pos[0] < self.grid_size - 1:
            self.agent_pos[0] += 1

        done = np.array_equal(self.agent_pos, self.goal_pos)
        reward = 1 if done else -0.01  # petite pénalité pour chaque mouvement
        truncated = False
        info = {}

        obs = self._get_obs()

        if self.render_mode == "human":
            self.render()

        return obs, reward, done, truncated, info

    def _get_obs(self):
        return np.copy(self.agent_pos)

    def render(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            self.window_size = 500
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()

        cell_size = self.window_size // self.grid_size
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        # But (G)
        pygame.draw.rect(canvas, (0, 255, 0),
                         pygame.Rect(self.goal_pos[0]*cell_size, self.goal_pos[1]*cell_size, cell_size, cell_size))

        # Agent (A)
        pygame.draw.rect(canvas, (0, 0, 255),
                         pygame.Rect(self.agent_pos[0]*cell_size, self.agent_pos[1]*cell_size, cell_size, cell_size))

        if self.render_mode == "human":
            self.window.blit(canvas, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

        return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
