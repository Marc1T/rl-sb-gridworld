# envs/gridworld_env_v2.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random


class GridWorldEnv(gym.Env):
    """
    GridWorld v2 – environnement intelligent et adaptable
    Compatible Stable-Baselines3 (PPO, DQN, A2C…)
    """

    metadata = {"render_modes": ["human", "none"], "render_fps": 5}

    def __init__(self, grid_size=7, n_obstacles=4, render_mode="none"):
        super(GridWorldEnv, self).__init__()

        self.grid_size = grid_size
        self.n_obstacles = n_obstacles
        self.render_mode = render_mode

        # Définition des actions : haut, bas, gauche, droite
        self.action_space = spaces.Discrete(4)

        # Observation : position agent + but + obstacles (sous forme vectorielle)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(grid_size, grid_size, 3), dtype=np.float32
        )

        # Couleurs
        self.agent_color = (0, 0, 1)    # Bleu
        self.goal_color = (0, 1, 0)     # Vert
        self.obstacle_color = (1, 0, 0) # Rouge

        # Initialisation du rendu
        self.fig, self.ax = None, None
        self.agent_pos = None
        self.goal_pos = None
        self.obstacles = None

        self.reset(seed=None)

    # --------------------------
    # ENVIRONMENT CORE
    # --------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Position aléatoire de l'agent et du but
        self.agent_pos = self._random_position()
        self.goal_pos = self._random_position(exclude=[self.agent_pos])

        # Génération aléatoire des obstacles
        self.obstacles = []
        for _ in range(self.n_obstacles):
            pos = self._random_position(exclude=[self.agent_pos, self.goal_pos] + self.obstacles)
            self.obstacles.append(pos)

        obs = self._get_obs()
        info = {}

        return obs, info

    def step(self, action):
        # Calcul du prochain mouvement
        x, y = self.agent_pos
        if action == 0 and y > 0: y -= 1           # Haut
        elif action == 1 and y < self.grid_size-1: y += 1  # Bas
        elif action == 2 and x > 0: x -= 1         # Gauche
        elif action == 3 and x < self.grid_size-1: x += 1  # Droite

        new_pos = (x, y)
        reward = -0.05  # petite pénalité à chaque pas
        terminated = False

        if new_pos in self.obstacles:
            reward = -1.0
        else:
            self.agent_pos = new_pos
            if self.agent_pos == self.goal_pos:
                reward = 1.0
                terminated = True

        obs = self._get_obs()
        info = {}

        return obs, reward, terminated, False, info

    # --------------------------
    # HELPER FUNCTIONS
    # --------------------------

    def _random_position(self, exclude=None):
        if exclude is None:
            exclude = []
        while True:
            pos = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
            if pos not in exclude:
                return pos

    def _get_obs(self):
        grid = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.float32)
        for (ox, oy) in self.obstacles:
            grid[oy, ox] = self.obstacle_color
        ax, ay = self.agent_pos
        gx, gy = self.goal_pos
        grid[ay, ax] = self.agent_color
        grid[gy, gx] = self.goal_color
        return grid

    # --------------------------
    # RENDERING
    # --------------------------

    def render(self):
        if self.render_mode == "none":
            return

        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(5, 5))
            self.ax.set_xticks(np.arange(-0.5, self.grid_size, 1))
            self.ax.set_yticks(np.arange(-0.5, self.grid_size, 1))
            self.ax.set_xticklabels([])
            self.ax.set_yticklabels([])
            self.ax.grid(True)

        self.ax.clear()
        self.ax.set_xlim(-0.5, self.grid_size - 0.5)
        self.ax.set_ylim(-0.5, self.grid_size - 0.5)
        self.ax.grid(True)

        # Dessiner obstacles
        for (x, y) in self.obstacles:
            self.ax.add_patch(patches.Rectangle((x - 0.5, y - 0.5), 1, 1, color="red"))

        # Dessiner but
        gx, gy = self.goal_pos
        self.ax.add_patch(patches.Rectangle((gx - 0.5, gy - 0.5), 1, 1, color="green"))

        # Dessiner agent
        ax, ay = self.agent_pos
        self.ax.add_patch(patches.Circle((ax, ay), 0.3, color="blue"))

        plt.draw()
        plt.pause(0.1)

    def close(self):
        if self.fig:
            plt.close(self.fig)
            self.fig = None
