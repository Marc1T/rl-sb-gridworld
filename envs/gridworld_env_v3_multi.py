# envs/gridworld_env_v3_multi.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Optional, Dict


class GridWorldMultiAgentEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 5}

    def __init__(self, config: Optional[Dict] = None):
        super().__init__()

        # ---- Configuration par défaut ----
        default_config = {
            "grid_size": 7,
            "n_agents": 2,
            "n_goals": 2,
            "n_obstacles": 4,
            "max_steps": 100,
            "reward_goal": 10.0,
            "reward_step": -1.0,
            "reward_collision": -5.0,
            "render_mode": None,
            "obstacle_mode": "random",  # "random" ou "fixed"
            "seed": None
        }

        self.config = {**default_config, **(config or {})}
        self.grid_size = self.config["grid_size"]
        self.n_agents = self.config["n_agents"]
        self.n_goals = self.config["n_goals"]
        self.n_obstacles = self.config["n_obstacles"]
        self.max_steps = self.config["max_steps"]
        self.render_mode = self.config["render_mode"]

        if self.config["seed"] is not None:
            np.random.seed(self.config["seed"])

        # ---- Espaces Gym ----
        self.action_space = spaces.MultiDiscrete([4] * self.n_agents)
        obs_size = (self.n_agents + self.n_goals) * 2
        self.observation_space = spaces.Box(
            low=0, high=self.grid_size - 1,
            shape=(obs_size,), dtype=np.float32
        )

        # ---- États internes ----
        self.agent_positions = None
        self.goal_positions = None
        self.obstacle_positions = None
        self.current_step = 0

        # Pour affichage
        self.fig, self.ax = None, None

    # ---------------------- #
    #        Reset
    # ---------------------- #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0

        # Placer agents, cibles, obstacles
        positions = np.random.choice(
            self.grid_size ** 2,
            self.n_agents + self.n_goals + self.n_obstacles,
            replace=False
        )
        all_coords = np.array(np.unravel_index(positions, (self.grid_size, self.grid_size))).T

        self.agent_positions = all_coords[:self.n_agents]
        self.goal_positions = all_coords[self.n_agents:self.n_agents + self.n_goals]
        self.obstacle_positions = all_coords[self.n_agents + self.n_goals:]

        obs = self._get_obs()
        info = {}
        return obs, info

    # ---------------------- #
    #        Step
    # ---------------------- #
    def step(self, actions: np.ndarray):
        self.current_step += 1
        rewards = np.zeros(self.n_agents)
        dones = np.zeros(self.n_agents, dtype=bool)

        for i in range(self.n_agents):
            action = actions[i]
            move = np.array([0, 0])
            if action == 0: move = [-1, 0]  # Haut
            elif action == 1: move = [1, 0]  # Bas
            elif action == 2: move = [0, -1]  # Gauche
            elif action == 3: move = [0, 1]  # Droite

            new_pos = self.agent_positions[i] + move

            # Vérification des limites
            if (0 <= new_pos[0] < self.grid_size) and (0 <= new_pos[1] < self.grid_size):
                # Collision obstacle ?
                if not any((new_pos == obs).all() for obs in self.obstacle_positions):
                    self.agent_positions[i] = new_pos

            # Récompenses
            if any((self.agent_positions[i] == g).all() for g in self.goal_positions):
                rewards[i] += self.config["reward_goal"]
                dones[i] = True
            else:
                rewards[i] += self.config["reward_step"]

        # Collisions entre agents
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                if np.array_equal(self.agent_positions[i], self.agent_positions[j]):
                    rewards[i] += self.config["reward_collision"]
                    rewards[j] += self.config["reward_collision"]

        # Calcul des sorties
        terminated = bool(np.all(dones) or self.current_step >= self.max_steps)
        truncated = False
        reward = float(np.mean(rewards))
        obs = self._get_obs()
        info = {"rewards_per_agent": rewards.tolist()}

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    # ---------------------- #
    #        Observe
    # ---------------------- #
    def _get_obs(self):
        obs = np.concatenate([self.agent_positions.flatten(), self.goal_positions.flatten()])
        return obs.astype(np.float32)

    # ---------------------- #
    #        Render
    # ---------------------- #
    def render(self):
        if self.fig is None:
            self.fig, self.ax = plt.subplots()
            plt.ion()
        self.ax.clear()

        # Grille
        self.ax.set_xticks(np.arange(self.grid_size + 1))
        self.ax.set_yticks(np.arange(self.grid_size + 1))
        self.ax.grid(True)

        # Obstacles
        for obs in self.obstacle_positions:
            self.ax.add_patch(patches.Rectangle((obs[1], obs[0]), 1, 1, color="black"))

        # Cibles
        for goal in self.goal_positions:
            self.ax.add_patch(patches.Circle((goal[1] + 0.5, goal[0] + 0.5), 0.3, color="green"))

        # Agents
        colors = ["blue", "red", "orange", "purple"]
        for i, pos in enumerate(self.agent_positions):
            self.ax.add_patch(patches.Circle((pos[1] + 0.5, pos[0] + 0.5), 0.3, color=colors[i % len(colors)]))

        self.ax.set_xlim(0, self.grid_size)
        self.ax.set_ylim(0, self.grid_size)
        self.ax.invert_yaxis()
        self.ax.set_title(f"Step {self.current_step}")
        plt.pause(0.1)

    def close(self):
        if self.fig:
            plt.close(self.fig)
