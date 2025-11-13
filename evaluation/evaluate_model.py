# evaluation/evaluate_model.py
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.gridworld_env_v1 import GridWorldEnv
from utils.config import config

def evaluate_model(model_path: str, n_episodes: int = 5):
    """Charge un modèle PPO et fait tourner plusieurs épisodes pour visualiser son comportement"""
    env_size = config["env"]["size"]
    env = DummyVecEnv([lambda: GridWorldEnv(grid_size=4, render_mode="human")])

    model = PPO.load(model_path, env=env)

    for episode in range(1, n_episodes+1):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]  # DummyVecEnv retourne des arrays
        print(f"Episode {episode}: Reward total = {total_reward}")

    env.close()

if __name__ == "__main__":
    model_file = "ppo_gridworld"  # chemin vers ton modèle sauvegardé
    evaluate_model(model_file, n_episodes=3)
