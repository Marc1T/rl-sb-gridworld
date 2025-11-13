# evaluation/evaluate_model_v3.py
import sys, os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import numpy as np
from stable_baselines3 import PPO
from envs.gridworld_env_v3_multi import GridWorldMultiAgentEnv
from utils.config import config

def evaluate_model_v3(model_path: str, n_episodes: int = 5):
    """Évalue et visualise un modèle PPO sur l’environnement multi-agent"""
    env = GridWorldMultiAgentEnv(config=config["env_v3"])
    model = PPO.load(model_path, env=env)

    for ep in range(1, n_episodes + 1):
        obs, info = env.reset()
        done = False
        total_rewards = np.zeros(env.n_agents)
        print(f"\n🎮 Épisode {ep} — {env.n_agents} agents")

        while not done:
            actions, _ = model.predict(obs, deterministic=True)
            obs, rewards, done, _, _ = env.step(actions)
            total_rewards += rewards
            env.render()

        print(f"🏁 Fin de l’épisode {ep} — Récompenses : {total_rewards}")

    env.close()

if __name__ == "__main__":
    model_file = os.path.join(config["paths"]["best_model"], "ppo_gridworld_v3.zip")
    evaluate_model_v3(model_file, n_episodes=3)
