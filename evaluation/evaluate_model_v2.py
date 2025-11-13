# evaluation/evaluate_model_v2.py
import sys
import os
import time

# --- Configuration du chemin du projet ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import numpy as np
from stable_baselines3 import PPO
from envs.gridworld_env_v2 import GridWorldEnv
from utils.config import config


def evaluate_model_v2(model_path: str, n_episodes: int = 5, render: bool = True):
    """
    Évalue un modèle PPO sur GridWorldEnv_v2.
    Visualise (optionnellement) le comportement et affiche les scores moyens.
    """
    print(f"\n🚀 Évaluation du modèle : {model_path}")
    env = GridWorldEnv(grid_size=config["env"]["size"], render_mode="human" if render else None)
    model = PPO.load(model_path)

    rewards = []
    steps_per_episode = []

    for episode in range(1, n_episodes + 1):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        step_count = 0

        while not done:
            # Prédire l’action depuis l’observation
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step_count += 1

            if render:
                env.render()
                time.sleep(0.2)  # ralentir pour mieux visualiser

        rewards.append(total_reward)
        steps_per_episode.append(step_count)
        print(f"🎯 Épisode {episode}: Reward total = {total_reward:.2f}, étapes = {step_count}")

    env.close()

    # Statistiques globales
    print("\n📊 Résumé de l'évaluation :")
    print(f"→ Moyenne des récompenses : {np.mean(rewards):.2f}")
    print(f"→ Récompense min/max : {np.min(rewards):.2f} / {np.max(rewards):.2f}")
    print(f"→ Moyenne des étapes par épisode : {np.mean(steps_per_episode):.1f}")

    print("\n✅ Évaluation terminée.")


if __name__ == "__main__":
    model_file = "ppo_gridworld_v2.zip"  # fichier du modèle sauvegardé
    evaluate_model_v2(model_file, n_episodes=5, render=True)
