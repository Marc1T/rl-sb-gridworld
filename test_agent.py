import os
import gymnasium as gym
from stable_baselines3 import PPO
from envs.gridworld_env import GridWorldEnv
from gymnasium.wrappers import RecordVideo

# 📁 Vérifie que le dossier vidéo existe
video_folder = "./videos/"
os.makedirs(video_folder, exist_ok=True)

# 🎮 Initialisation de l’environnement avec rendu
env = GridWorldEnv(grid_size=6, render_mode="rgb_array")
env = RecordVideo(env, video_folder=video_folder, name_prefix="gridworld_agent", episode_trigger=lambda ep: True)

# 📦 Chargement du modèle entraîné
model = PPO.load("ppo_gridworld")

# ▶️ Exécution de l’agent
obs, info = env.reset()
for step in range(200):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        obs, info = env.reset()

# 🧹 Nettoyage
env.close()
print("🎬 Vidéo enregistrée dans le dossier ./videos/")