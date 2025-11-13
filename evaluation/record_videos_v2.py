# evaluation/record_videos_v2.py
import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration du chemin du projet ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from stable_baselines3 import PPO
from envs.gridworld_env_v2 import GridWorldEnv
from utils.config import config


def record_video(model_path: str, output_dir: str = "./videos_v2", n_episodes: int = 3):
    """
    Charge un modèle PPO et enregistre des vidéos des épisodes de test
    depuis les frames Matplotlib du rendu de GridWorldEnv_v2.
    """
    os.makedirs(output_dir, exist_ok=True)
    env = GridWorldEnv(grid_size=config["env"]["size"], render_mode="rgb_array")

    model = PPO.load(model_path)

    for episode in range(1, n_episodes + 1):
        obs, _ = env.reset()
        done = False
        frames = []
        total_reward = 0

        while not done:
            # Action prédite par le modèle
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            # --- Capture frame Matplotlib ---
            frame = env.render()  # doit renvoyer un array RGB
            if frame is not None:
                frames.append(frame)

        print(f"✅ Episode {episode} terminé — Reward total = {total_reward}")

        # --- Sauvegarde en vidéo MP4 ---
        if len(frames) > 0:
            video_path = os.path.join(output_dir, f"episode_{episode}.mp4")
            height, width, _ = frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video = cv2.VideoWriter(video_path, fourcc, 10, (width, height))

            for frame in frames:
                # Convertir RGB -> BGR (pour OpenCV)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video.write(frame_bgr)

            video.release()
            print(f"🎥 Vidéo sauvegardée : {video_path}")

    env.close()
    plt.close("all")
    print("🏁 Enregistrement terminé.")


if __name__ == "__main__":
    model_file = "ppo_gridworld_v2.zip"  # Nom de ton modèle sauvegardé
    record_video(model_file, output_dir="./videos_v2", n_episodes=3)
