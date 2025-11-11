# evaluation/record_videos.py
import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from envs.gridworld_env_v1 import GridWorldEnv
from utils.config import config

def record_model_video(model_path: str, video_folder: str = None, n_episodes: int = 5):
    """Enregistre des vidéos de plusieurs épisodes pour un modèle entraîné"""
    env_size = config["env"]["size"]
    video_folder = video_folder or config["paths"]["video_folder"]
    os.makedirs(video_folder, exist_ok=True)

    env = DummyVecEnv([lambda: GridWorldEnv(grid_size=env_size, render_mode="rgb_array")])
    env = VecVideoRecorder(
        env,
        video_folder=video_folder,
        record_video_trigger=lambda x: x < n_episodes,  # enregistre les n premiers épisodes
        video_length=200,
        name_prefix="ppo-gridworld"
    )

    model = PPO.load(model_path, env=env)
    
    obs = env.reset()
    for step in range(n_episodes * 200):  # 200 = longueur max par vidéo
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        if done[0]:
            obs = env.reset()

    env.close()
    print(f"✅ Vidéos sauvegardées dans {video_folder}")

if __name__ == "__main__":
    model_file = "../ppo_gridworld.zip"
    record_model_video(model_file, n_episodes=5)
