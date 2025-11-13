# evaluation/record_videos.py
import os
import sys
# Ajout du chemin racine du projet pour les imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from envs.gridworld_env_v1 import GridWorldEnv
from utils.config import config


def record_video(model_path: str, video_name: str = "ppo-gridworld-video", video_length: int = 200):
    """
    Enregistre une vidéo d'un agent PPO dans l'environnement GridWorld.
    """
    # 1️⃣ Créer l'environnement
    env_size = config["env"]["size"]
    env = DummyVecEnv([lambda: GridWorldEnv(grid_size=env_size, render_mode="rgb_array")])

    # 2️⃣ Charger le modèle
    model = PPO.load(model_path, env=env)

    # 3️⃣ Définir le dossier de sortie vidéo
    video_folder = config["paths"]["video_folder"]
    os.makedirs(video_folder, exist_ok=True)

    # 4️⃣ Envelopper avec VecVideoRecorder
    env = VecVideoRecorder(
        env,
        video_folder=video_folder,
        record_video_trigger=lambda step: step == 0,  # enregistrer le premier épisode
        video_length=video_length,
        name_prefix=video_name
    )

    # 5️⃣ Lancer un épisode et enregistrer la vidéo
    obs = env.reset()
    for _ in range(video_length):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if done.any():
            obs = env.reset()

    env.close()
    print(f"🎬 Vidéo enregistrée dans : {video_folder}")

if __name__ == "__main__":
    model_file = "ppo_gridworld_v1"  # le nom de ton modèle sauvegardé
    record_video(model_file, video_name="ppo_gridworld_demo", video_length=200)
