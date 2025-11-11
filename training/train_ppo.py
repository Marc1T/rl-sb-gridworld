import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.callbacks import EvalCallback
from utils.config import config

from envs.gridworld_env_v1 import GridWorldEnv

# 1️⃣ Créer et envelopper l’environnement
env_size = config["env"]["size"]
env = GridWorldEnv(grid_size=env_size, render_mode="rgb_array")
env = Monitor(env)
env = DummyVecEnv([lambda: env])

# 2️⃣ Ajouter un enregistreur vidéo
video_folder = config["paths"]["video_folder"]

env = VecVideoRecorder(
    env,
    video_folder=config["paths"]["video_folder"],
    record_video_trigger=lambda x: x % 2000 == 0,
    video_length=200,
    name_prefix="ppo-gridworld"
)

# 3️⃣ Créer le modèle PPO
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=config["paths"]["tensorboard_log"],
    **config["ppo"]
)


# 4️⃣ Définir un callback d’évaluation
eval_env = DummyVecEnv([lambda: GridWorldEnv(grid_size=env_size)])
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=config["paths"]["best_model"],
    log_path=config["paths"]["logs"],
    eval_freq=config["training"]["eval_freq"],
    deterministic=True,
    render=False
)

# 5️⃣ Entraîner le modèle et Sauvegarder le modèle
model.learn(total_timesteps=config["ppo"]["total_timesteps"], callback=eval_callback)
model.save("ppo_gridworld")

# Sauvegarder les métriques
from utils.logger import save_metrics
metrics = {"episode_rewards": eval_callback.evaluations_results}
save_metrics(metrics, folder=config["paths"]["logs"])

env.close()
print("✅ Entraînement terminé et modèle sauvegardé !")