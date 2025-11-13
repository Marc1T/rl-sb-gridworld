import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed
from utils.config import config
from utils.logger import save_metrics
from envs.gridworld_env_v1 import GridWorldEnv

# 📁 Setup
os.makedirs(config["paths"]["video_folder"], exist_ok=True)
os.makedirs(config["paths"]["logs"], exist_ok=True)
os.makedirs(config["paths"]["best_model"], exist_ok=True)

# 🎮 Environnement
env_size = config["env"]["size"]
render_mode = config["env"]["render_mode"] or "rgb_array"
env = GridWorldEnv(grid_size=env_size, render_mode=render_mode)
env = Monitor(env)
env = DummyVecEnv([lambda: env])

# 🎥 Enregistreur vidéo
env = VecVideoRecorder(
    env,
    video_folder=config["paths"]["video_folder"],
    record_video_trigger=lambda x: x % config["training"]["eval_freq"] == 0,
    video_length=200,
    name_prefix="ppo-gridworld"
)

# 🧩 Hyperparamètres
ppo_kwargs = config["ppo"]
total_timesteps = config["training"]["total_timesteps"]
set_random_seed(config["training"]["seed"])

# 🚀 Modèle PPO
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=config["paths"]["tensorboard_log"],
    **ppo_kwargs
)

# 📊 Callback d’évaluation
eval_env = DummyVecEnv([lambda: GridWorldEnv(grid_size=env_size)])
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=config["paths"]["best_model"],
    log_path=config["paths"]["logs"],
    eval_freq=config["training"]["eval_freq"],
    n_eval_episodes=config["training"]["n_eval_episodes"],
    deterministic=True,
    render=False
)

# 🏋️ Entraînement
print(f"🚀 Entraînement PPO pour {total_timesteps} timesteps...")
model.learn(total_timesteps=total_timesteps, callback=eval_callback)
model.save("ppo_gridworld")
print("✅ Modèle sauvegardé.")

# 📈 Sauvegarde des métriques
metrics = {"episode_rewards": eval_callback.evaluations_results}
save_metrics(metrics, folder=config["paths"]["logs"])
env.close()
print("🏁 Entraînement terminé.")