import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.callbacks import EvalCallback

from envs.gridworld_env_v1 import GridWorldEnv

# 1️⃣ Créer et envelopper l’environnement
env = GridWorldEnv(size=5)
env = Monitor(env)
env = DummyVecEnv([lambda: env])

# 2️⃣ Ajouter un enregistreur vidéo
video_folder = "./videos/"
env = VecVideoRecorder(
    env,
    video_folder=video_folder,
    record_video_trigger=lambda x: x % 2000 == 0,
    video_length=200,
    name_prefix="ppo-gridworld"
)

# 3️⃣ Créer le modèle PPO
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./ppo_gridworld_tensorboard/"
)

# 4️⃣ Définir un callback d’évaluation
eval_env = DummyVecEnv([lambda: GridWorldEnv(size=5)])
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./best_model/",
    log_path="./logs/",
    eval_freq=2000,
    deterministic=True,
    render=False
)

# 5️⃣ Entraîner le modèle
model.learn(total_timesteps=20_000, callback=eval_callback)

# 6️⃣ Sauvegarder le modèle
model.save("ppo_gridworld")
print("✅ Entraînement terminé et modèle sauvegardé !")

env.close()
