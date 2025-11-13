# training/train_ppo_v3.py
import sys, os, json

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from datetime import datetime
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from envs.gridworld_env_v3_multi import GridWorldMultiAgentEnv
from utils.config import config

def make_env():
    """Crée l'environnement multi-agent"""
    return GridWorldMultiAgentEnv(config=config["env_v3"])

def main():
    # --- Préparation ---
    log_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(config["paths"]["logs"], exist_ok=True)
    os.makedirs(config["paths"]["best_model"], exist_ok=True)
    os.makedirs(config["paths"]["tensorboard_log"], exist_ok=True)

    # Sauvegarder la config utilisée
    snapshot_path = os.path.join(config["paths"]["logs"], f"config_v3_{log_time}.json")
    with open(snapshot_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"📁 Config sauvegardée dans {snapshot_path}")

    # --- Environnement ---
    env = DummyVecEnv([lambda: Monitor(make_env())])
    eval_env = DummyVecEnv([lambda: make_env()])

    # --- Modèle PPO ---
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=config["paths"]["tensorboard_log"],
        **config["ppo_v3"]
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=config["paths"]["best_model"],
        log_path=config["paths"]["logs"],
        eval_freq=2000,
        deterministic=True,
        render=False,
    )

    print("🚀 Début de l'entraînement PPO multi-agent...")
    model.learn(total_timesteps=config["training"]["total_timesteps_v3"], callback=eval_callback)
    model.save(os.path.join(config["paths"]["best_model"], "ppo_gridworld_v3"))
    print("✅ Entraînement terminé et modèle sauvegardé !")

if __name__ == "__main__":
    main()
