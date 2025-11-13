# training/train_ppo_v2.py
import os
import sys
import json
from datetime import datetime

# Permet d'importer depuis la racine du projet (pratique pour exécution directe)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

from envs.gridworld_env_v2 import GridWorldEnv
from utils.config import config
from utils.logger import save_metrics

# Optionnel : importer la fonction d'enregistrement vidéo si elle existe
try:
    from evaluation.record_videos import record_video
    _has_record_video = True
except Exception:
    _has_record_video = False


def ensure_paths():
    for p in config["paths"].values():
        os.makedirs(p, exist_ok=True)


def save_config_snapshot(out_folder: str):
    os.makedirs(out_folder, exist_ok=True)
    snapshot_path = os.path.join(out_folder, f"config_used_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(snapshot_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"📁 Config snapshot saved to: {snapshot_path}")


def make_env_fn(grid_size, render_mode):
    def _init():
        env = GridWorldEnv(grid_size=grid_size, n_obstacles=config.get("env", {}).get("n_obstacles", 2),
                           render_mode=render_mode)
        env = Monitor(env)
        return env
    return _init


def main():
    ensure_paths()

    env_size = config["env"].get("size", 5)
    # for training prefer non-rendering mode (faster)
    train_render_mode = config["env"].get("render_mode", "none")
    video_folder = config["paths"]["video_folder"]
    tb_log = config["paths"]["tensorboard_log"]
    best_model_path = config["paths"]["best_model"]
    logs_path = config["paths"]["logs"]

    # snapshot config
    save_config_snapshot(logs_path)

    # --- Create training environment (vectorized)
    train_env = DummyVecEnv([make_env_fn(env_size, train_render_mode)])

    # --- Prepare model kwargs from config (filter out total_timesteps)
    ppo_cfg = dict(config["ppo"])
    total_timesteps = ppo_cfg.pop("total_timesteps", config["training"].get("total_timesteps", 100))

    # optional policy kwargs example (you can edit in utils/config.py)
    policy_kwargs = config.get("policy_kwargs", None)
    if policy_kwargs:
        ppo_cfg["policy_kwargs"] = policy_kwargs

    print("🧩 PPO hyperparams used:")
    for k, v in ppo_cfg.items():
        if k != "policy_kwargs":
            print(f"  - {k}: {v}")

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log=tb_log,
        **ppo_cfg
    )

    # --- Eval callback: use a non-rendering eval env for speed
    eval_env = DummyVecEnv([make_env_fn(env_size, "none")])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_path,
        log_path=logs_path,
        eval_freq=config["training"].get("eval_freq", 2000),
        n_eval_episodes=config["training"].get("n_eval_episodes", 5),
        deterministic=True,
        render=False
    )

    # --- Train
    print(f"🚀 Starting training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    print("✅ Training finished.")

    # --- Save final model
    final_model_name = os.path.join(best_model_path, "ppo_gridworld_v2")
    model.save(final_model_name)
    print(f"💾 Model saved to: {final_model_name}")

    # --- Evaluate policy on a few episodes
    print("📊 Evaluating policy (deterministic)...")
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=config["training"].get("n_eval_episodes", 5), deterministic=True)
    metrics = {
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward)
    }
    save_metrics(metrics, folder=logs_path, filename=f"eval_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    # --- Optional: record video if helper available and env supports rgb_array
    # GridWorld v2 uses matplotlib rendering (render_mode="human" or "none").
    # VecVideoRecorder used by many utilities requires 'rgb_array' support.
    # We therefore only attempt to record using evaluation.record_videos.record_video()
    # if that helper is present. Otherwise we skip and inform the user.
    if _has_record_video:
        try:
            print("🎬 Attempting to record a demo video using evaluation.record_videos.record_video() ...")
            # model path to pass: use the saved final model path (stable-baselines adds .zip when saving)
            model_path = final_model_name
            # call helper (record_video should handle env creation and render_mode)
            record_video(model_path=model_path, video_name="ppo_gridworld_v2_demo", video_length=200)
        except Exception as e:
            print(f"⚠️ Recording failed or not supported for this env: {e}")
    else:
        print("ℹ️ evaluation.record_videos not available — skipping automatic video recording.")
        print("   -> You can run evaluation/record_videos.py manually after adapting it to gridworld_v2 (render_mode).")

    # close envs
    train_env.close()
    eval_env.close()
    print("🏁 Done.")


if __name__ == "__main__":
    main()
