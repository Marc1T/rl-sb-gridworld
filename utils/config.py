# utils/config.py

config = {
    "env": {
        "size": 5,
        "render_mode": None  # "human" pour voir en direct, "rgb_array" pour vidéo
    },
    "ppo": {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "target_kl": 0.03,
        "total_timesteps": 20_000
    },
    "paths": {
        "video_folder": "./videos/",
        "tensorboard_log": "./ppo_gridworld_tensorboard/",
        "best_model": "./best_model/",
        "logs": "./logs/"
    },
    "training": {
        "eval_freq": 2000,
        "n_eval_episodes": 10
    }
}
