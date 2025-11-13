# 🧠 GridWorld Reinforcement Learning Project

## 🎯 Objectif du projet

Développer et entraîner des agents d’apprentissage par renforcement (RL) capables d’évoluer dans un environnement **GridWorld** configurable, de la version simple (v1) à la version multi-agents compétitive (v3).
Le projet démontre l’usage de **Stable-Baselines3**, de **Gymnasium**, et de **RL-Baselines3-Zoo** dans un cadre modulaire et extensible.

---

## 🗂️ Structure du projet

```bash
gridworld_project/
│
├── envs/
│   ├── __init__.py                  # Enregistrement des environnements Gym
│   ├── gridworld_env_v1.py          # Environnement simple (1 agent, 1 but)
│   ├── gridworld_env_v2.py          # Version améliorée (configurable, obstacles)
│   ├── gridworld_env_v3_multi.py    # Version multi-agents, compétitive ou coopérative
│
├── training/
│   ├── __init__.py
│   ├── train_ppo_v1.py              # Entraînement PPO pour GridWorld-v1
│   ├── train_ppo_v2.py              # Entraînement PPO pour GridWorld-v2
│   ├── train_ppo_v3.py              # Entraînement PPO multi-agents (v3)
│
├── evaluation/
│   ├── __init__.py
│   ├── evaluate_model_v1.py         # Évaluation d’un modèle entraîné (v1)
│   ├── evaluate_model_v2.py         # Évaluation et rendu matplotlib (v2)
│   ├── record_videos_v2.py          # Génération de vidéos MP4 via matplotlib
│
├── rl_zoo_configs/
│   ├── GridWorld-v3.yml             # Hyperparamètres pour RL-Baselines3-Zoo
│
├── logs/                            # Journaux, snapshots de config, checkpoints
├── tensorboard/                     # Logs TensorBoard
├── train_gridworld_v3.py            # Wrapper pour utiliser RL Zoo directement
└── README.md                        # (ce fichier)
```

---

## ⚙️ Installation

### 1️⃣ Créer l’environnement virtuel

```bash
conda create -n rl-sb python=3.12
conda activate rl-sb
```

### 2️⃣ Installer les dépendances

Installe les packages nécessaires pour RL et l’environnement GridWorld :

```bash
pip install stable-baselines3[extra] gymnasium matplotlib numpy
pip install rl-baselines3-zoo pygame
```

💡 Pour plus de détails et options d’installation, consulte la documentation officielle :

* **Stable-Baselines3** : [Quickstart Guide](https://stable-baselines3.readthedocs.io/en/master/guide/quickstart.html)
* **RL Baselines3 Zoo** : [Documentation](https://rl-baselines3-zoo.readthedocs.io/en/master/index.html)


### 3️⃣ Vérifier que tout est fonctionnel

```bash
python -c "import gymnasium as gym; import envs; env=gym.make('GridWorld-v3'); print(env.reset())"
```

---

## 🚀 Entraînement

### 🔹 Version 1 – Basique

```bash
python training/train_ppo_v1.py
```

### 🔹 Version 2 – Configurable avec obstacles

```bash
python training/train_ppo_v2.py
```

### 🔹 Version 3 – Multi-agents (compétitif/co-opératif)

```bash
python training/train_ppo_v3.py
```

### 🔹 Avec RL-Baselines3-Zoo

```bash
python train_gridworld_v3.py --algo ppo --env GridWorld-v3 --conf rl_zoo_configs/GridWorld-v3.yml --tensorboard-log ./tensorboard/
```

---

## 🧩 Configuration RL Zoo

📄 `rl_zoo_configs/GridWorld-v3.yml`

```yaml
GridWorld-v3:
  n_envs: 1
  n_timesteps: 100000
  policy: 'MlpPolicy'
  n_steps: 128
  batch_size: 64
  gamma: 0.99
  gae_lambda: 0.95
  ent_coef: 0.01
  learning_rate: 2.5e-4
  clip_range: 0.2
  n_epochs: 10
  vf_coef: 0.5
  max_grad_norm: 0.5
  use_sde: False
  normalize_advantage: True
  policy_kwargs: "dict(net_arch=[128, 128])"
```

## 📊 Visualiser les métriques avec TensorBoard

Chaque script d’entraînement (`train_ppo_v1.py`, `train_ppo_v2.py`, `train_ppo_v3.py`) est configuré pour logger les métriques vers TensorBoard. Tu peux ainsi suivre en temps réel :

* Reward moyen par épisode
* Policy loss et value loss
* Entropy
* Gradients et norm
* Et plus selon la version

### 1️⃣ Lancer TensorBoard

Ouvre un terminal à la racine du projet et exécute :

```bash
tensorboard --logdir ./tensorboard/
```

Par défaut, TensorBoard s’ouvre sur [http://localhost:6006](http://localhost:6006).

---

### 2️⃣ Organisation des logs

Pour ne pas mélanger les versions, chaque script d’entraînement écrit dans un sous-dossier dédié :

| Version | Script d’entraînement           | Dossier TensorBoard     |
| ------- | ------------------------------- | ----------------------- |
| v1      | `train_ppo_v1.py`               | `./tensorboard/PPO_v1/` |
| v2      | `train_ppo_v2.py`               | `./tensorboard/PPO_v2/` |
| v3      | `train_ppo_v3.py` (multi-agent) | `./tensorboard/PPO_v3/` |

💡 Astuce : tu peux modifier la variable `tensorboard_log` dans le script pour changer le dossier de sortie.

---

### 3️⃣ Exemple d’utilisation

Pour entraîner et suivre v2 :

```bash
python training/train_ppo_v2.py
tensorboard --logdir ./tensorboard/PPO_v2/
```

Ouvre ensuite ton navigateur sur [http://localhost:6006](http://localhost:6006) pour visualiser les métriques en temps réel pendant l’entraînement.

---

### 4️⃣ Conseils pratiques

* Si tu relances un entraînement, TensorBoard fusionnera les nouveaux logs avec les anciens dans le même dossier.
* Pour comparer plusieurs versions, ouvre TensorBoard et coche plusieurs courbes (`PPO_v1`, `PPO_v2`, `PPO_v3`) simultanément.

### 🔹 Enregistrement de vidéos

Exécuter :

```bash
python evaluation/record_videos_v2.py
```

Les vidéos seront sauvegardées dans `./videos/`.

---

## 🧪 Évaluation d’un modèle

Exemple (version 2 ou 3) :

```bash
python evaluation/evaluate_model_v2.py --model_path ./logs/best_model.zip
```

Cela affiche le comportement de l’agent dans la grille et peut générer un score moyen sur plusieurs épisodes.

---

## ⚙️ Exemple de configuration personnalisée

Tu peux créer des environnements paramétrables :

```python
from envs.gridworld_env_v3_multi import GridWorldMultiAgentEnv
env = GridWorldMultiAgentEnv({
    "grid_size": 8,
    "n_agents": 3,
    "n_goals": 2,
    "n_obstacles": 5,
    "max_steps": 150,
    "obstacle_mode": "fixed"
})
obs, info = env.reset()
env.render()
```

---

## 🧠 Points clés du projet

* **v1** : un agent simple avec apprentissage basique.
* **v2** : ajout d’obstacles, d’une grille configurable et rendu visuel.
* **v3** : environnement **multi-agents** avec collisions, coopération et compétition.
* **Support complet RL-Zoo** pour un entraînement reproductible.
* **Modularité maximale** → tous les scripts sont indépendants et extensibles.

---

## 💾 Sauvegarde & Reprise

Les modèles sont sauvegardés automatiquement dans :

```
./logs/PPO_GridWorld-vX_<timestamp>/best_model.zip
```

Pour reprendre un entraînement :

```python
from stable_baselines3 import PPO
model = PPO.load("logs/best_model.zip")
```

---

## 🧩 Auteur

👨‍💻 **Marc Thierry Nankouli**
Élève ingénieur en IA et Data Technologies
Projet personnel de recherche en apprentissage par renforcement et conception d’environnements simulés.
