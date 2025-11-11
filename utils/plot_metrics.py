# utils/plot_metrics.py
import matplotlib.pyplot as plt
import json

def plot_rewards_from_json(json_file: str, title: str = "Rewards"):
    """
    Lit un fichier JSON contenant une liste de récompenses et trace la courbe.
    Exemple de JSON : {"episode_rewards": [1, 2, 3, 4, 5]}
    """
    with open(json_file, "r") as f:
        data = json.load(f)
    
    rewards = data.get("episode_rewards", [])
    if not rewards:
        print("⚠️ Aucun reward trouvé dans le fichier JSON")
        return
    
    plt.figure(figsize=(8,5))
    plt.plot(rewards, label="Reward par épisode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
