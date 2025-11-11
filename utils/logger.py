# utils/logger.py
import os
import json
import datetime

def save_metrics(metrics: dict, folder: str, filename: str = None):
    """
    Sauvegarde un dictionnaire de métriques dans un fichier JSON.
    """
    os.makedirs(folder, exist_ok=True)
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"metrics_{timestamp}.json"
    path = os.path.join(folder, filename)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"✅ Métriques sauvegardées dans {path}")
    return path
