# Mario AI

Intelligence artificielle conçue pour jouer à Super Mario Bros. de manière autonome, à partir des seules données visuelles du jeu.

Le projet repose sur deux modèles :

- **Détection** : un modèle YOLOv8 repère et identifie les objets à l'écran (Mario, ennemis, obstacles, blocs), à partir d'un jeu de données annoté manuellement.
- **Décision** : un modèle d'apprentissage par renforcement, inspiré de l'architecture World Models (VAE et MDN-RNN), exploite ces données pour choisir les actions de jeu.

## État du projet

Projet personnel initié en terminale (2023-2024), non finalisé.

- Jeu de données annoté et modèle de détection entraîné :
  [Mario IA DataBase Classification](https://app.roboflow.com/baptiste-hustaix-znm0u/mario-ia-database-classification/3)
  [Mario AI DataBase Detection](https://app.roboflow.com/baptiste-hustaix-znm0u/mario-ai-database-detection/1)
  
- Modèle de décision : en cours de développement, non abouti

## Technologies

- Python, PyTorch
- YOLOv8 (détection d'objets)
- Roboflow (annotation du jeu de données)
- BizHawk (émulateur, interface avec le jeu)

## Structure

| Élément | Rôle |
|---|---|
| `YoloV8/detect` | Modèle de détection d'objets |
| `CustomDataset.py` | Chargement du jeu de données |
| `VAE.py` | Encodage des images du jeu |
| `MDN-RNN.py` | Modèle de prédiction des états du jeu |
| `train.py`, `test.py` | Entraînement et évaluation |
| `main.py` | Point d'entrée |
| `BizHawk`, `birdseyelib.py` | Émulateur et communication avec le jeu |

## Licence

Distribué sous licence Apache 2.0, voir [LICENSE](LICENSE).

## Auteur

Baptiste Hustaix, [@NicogameFR](https://github.com/NicogameFR)
