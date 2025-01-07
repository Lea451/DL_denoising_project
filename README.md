# Projet DL Denoising

**Cours de Deep Learning pour le débruitage audio et le traitement du signal**

Ce projet implémente une **architecture ResUnet** pour débruiter des signaux audio, combinant des techniques de deep learning et de traitement du signal.

---

## Structure du projet

| **Dossier/Fichier**         | **Description**                                                                                                                                                                   |
|-----------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `audio_files/`             | Contient les signaux audio utilisés pour le projet : sous-dossiers **clean_signals/** et **noisy_signals/**.                                                                    |
| `checkpoints/`             | Contient les checkpoints des modèles entraînés.                                                                                                                                 |
| `config/`                  | Inclut les fichiers de configuration (ex. : `config_file.py`) définissant les paramètres d'entraînement et les chemins pour enregistrer les résultats.       |
| `data/`                    | Contient les fichiers `.npy` générés à partir des signaux audio.                                                                                              |
| `models/`                  | Implémente l'architecture ResUnet et les modèles associés.                                                                                                           |
| `notebooks/`               | Notebooks Jupyter présentant les résultats, visualisations et évaluations des signaux débruités.                                                             |
| `notebooks/Audio - exemples/` | Contient les fichiers audio d'exemple pour comparer les signaux propres, bruités et débruités.                                                                     |
| `scripts/`                 | Contient les scripts Python pour la préparation des données, l'entraînement et les tests.                                                                               |
| `tests/`                   | Scripts pour les tests unitaires de différentes parties du code.                                                                                                               |
| `compare_signals.py`       | Script pour comparer les signaux originaux et débruités, et calculer les métriques d'évaluation (ex. : STOI).                                                       |
| `main.py`                  | Point d'entrée principal pour l'entraînement et l'évaluation.                                                                                                          |
| `postprocessing.py`        | Fonctions pour le post-traitement des sorties du modèle, y compris l'enregistrement des signaux débruités en fichiers `.wav`.                                          |

---

## Installation et utilisation

### Prétraitement
1. Placez vos fichiers audio dans les répertoires suivants :
   - `audio_files/clean_signals/`
   - `audio_files/noisy_signals/`
2. Lancez le script `scripts/data_utils.py` pour générer les fichiers `.npy` à partir des fichiers audio.

### Entraînement
1. Configurez les paramètres d'entraînement dans `config/config_file.py`.
2. Lancez l'entraînement avec :
   ```bash
   python main.py --exp='config_file'
   ```

### Tests
1. Testez un modèle entraîné avec :
   ```bash
   python main.py --exp='config_file' --evaluate=True --directory='(chemin vers le modèle)'
   ```

---

## Résultats

### Visualisation des formes d'ondes

![Formes d'ondes des modèles](notebooks/Audio_exemples/plot.png) 

### Comparaisons audio
- [Écouter le signal propre (143)](notebooks/Audio_exemples/143_clean.wav)
- [Écouter le signal bruité (143)](notebooks/Audio_exemples/143_noisy.wav)
- [Écouter le signal débruité (modèle 1 - 143)](notebooks/Audio_exemples/143_denoised_1.wav)
- [Écouter le signal débruité (modèle 2 - 143)](notebooks/Audio_exemples/143_denoised_2.wav)

- [Écouter le signal propre (748)](notebooks/Audio_exemples/748_clean.wav)
- [Écouter le signal bruité (748)](notebooks/Audio_exemples/748_noisy.wav)
- [Écouter le signal débruité (modèle 1 - 748)](notebooks/Audio_exemples/748_denoised1.wav)
- [Écouter le signal débruité (modèle 2 - 748)](notebooks/Audio_exemples/748_denoised2.wav)

### Métriques
- **STOI :** 0.95
- **PESQ :** 2.7

---

## Notebooks
Explorez les notebooks dans `notebooks/` pour visualiser les résultats, spectrogrammes et métriques d'évaluation.

---


   ```

---
