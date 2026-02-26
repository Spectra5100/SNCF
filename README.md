# TARDIS - SNCF Data Analysis Service 

##  Présentation du Projet
Ce projet consiste à intégrer le nouveau service d'analyse de données de la SNCF. L'objectif est d'améliorer l'efficacité des trajets ferroviaires en analysant les données historiques des retards pour identifier des schémas et développer un modèle prédictif capable d'anticiper les retards.

Le résultat final est un dashboard interactif Streamlit destiné à aider les voyageurs à mieux planifier leurs trajets en visualisant les statistiques et en obtenant des prédictions en temps réel.

##  Stack Technique
* **Langage** : Python
* **Bibliothèques** : pandas, numpy, matplotlib, seaborn, scikit-learn
* **Interface** : Streamlit
* **Formatage & Style** : Ruff

##  Structure des Livrables
Le dépôt contient les fichiers suivants, conformément aux exigences du projet :

| Fichier | Description |
| :--- | :--- |
| `requirements.txt` | Liste des dépendances Python nécessaires. |
| `tardis_eda.ipynb` | Notebook de nettoyage et d'analyse exploratoire des données. |
| `cleaned_dataset.csv` | Dataset traité et nettoyé après l'étape d'EDA. |
| `tardis_model.ipynb` | Notebook d'entraînement et d'évaluation du modèle. |
| `model.pkl` | Fichier du modèle de régression sauvegardé. |
| `tardis_dashboard.py` | Script de l'application interactive Streamlit. |
| `README.md` | Documentation et instructions (ce fichier). |

##  Étapes du Projet

### 1. Exploration et Nettoyage (EDA)
* Traitement des valeurs manquantes et des doublons.
* Conversion des types de données (dates, catégories).
* Feature Engineering : création d'au moins une nouvelle caractéristique utile.

### 2. Analyse Visuelle
* Calcul de statistiques descriptives.
* Création d'au moins 3 visualisations pertinentes pour comprendre les causes des retards.

### 3. Modélisation Prédictive
* Développement d'un modèle de régression pour prédire le retard en minutes.
* Comparaison par rapport à une baseline (moyenne).

### 4. Dashboard Streamlit
* Affichage de graphiques et d'indicateurs clés.
* Interface permettant de saisir des paramètres de trajet pour obtenir une prédiction.
* Utilisation d'au moins un élément interactif (filtre ou sélecteur).

##  Qualité du Code
Le code doit être formaté et vérifié avec **Ruff** pour garantir la propreté et la lisibilité :
```bash
ruff format .
ruff check .