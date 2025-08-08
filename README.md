## Installation et exécution
---
### 1️⃣ Installation :

```bash
pip install -r requirements.txt
```
### Execution : 

```python
python deploiement.py
```
---

## Plan du projet

| N° | Partie | Description |
|----|--------|-------------|
| 1 | Importation des bibliothèques | Chargement des modules nécessaires (Keras, NumPy, Matplotlib, etc.) |
| 2 | Chargement du dataset | Lecture et préparation des images avec `ImageDataGenerator` |
| 3 | Architecture du modèle | Utilisation d’InceptionV3 avec couches personnalisées |
| 4 | Compilation | Configuration de l’optimiseur, fonction de perte et métriques |
| 5 | Entraînement | Apprentissage du modèle sur les données d’entraînement |
| 6 | Évaluation | Mesure des performances sur train et validation |
| 7 | Matrice de confusion | Visualisation des prédictions par classe |
| 8 | Prédiction visuelle | Affichage d’images avec labels réels et prédits |
| 9 | Sauvegarde du modèle | Enregistrement du modèle entraîné au format `.h5` |


---



##  Exemples de prédiction

**Exemple (1)**  
<img width="451" height="251" alt="cht1" src="https://github.com/user-attachments/assets/e089e814-166e-420f-b8e0-c89d557f4172" />

**Exemple (2)**  
<img width="450" height="250" alt="doge1" src="https://github.com/user-attachments/assets/224779a7-0236-4b07-9b88-d377f45c000c" />

---

##  Classes du modèle

| N°  | Classe     |
|-----|------------|
| 1   | butterfly  |
| 2   | cat        |
| 3   | chicken    |
| 4   | cow        |
| 5   | dog        |
| 6   | elephant   |
| 7   | horse      |
| 8   | sheep      |
| 9   | spider     |
| 10  | squirrel   |


