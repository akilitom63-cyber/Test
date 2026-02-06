# Modèle d'apprentissage

Ce dépôt contient un exemple minimal de modèle d'apprentissage supervisé. Le script
`model_apprentissage.py` implémente une régression linéaire 1D entraînée par descente
de gradient, sans dépendance externe.

## Utilisation

```bash
python model_apprentissage.py
```

Le script :
- génère un jeu de données synthétique `y = 3x + 2`,
- entraîne le modèle,
- affiche les paramètres appris et quelques prédictions.
