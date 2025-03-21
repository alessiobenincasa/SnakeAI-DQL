# 🧠 Projet de Deep Learning avec PyTorch

## 📚 Introduction
Ce projet implémente des modèles de deep learning pour la reconnaissance d'images en utilisant PyTorch. Il explore différentes architectures de réseaux de neurones et techniques d'optimisation pour améliorer la précision de la classification.

## 🎯 Objectifs du Projet
- Implémenter des réseaux de neurones convolutifs (CNN) pour la classification d'images
- Explorer différentes architectures de modèles
- Optimiser les performances avec des techniques avancées
- Fournir une base de code claire et réutilisable

## 🔬 Concepts Théoriques

### Réseaux de Neurones Convolutifs (CNN)
Les CNN sont des architectures de deep learning spécialement conçues pour traiter des données structurées en grille, comme les images. Ils utilisent trois concepts principaux :

1. **Couches de Convolution** : 
   - Extraient les caractéristiques locales des images
   - Utilisent des filtres (kernels) qui parcourent l'image
   - Permettent la détection de motifs hiérarchiques

2. **Pooling** :
   - Réduit la dimensionnalité des features maps
   - Améliore la robustesse aux variations de position
   - Types courants : Max Pooling, Average Pooling

3. **Couches Entièrement Connectées** :
   - Combinent les caractéristiques pour la classification finale
   - Transforment les features en prédictions de classes

### Techniques d'Optimisation
- **Batch Normalization** : Normalise les activations pour une convergence plus rapide
- **Dropout** : Prévient le surapprentissage en désactivant aléatoirement des neurones
- **Data Augmentation** : Augmente la diversité des données d'entraînement

## 🛠️ Installation

```bash
# Créer un environnement virtuel
python -m venv venv

# Activer l'environnement
source venv/bin/activate  # Linux/Mac
# ou
.\venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt
```

## 📊 Structure du Projet

- `snake.py`: Main entry point
- `environment/`: Contains the game board and rules
- `agent/`: Q-learning agent implementation
- `visualization/`: Pygame-based visualization
- `models/`: Directory for saved model states 

Run the game with different options:

```bash
# Train for 10 sessions and save the model
./snake -sessions 10 -save models/10sess.txt -visual off

# Load a trained model and run in visual mode
./snake -visual on -load models/100sess.txt -sessions 10 -dontlearn

# Run in step-by-step mode
./snake -visual on -load models/100sess.txt -sessions 10 -dontlearn -step-by-step
``` 