# 🐍 Projet Snake avec Q-Learning

## 📚 Introduction
Ce projet implémente un agent intelligent capable d'apprendre à jouer au jeu Snake en utilisant le Q-Learning, une technique d'apprentissage par renforcement. L'agent apprend progressivement à optimiser ses mouvements pour maximiser son score.

## 🎯 Objectifs du Projet
- Implémenter un agent Q-Learning pour le jeu Snake
- Explorer les paramètres d'apprentissage optimaux
- Visualiser l'évolution de l'apprentissage
- Fournir une base de code claire et réutilisable

## 🔬 Concepts Théoriques

### Q-Learning
Le Q-Learning est une méthode d'apprentissage par renforcement qui permet à un agent d'apprendre une politique optimale en interagissant avec son environnement :

1. **Table Q (Q-Table)** : 
   - Stocke les valeurs Q pour chaque paire état-action
   - Q(s,a) représente la récompense attendue pour une action a dans l'état s
   - Se met à jour au fur et à mesure de l'apprentissage

2. **Formule de mise à jour** :
   ```
   Q(s,a) = Q(s,a) + α[R + γ·max(Q(s',a')) - Q(s,a)]
   ```
   où :
   - α : Taux d'apprentissage (learning rate)
   - R : Récompense immédiate
   - γ : Facteur d'actualisation (discount factor)
   - s' : État suivant
   - a' : Action possible dans l'état suivant

3. **Exploration vs Exploitation** :
   - ε-greedy : Équilibre entre exploration de nouvelles actions et exploitation des connaissances
   - Diminution progressive de ε pour favoriser l'exploitation

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