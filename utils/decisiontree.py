import numpy as np
import pandas as pd
import json
import os

def gini_impurity(y):
    """Calcule l'impureté de Gini"""
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return 1 - np.sum(probs**2)

def best_seuil(X_col, y):
    """Trouve le meilleur seuil pour une colonne"""
    bests = {"seuil": None, "gini": float("inf")}
    valeurs = np.unique(X_col)
    for i in range(len(valeurs) - 1):
        seuil = (valeurs[i] + valeurs[i+1]) / 2
        gauche = y[X_col <= seuil]
        droite = y[X_col > seuil]
        gini_g = gini_impurity(gauche)
        gini_d = gini_impurity(droite)
        gini_total = (len(gauche) * gini_g + len(droite) * gini_d) / len(y)
        if gini_total < bests["gini"]:
            bests["seuil"] = seuil
            bests["gini"] = gini_total
    return bests["seuil"]

def best_split(X, y):
    """Trouve la meilleure division pour l'arbre"""
    best = {"feature": None, "seuil": None, "gini": float("inf")}
    for col in X.columns:
        seuil = best_seuil(X[col].values, y)
        if seuil is None:
            continue
        gauche = y[X[col] <= seuil]
        droite = y[X[col] > seuil]
        gini_total = (
            len(gauche) * gini_impurity(gauche) +
            len(droite) * gini_impurity(droite)
        ) / len(y)
        if gini_total < best["gini"]:
            best.update({"feature": col, "seuil": seuil, "gini": gini_total})
    return best

def create_tree(X, y, profondeur=0, max_profondeur=5):
    """Crée un arbre de décision"""
    # Condition d'arrêt
    if len(np.unique(y)) == 1 or profondeur == max_profondeur:
        return {"label": y.iloc[0], "leaf": True}
    # Trouve la meilleure division
    split = best_split(X, y)
    if split["feature"] is None:
        return {"label": y.mode()[0], "leaf": True}
    feature = split["feature"]
    seuil = split["seuil"]
    # Division des données
    gauche_idx = X[feature] <= seuil
    droite_idx = X[feature] > seuil
    # Création récursive des sous-arbres
    tree = {
        "feature": feature,
        "seuil": seuil,
        "gini": split["gini"],
        "left": create_tree(X[gauche_idx], y[gauche_idx], profondeur+1, max_profondeur),
        "right": create_tree(X[droite_idx], y[droite_idx], profondeur+1, max_profondeur),
        "leaf": False
    }
    return tree

def save_tree(tree, filename="cache/tree_cache/decision_tree.json"):
    """Sauvegarde l'arbre dans un fichier JSON"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        json.dump(tree, f)

def load_tree(filename="cache/tree_cache/decision_tree.json"):
    """Charge l'arbre depuis un fichier JSON"""
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    return None

def predict_with_tree(features, tree):
    """Fait une prédiction avec l'arbre de décision"""
    if tree.get("leaf", False):
        return tree["label"]
    if features[tree["feature"]] <= tree["seuil"]:
        return predict_with_tree(features, tree["left"])
    else:
        return predict_with_tree(features, tree["right"])