import pandas as pd
from math import ceil, floor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from models.models import TrashImage
from utils.decisiontree import create_tree
from utils.classification import classify_image

def test_manual():
    """Test du modèle fait maison avec données manuelles"""
    imgs = TrashImage.query.filter_by(type='Manual').all()
    # Création du DataFrame
    df = pd.DataFrame([{
        'width': img.width,
        'height': img.height,
        'filesize_kb': img.filesize_kb,
        'avg_color_r': img.avg_color_r,
        'avg_color_g': img.avg_color_g,
        'avg_color_b': img.avg_color_b,
        'contrast': img.contrast,
        'saturation': img.saturation,
        'luminosity': img.luminosity,
        'edge_density': img.edge_density,
        'entropy': img.entropy,
        'texture_variance': img.texture_variance,
        'edge_energy': img.edge_energy,
        'top_variance': img.top_variance,
        'top_entropy': img.top_entropy,
        'hist_peaks': img.hist_peaks,
        'dark_ratio': img.dark_ratio,
        'bright_ratio': img.bright_ratio,
        'color_uniformity': img.color_uniformity,
        'circle_count': img.circle_count,
        'annotation': img.annotation
    } for img in imgs])
    # Mélange des données
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    # Division train/test
    split_idx = ceil(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    # Entraînement
    X_train = train_df.drop("annotation", axis=1)
    y_train = train_df["annotation"]
    tree = create_tree(X_train, y_train)
    # Test
    X_test = test_df.drop("annotation", axis=1)
    y_test = list(test_df["annotation"])
    y_pred = []
    for _, row in X_test.iterrows():
        pred = classify_image(
            row["filesize_kb"], row["avg_color_r"], row["avg_color_g"], row["avg_color_b"],
            row["width"], row["height"], row["contrast"], row["saturation"], row["luminosity"],
            row["edge_density"], row["entropy"], row["texture_variance"], row["edge_energy"],
            row["top_variance"], row["top_entropy"], row["hist_peaks"],
            row["dark_ratio"], row["bright_ratio"], row["color_uniformity"], row["circle_count"],
            tree
        )
        y_pred.append(pred)
    # Calcul des métriques
    total = len(y_test)
    correct = sum(p == t for p, t in zip(y_pred, y_test))
    accuracy = correct / total if total > 0 else 0
    # Matrice de confusion
    labels = sorted(set(y_test + y_pred))
    matrix = {true: {pred: 0 for pred in labels} for true in labels}
    for t, p in zip(y_test, y_pred):
        matrix[t][p] += 1
    # Calcul des métriques par classe
    precision_scores = {}
    recall_scores = {}
    f1_scores = {}
    for label in labels:
        TP = matrix[label][label]
        FP = sum(matrix[other][label] for other in labels if other != label)
        FN = sum(matrix[label][other] for other in labels if other != label)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        precision_scores[label] = precision
        recall_scores[label] = recall
        f1_scores[label] = f1
    # Affichage des résultats
    print(f"\nAccuracy (arbre de décision fait maison) : {accuracy:.3f}")
    print("\nMétriques par classe :")
    for label in labels:
        print(f"Classe {label} :")
        print(f"  - Précision : {precision_scores[label]:.3f}")
        print(f"  - Rappel    : {recall_scores[label]:.3f}")
        print(f"  - F1-score  : {f1_scores[label]:.3f}")
    print("\nMatrice de confusion :")
    header = "       " + "  ".join(f"{label:>5}" for label in labels)
    print(header)
    for true_label in labels:
        row = f"{true_label:>7} " + "  ".join(f"{matrix[true_label][pred_label]:5}" for pred_label in labels)
        print(row)

def test_sklearn():
    """Test avec scikit-learn pour comparaison"""
    imgs = TrashImage.query.filter_by(type='Manual').all()
    # Création du DataFrame
    df = pd.DataFrame([{
        'width': img.width,
        'height': img.height,
        'filesize_kb': img.filesize_kb,
        'avg_color_r': img.avg_color_r,
        'avg_color_g': img.avg_color_g,
        'avg_color_b': img.avg_color_b,
        'contrast': img.contrast,
        'saturation': img.saturation,
        'luminosity': img.luminosity,
        'edge_density': img.edge_density,
        'entropy': img.entropy,
        'texture_variance': img.texture_variance,
        'edge_energy': img.edge_energy,
        'top_variance': img.top_variance,
        'top_entropy': img.top_entropy,
        'hist_peaks': img.hist_peaks,
        'dark_ratio': img.dark_ratio,
        'bright_ratio': img.bright_ratio,
        'color_uniformity': img.color_uniformity,
        'circle_count': img.circle_count,
        'annotation': img.annotation
    } for img in imgs])
    # Mélange des données
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    # Division train/test
    test_size = max(1, floor(len(df) * 0.2))
    train_size = len(df) - test_size
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    # Préparation des données
    X_train = train_df.drop("annotation", axis=1)
    y_train = train_df["annotation"]
    X_test = test_df.drop("annotation", axis=1)
    y_test = test_df["annotation"]
    # Entraînement du modèle scikit-learn
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    # Prédiction
    y_pred = clf.predict(X_test)
    # Évaluation
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy (scikit-learn) : {accuracy:.3f}")
    print("\nMétriques par classe :")
    print(classification_report(y_test, y_pred, digits=3))
    # Matrice de confusion
    matrix = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    print("Matrice de confusion :")
    header = "       " + "  ".join(f"{label:>5}" for label in clf.classes_)
    print(header)
    for i, true_label in enumerate(clf.classes_):
        row = f"{true_label:>7} " + "  ".join(f"{matrix[i][j]:5}" for j in range(len(clf.classes_)))
        print(row)