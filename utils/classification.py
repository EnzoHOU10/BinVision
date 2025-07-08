from models.models import TrashImage, ClassificationRule, Settings, get_rule
from utils.decisiontree import load_tree, predict_with_tree

def calculate_seuils_from_db():
    """Calcule les seuils basés sur les données de la base"""
    # Initialisation des structures
    seuils = {name: [0, 900, 0] for name in [
        "avg_color_r", "avg_color_g", "avg_color_b", "filesize_kb",
        "width", "height", "contrast", "saturation", "luminosity",
        "edge_density", "entropy", "texture_variance", "edge_energy",
        "top_variance", "top_entropy", "hist_peaks", "dark_ratio",
        "bright_ratio", "color_uniformity", "circle_count"
    ]}
    seuils_plein = {name: [0, 900, 0] for name in seuils.keys()}
    seuils_vide = {name: [0, 900, 0] for name in seuils.keys()}
    rules = ClassificationRule.query.all()
    images = TrashImage.query.all()
    cpt = 0
    cpt_plein = 0
    cpt_vide = 0
    for image in images:
        if image.type == 'Manual':
            cpt += 1
            if image.annotation == 'Pleine':
                cpt_plein += 1
            else:
                cpt_vide += 1
            for rule in rules:
                value = getattr(image, rule.name, None)
                if value is not None:
                    seuils[rule.name][0] += value
                    if value < seuils[rule.name][1]:
                        seuils[rule.name][1] = value
                    if value > seuils[rule.name][2]:
                        seuils[rule.name][2] = value
                        
                    if image.annotation == 'Pleine':
                        seuils_plein[rule.name][0] += value
                        if value < seuils_plein[rule.name][1]:
                            seuils_plein[rule.name][1] = value
                        if value > seuils_plein[rule.name][2]:
                            seuils_plein[rule.name][2] = value
                    else:
                        seuils_vide[rule.name][0] += value
                        if value < seuils_vide[rule.name][1]:
                            seuils_vide[rule.name][1] = value
                        if value > seuils_vide[rule.name][2]:
                            seuils_vide[rule.name][2] = value
    if cpt > 0:
        for rule in rules:
            seuils[rule.name][0] = seuils[rule.name][0] / cpt
        if cpt_plein > 0:
            for rule in rules:
                seuils_plein[rule.name][0] = seuils_plein[rule.name][0] / cpt_plein       
        if cpt_vide > 0:
            for rule in rules:
                seuils_vide[rule.name][0] = seuils_vide[rule.name][0] / cpt_vide
        return seuils, seuils_plein if cpt_plein > 0 else None, seuils_vide if cpt_vide > 0 else None
    return None, None, None

def classify_image(filesize_kb, avg_r, avg_g, avg_b, width, height, contrast, 
                  saturation, luminosity, edge_density, entropy, texture_variance, 
                  edge_energy, top_variance, top_entropy, hist_peaks, dark_ratio, 
                  bright_ratio, color_uniformity, circle_count, tree=None):
    """Classifie une image comme 'Pleine' ou 'Vide'"""
    features = {
        "width": width,
        "height": height,
        "filesize_kb": filesize_kb,
        "avg_color_r": avg_r,
        "avg_color_g": avg_g,
        "avg_color_b": avg_b,
        "contrast": contrast,
        "saturation": saturation,
        "luminosity": luminosity,
        "edge_density": edge_density,
        "entropy": entropy,
        "texture_variance": texture_variance,
        "edge_energy": edge_energy,
        "top_variance": top_variance,
        "top_entropy": top_entropy,
        "hist_peaks": hist_peaks,
        "dark_ratio": dark_ratio,
        "bright_ratio": bright_ratio,
        "color_uniformity": color_uniformity,
        "circle_count": circle_count,
    }
    settings = Settings.query.first()
    if settings.use_auto_rules:
        return _classify_with_rules(features)
    else:
        return _classify_with_tree(features, tree)

def _classify_with_rules(features):
    """Classification basée sur les règles"""
    seuils, seuils_plein, seuils_vide = calculate_seuils_from_db()
    score = 0
    # Règles de base
    if features["height"] > get_rule("height"):
        score += 1
    if features["width"] > get_rule("width"):
        score += 1
    if features["filesize_kb"] > get_rule("filesize_kb"):
        score += 2
    if features["edge_density"] > get_rule("edge_density"):
        score += 2
    if features["contrast"] > get_rule("contrast"):
        score += 1
    if features["entropy"] > get_rule("entropy"):
        score += 1
    if features["edge_energy"] > get_rule("edge_energy"):
        score += 1
    if features["top_entropy"] > get_rule("top_entropy"):
        score += 1
    if features["dark_ratio"] > get_rule("dark_ratio"):
        score += 1
    if features["saturation"] < get_rule("saturation"):
        score += 1
    if features["luminosity"] < get_rule("luminosity"):
        score += 1
    if features["avg_color_r"] < get_rule("avg_color_r"):
        score += 1
    if features["avg_color_g"] < get_rule("avg_color_g"):
        score += 1
    if features["avg_color_b"] < get_rule("avg_color_b"):
        score += 1
    if features["texture_variance"] < get_rule("texture_variance"):
        score += 1
    if features["bright_ratio"] < get_rule("bright_ratio"):
        score -= 1
    if features["color_uniformity"] > get_rule("color_uniformity"):
        score -= 1
    if features["circle_count"] > get_rule("circle_count"):
        score -= 1
    if features["hist_peaks"] < get_rule("hist_peaks"):
        score += 1
    # Règles combinées
    if (features["dark_ratio"] > get_rule("dark_ratio") and features["avg_color_r"] < get_rule("avg_color_r") and 
        features["avg_color_g"] < get_rule("avg_color_g") and features["avg_color_b"] < get_rule("avg_color_b")):
        score += 2
    if features["entropy"] > get_rule("entropy") and features["texture_variance"] > get_rule("texture_variance"):
        score += 1
    if features["saturation"] < get_rule("saturation") and features["luminosity"] < get_rule("luminosity"):
        score += 1
    if features["bright_ratio"] > get_rule("bright_ratio") and features["dark_ratio"] < get_rule("dark_ratio"):
        score -= 2
    if features["edge_density"] < get_rule("edge_density") and features["entropy"] < get_rule("entropy"):
        score -= 2
    if features["circle_count"] >= get_rule("circle_count") and features["color_uniformity"] > get_rule("color_uniformity"):
        score -= 2
    if features["filesize_kb"] < get_rule("filesize_kb") and features["contrast"] < get_rule("contrast"):
        score -= 2
    # Comparaison avec les seuils calculés
    if seuils_plein and seuils_vide:
        def in_range(val, minv, maxv):
            return minv <= val <= maxv
        for name, val in features.items():
            if name in seuils_plein and name in seuils_vide:
                plein_min, plein_max = seuils_plein[name][1], seuils_plein[name][2]
                vide_min, vide_max = seuils_vide[name][1], seuils_vide[name][2]
                if in_range(val, plein_min, plein_max) and not in_range(val, vide_min, vide_max):
                    score += 1
                if not in_range(val, plein_min, plein_max) and in_range(val, vide_min, vide_max):
                    score -= 1
    return "Pleine" if score >= 7 else "Vide"

def _classify_with_tree(features, tree):
    """Classification basée sur l'arbre de décision."""
    if tree is None:
        tree = load_tree()
        if tree is None:
            return "Vide"
    def predict(features, tree):
        if tree.get("leaf", False):
            return tree["label"]
        if features[tree["feature"]] <= tree["seuil"]:
            return predict(features, tree["left"])
        else:
            return predict(features, tree["right"])
    return predict(features, tree)