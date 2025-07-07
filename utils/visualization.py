import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import pandas as pd
import os
from sqlalchemy.orm import class_mapper
from models.models import TrashImage

def generate_matplotlib(output_dir='static/matplotlib'):
    """Génère les graphiques matplotlib pour l'analyse des données"""
    images = TrashImage.query.all()
    os.makedirs(output_dir, exist_ok=True)
    # Conversion des données en DataFrame
    data = []
    for img in images:
        row = {col.key: getattr(img, col.key) for col in class_mapper(TrashImage).columns}
        data.append(row)
    df = pd.DataFrame(data)
    # Définition des sous-ensembles
    subsets = {
        "totalgraph": df,
        "pleinegraph": df[df['annotation'] == 'Pleine'],
        "videgraph": df[df['annotation'] == 'Vide'],
    }
    filenames = {
        "totalgraph": "totalgraph.png",
        "pleinegraph": "pleinegraph.png",
        "videgraph": "videgraph.png",
    }
    paths = []
    for key, subset_df in subsets.items():
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Analyse des images - {key.capitalize()}", fontsize=16)
        # Graphique 1: Distribution des tailles de fichiers
        sns.histplot(subset_df['filesize_kb'].dropna(), bins=30, color="#245311", ax=axs[0,0])
        axs[0,0].set_title("Distribution Taille fichiers (KB)")
        axs[0,0].set_xlabel("Taille (KB)")
        axs[0,0].set_ylabel("Nombre d'images")
        # Graphique 2: Distribution du contraste
        sns.histplot(subset_df['contrast'].dropna(), bins=30, color='#B22222', ax=axs[0,1])
        axs[0,1].set_title("Distribution Contraste")
        axs[0,1].set_xlabel("Contraste")
        axs[0,1].set_ylabel("Nombre d'images")
        # Graphique 3: Scatter plot luminosité vs saturation
        sc = axs[1,0].scatter(
            subset_df['luminosity'], 
            subset_df['saturation'], 
            c=subset_df['filesize_kb'], 
            cmap='viridis', 
            alpha=0.7
        )
        axs[1,0].set_title("Luminosité vs Saturation (couleur = taille fichier)")
        axs[1,0].set_xlabel("Luminosité")
        axs[1,0].set_ylabel("Saturation")
        cbar = fig.colorbar(sc, ax=axs[1,0])
        cbar.set_label('Taille fichier (KB)')
        # Graphique 4: Barres de métriques
        metrics = ['texture_variance', 'entropy']
        means = subset_df[metrics].mean()
        stds = subset_df[metrics].std()
        axs[1,1].bar(metrics, means, yerr=stds, capsize=8, color=['black', '#B22222'])
        axs[1,1].set_title("Moyenne ± écart-type: Texture LBP & Entropie")
        axs[1,1].set_ylabel("Valeurs")
        axs[1,1].set_ylim(0, max(means + stds)*1.2 if not means.empty else 1)
        axs[1,1].set_facecolor('#f7f7f7')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        # Sauvegarde du graphique
        full_path = os.path.join(output_dir, filenames[key])
        web_path = f"../{output_dir}/{filenames[key]}".replace("//", "/")
        fig.savefig(full_path, bbox_inches='tight')
        plt.close(fig)
        paths.append(web_path)
    return paths