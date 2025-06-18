import os
import shutil

# Tous les dossiers à supprimer, y compris 'xps'
dirs_to_remove = [
    "xps",
    "small-ext",
    "smallish-ext",
    "medium-ext",
    "medium-plus-ext",
    "medium-plus-plus-ext",
    "medium-large-ext",
    "medium-large-plus-ext",
    "largish-ext",
    "largish-plus-ext",
    "very-large-ext",
    "very-very-large-ext",
    "huge-ext",
    "small-flow-ext",
    "smallish-flow-ext",
    "medium-flow-ext",
    "medium-plus-flow-ext",
    "medium-plus-plus-flow-ext",
    "medium-large-flow-ext",
    "medium-large-plus-flow-ext",
    "largish-flow-ext",
    "largish-plus-flow-ext",
    "very-large-flow-ext",
    "very-very-large-flow-ext",
    "huge-flow-ext",
    "medium-large-plus-flow",
    "largish-flow",
    "largish-plus-flow",
    "very-large-flow",
    "very-very-large-flow",
    "huge-flow",
    "tree_exp",
    "large_tree_exp",
    "Test"
]

deleted_dirs = 0
for folder in dirs_to_remove:
    if os.path.isdir(folder):
        print(f"Suppression de {folder}")
        shutil.rmtree(folder)
        deleted_dirs += 1
    else:
        print(f"{folder} introuvable, ignoré.")
        
# Suppression des fichiers .slurm dans le dossier courant
slurm_files = [f for f in os.listdir(".") if f.endswith(".slurm")]
deleted_files = 0
for f in slurm_files:
    try:
        os.remove(f)
        print(f"Fichier supprimé : {f}")
        deleted_files += 1
    except Exception as e:
        print(f"Erreur lors de la suppression de {f} : {e}")

print(f"\nNettoyage terminé. {deleted_dirs} dossier(s) supprimé(s), {deleted_files} fichier(s) .slurm supprimé(s).")
