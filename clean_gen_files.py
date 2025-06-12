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
    "huge-flow"
]

deleted = 0
for folder in dirs_to_remove:
    if os.path.isdir(folder):
        print(f"Suppression de {folder}")
        shutil.rmtree(folder)
        deleted += 1
    else:
        print(f"{folder} introuvable, ignoré.")

print(f"\nNettoyage terminé. {deleted} dossier(s) supprimé(s).")
