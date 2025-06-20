import os
from generator_8lang import ExperimentRunner, ExperimentConfig

def prompt_experiment_type():
    print("Quel type d'experience voulez-vous creer ?")
    print("1. Appels lineaires (A -> B -> C...)")
    print("2. Appels en arbre (A -> B+C...)")
    while True:
        choice = input("Entrez 1 ou 2 : ").strip()
        if choice in ["1", "2"]:
            return "linear" if choice == "1" else "tree"

def prompt_common_fields():
    print("\n--- Parametres communs ---")
    name = input("Nom de l'experience : ").strip()
    if not name:
        raise ValueError("Le nom de l'experience ne peut pas être vide.")
    context_size = int(input("Taille du contexte : "))
    depths_raw = input("Distances (ex: 1,2,3) : ").strip()
    depths = list(map(int, depths_raw.split(","))) if depths_raw else [1, 2, 3]
    n_questions = int(input("Nombre de questions par distance : "))
    n_padding = int(input("Nombre de lignes de padding (defaut 0) : ") or 0)
    n_comment_lines = int(input("Nombre de lignes de commentaire (defaut 0) : ") or 0)
    n_loops = int(input("Nombre de boucles (defaut 0) : ") or 0)
    n_if = int(input("Nombre de blocs if (defaut 0) : ") or 0)
    n_vars = int(input("Nombre de variables (defaut 0) : ") or 0)
    language = input("Langage (java, python, etc.) [default: java] : ").strip() or "java"

    return {
        "name": os.path.join("experiments", name),
        "context_size": context_size,
        "depths": depths,
        "n_questions": n_questions,
        "n_padding": n_padding,
        "n_comment_lines": n_comment_lines,
        "n_loops": n_loops,
        "n_if": n_if,
        "n_vars": n_vars,
        "language": language
    }

def prompt_tree_specific():
    print("\n--- Parametres specifiques a l'appel en arbre ---")
    tree_depth = int(input("Profondeur des appels : ") or 3)
    n_tree = int(input("Nombre d'arbres : ") or 3)
    calls_per_function = int(input("Appels par fonction : ") or 2)
    return {
        "tree_depth": tree_depth,
        "n_tree": n_tree,
        "calls_per_function": calls_per_function
    }

def main():
    try:
        runner = ExperimentRunner()
        experiment_type = prompt_experiment_type()
        common = prompt_common_fields()

        if experiment_type == "tree":
            tree_params = prompt_tree_specific()
            config = ExperimentConfig(
                **common,
                type="tree",
                **tree_params
            )
        else:
            config = ExperimentConfig(
                **common,
                type="linear"
            )

        runner.generate_experiment(config)
        print("\nExperience creee avec succes !")
        print(f"Dossier : {config.name}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nUne erreur est survenue : {e}")

if __name__ == "__main__":
    main()
