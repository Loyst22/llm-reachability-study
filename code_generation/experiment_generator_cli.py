from experiment import LinearCallExperiment
from experiment import TreeCallExperiment

def prompt_experiment_type():
    print("Quel type d'experience voulez-vous creer ?")
    print("1. Appels lineaires (A -> B -> C...)")
    print("2. Appels en arbre (A -> B+C, etc.)")
    while True:
        choice = input("Entrez 1 ou 2 : ")
        if choice in ["1", "2"]:
            return int(choice)
        
        
def prompt_basic_params():
    name = input("Nom de l'experience (sans espaces) : ")
    n_questions = int(input("Nombre de questions par distance (defaut = 10) : ") or 10)
    n_comments = int(input("Nombre de commentaires a inserer (defaut = 0) : ") or 0)
    n_loops = int(input("Nombre de boucles a inserer (defaut = 0) : ") or 0)
    n_if = int(input("Nombre de blocs if a inserer (defaut = 0) : ") or 0)
    return dict(name=name,
                n_questions_per_distance=n_questions,
                n_comments=n_comments,
                n_loops=n_loops,
                n_if=n_if
                )

def main():
    experiment_type = prompt_experiment_type()
    base_params = prompt_basic_params()

    if experiment_type == 1:
        n_methods = int(input("Nombre de methodes (defaut = 50) : ") or 50)
        exp = LinearCallExperiment(**base_params, n_methods=n_methods)
    else:
        depth = int(input("Profondeur de l'arbre (defaut = 3) : ") or 3)
        branching = int(input("Facteur de branchement (defaut = 2) : ") or 2)
        exp = TreeCallExperiment(**base_params, depth=depth, branching_factor=branching)

    exp.generate()
    print("Experience creee avec succès !")
    print(f"Dossier : {exp.experiment_path}")

if __name__ == "__main__":
    main()