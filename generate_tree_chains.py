from pathlib import Path
import random
import generate_chain as gen
import method_tree
import prompts as p

"""
            Première idée :
            
                                                              ┌────────────────┐
                                                   ┌─────────►│   Fonction 4   │
                                                   │          └────────────────┘
                                    ┌──────────────┴─┐
                          ┌────────►│   Fonction 2   │
                          │         └──────────────┬─┘
                          │                        │          ┌────────────────┐
                          │                        └─────────►│   Fonction 5   │
                          │                                   └────────────────┘
                          │
           ┌──────────────┴─┐
           │   Fonction 1   │
           └──────────────┬─┘
                          │
                          │                                   ┌────────────────┐
                          │                        ┌─────────►│   Fonction 6   │
                          │                        │          └────────────────┘
                          │         ┌──────────────┴─┐
                          └────────►│   Fonction 3   │
                                    └──────────────┬─┘
                                                   │          ┌────────────────┐
                                                   └─────────►│   Fonction 7   │
                                                              └────────────────┘

           ─────────────────────────────────────────────────────────────────────
                                  Chaînes de taille 3 max


Pour une taille de chaîne maximale donnée (noté n) il en découle une taille de contexte
minimale (nombre minimal de fonctions) : 2^n - 1

──►  Une taille de chaîne maximale de 7 donne un nombre de fonctions égal à 127.

Par ailleurs cela donne 2^(n-1) chaînes de taille maximale n, 2^(n-2) chaînes de taille n-1

──►  Dans l'exemple vu plus haut on a bien 2^2=4 chaînes de taille maximale (3) ce qui correspond
     au nombre de feuilles dans un arbre binaire complet

Cela a pour conséquence que la taille du contexte et le nombre de chaînes sont liés ou se bornent


Le raisonnement attendu pour un LLM doit-il prendre en compte le backtracking en détail ?

Exemple de raisonnement (relatif au dessin ci-dessus) :
Q: Does method `Fonction1` call method `Fonction 7`, either directly or indirectly?
A:
Let's think step by step:
1.1 Fonction1 calls Fonction2.
──► 2.1 Fonction2 calls Fonction4.
──────► 3.1 Fonction4 does not call anything.
──► 2.2 Fonction2 calls Fonction5.
──────► 3.2 Fonction5 does not call anything.
1.2 Fonction1 calls Fonction3.
──► 2.1 Fonction3 calls Fonction6.
──────► 3.1 Fonction4 does not call anything.
──► 2.2 Fonction3 calls Fonction7.
──────► 3.2 Fonction5 does not call anything.
Therefore, Fonction1 calls Fonction7 indirectly. FINAL ANSWER: YES
"""

""" 
The heuristic used to define the complexity of a chain between two given methods is the number of methods
we have to go through to reach the target method from the source method assuming that we do a depth-first search
Given this heuristic, for a given tree depth, we can automatically deduct the max length of a chain :
A n depth tree has 2^(n+1)-1 methods, and the max length of a chain is 2*number of methods - 2*2^n

Example : a complete binary tree of depth 3 has 15 methods, and the max length of a chain is 2*15 - 2*8 = 30 - 16 = 14

Note that we don't consider backtracking in this heuristic, but it might be relevant for the LLM reasoning process.
"""

def generate_single_call_tree(tree_depth:int):
    """Generate a list of method bodies that call each other in a tree-like structure.

    Args:
        method_names (list): A list of method names to be used in the tree structure.
    """
    n_methods = 2**(tree_depth + 1) - 1
    method_names = gen.generate_unique_method_names(n_methods)
    print(f"Generating a tree with {n_methods} methods and depth {tree_depth}")
    # print("Method names:", method_names)
    root = method_tree.build_binary_tree(tree_depth, method_names)
    return root

def generate_single_call_tree_from_names(tree_depth:int, method_names:list):
    """Generate a single call tree from a list of method names.

    Args:
        tree_depth (int): The depth of the tree to be generated.
        method_names (list): A list of method names to be used in the tree structure.
    """
    print(f"Generating a tree with {len(method_names)} methods and depth {tree_depth}")
    # print("Method names:", method_names)
    root = method_tree.build_binary_tree(tree_depth, method_names)
    return root

def generate_many_call_trees(dir:str, tree_depth:int, n_trees:int):
    """Generate a list of method bodies that call each other in a tree-like structure.

    Args:
        tree_depth (int): The depth of the tree to be generated.
        n_trees (int): The number of trees to generate.
    """
    method_names = gen.generate_unique_method_names(n_trees * (2**(tree_depth + 1) - 1))
    trees = []
    for i in range(n_trees):
        print(f"Generating tree {i+1}/{n_trees} with depth {tree_depth}")
        tree = generate_single_call_tree_from_names(tree_depth, method_names[i * (2**(tree_depth + 1) - 1):(i + 1) * (2**(tree_depth + 1) - 1)])
        trees.append(tree)
        
    for tree in trees:
        tree.write_tree_to_file(f"{dir}/tree_structures/tree_structure_{trees.index(tree)}.txt")
        
    return trees, method_names
    
def generate_tree_method_calls(trees:list):
    """Generate a list of Java method bodies that call each other in a tree like structure.
    Without comments.

    Args:
        tree (list): A list of trees describing the method calls.

    Returns:
        list : A list of strings, each representing a Java method body that calls the next methods in the tree-like chains.
    """
    method_bodies = []
    
    # Loop through each tree in the list of trees
    for tree in trees:
        method_bodies += generate_tree_method_calls_rec(tree)

    return method_bodies    


def generate_tree_method_calls_rec(tree:method_tree.Node):
    """Generate the methods from the tree structure recursively.

    Args:
        tree (Node): A Node object representing a node of the tree.
    """
    if tree is None:
        return []
    
    method_bodies = [generate_single_method_body(tree)]
    method_bodies += generate_tree_method_calls_rec(tree.left)
    method_bodies += generate_tree_method_calls_rec(tree.right)
    
    return method_bodies

def generate_single_method_body(subtree:method_tree.Node):
    if subtree.left is None and subtree.right is None:
        return f"    public void {subtree.name}() {{\n        // End of chain\n    }}"
    else: 
        return f"    public void {subtree.name}() {{\n        {subtree.left.name}();\n        {subtree.right.name}();\n    }}"


def generate_class_from_multiple_trees(directory:str, class_name:str, trees:list, method_names:list, selection:list):
    """Generate a class with methods that call each other in a tree-like structure.

    Args:
        directory (str): The directory where the generated files will be saved.
        tree_depth (int): The depth of the tree to be generated.
    """
    
    # print(f"Generating classes for trees :")
    # for tree in trees:
    #     tree.print_tree()
    
    method_bodies = generate_tree_method_calls(trees)
    
    print(f"Generated {len(method_bodies)} method bodies")
    
    # Shuffle the method bodies to create random order in the class
    random.shuffle(method_bodies)

    # Construct the class with shuffled method bodies
    class_body = f"public class {class_name} {{\n"
    class_body += "\n\n".join(method_bodies)
    class_body += "\n}"
    
    dir = Path(directory)
    dir.mkdir(parents=True, exist_ok=True)
    gen.write_class_to_file(class_body,  dir / "TheClass.java")
    gen.write_methods_to_file(method_names,  dir / "methods.txt")  
    write_chains_to_file(selection, dir / "chains.txt")
    gen.write_questions_to_file(selection, dir / "reachability_questions.txt")
    gen.write_prompt_to_file(p.in_context_tree_calls, class_body, dir / "system.txt")
    
def generate_exp(exp_name:str, n_trees:int, tree_depth:int, max_chain_length:int = None):
    """Generate an experiment with multiple trees and save the class to a file.

    Args:
        exp_name (str): The name of the experiment.
        n_trees (int): The number of trees to generate.
        tree_depth (int): The depth of each tree.
    """
    print(f"Generating {n_trees} trees with depth {tree_depth} for experiment {exp_name}")
    trees, method_names = generate_many_call_trees(exp_name, tree_depth, n_trees)
    print(f"Generated {len(trees)} trees")
    method_tree.depth_first_traversal(trees[0])
    chains, questions = find_all_valid_chains(trees)
    selection = []
    
    # The maximum length of a chain is deduced from the tree depth
    if max_chain_length is None:
        max_chain_length = (2**(tree_depth + 1) - 1)
    
    for depth in range(2, max_chain_length + 1):
        print(f"Generating questions for chains of depth {depth}")
        # TODO : choose a better number of questions to select (e.g. 100 is kind of arbitrary) 
        selection.extend(gen.select_n_of_distance(questions, depth, 100))
        # TODO : handle negative depths 
        # selection.extend(gen.select_n_of_distance(questions, -depth, 100))
    print(f"Selected {len(selection)} questions")
    print(f"Questions per distance after selection: {gen.count_distances(selection)}")
    
    generate_class_from_multiple_trees(directory=exp_name, class_name="TheClass", trees=trees, method_names=method_names, selection=selection)
    
def find_all_valid_chains(trees:list):
    """Find all chains in the trees with a maximum length.

    Args:
        trees (list): A list of trees to search for chains.
        max_chain_length (int): The maximum length of the chains to find.
    """
    all_chains = []
    for tree in trees:
        chains = method_tree.find_all_valid_chains_depth_first(tree)
        all_chains.extend(chains)
        print(f"Found {len(chains)} chains in tree {trees.index(tree) + 1}")
        # print(f"Chains found in tree {trees.index(tree) + 1}: {chains}")
    
    print(f"Total chains found: {len(all_chains)}")
    
    questions_with_distances_and_chains = generate_questions_from_valid_chains(all_chains)
    # print(f"Questions generated from valid chains: {questions_with_distances_and_chains}")
    
    chain_distances = gen.count_distances(questions_with_distances_and_chains)
    print(f"Chain distances: {chain_distances}")
    
    return all_chains, questions_with_distances_and_chains

def generate_questions_from_valid_chains(chains:list, max_chain_length:int = None):
    """Generate questions from valid chains.

    Args:
        chains (list): A list of valid chains to generate questions from.
        max_chain_length (int): The maximum length of the chains to consider.
    """
    questions_with_distances_and_chains = []
    for chain in chains:
        if max_chain_length is None or len(chain) <= max_chain_length:
            question = (
                f"Does `{chain[0]}` call `{chain[-1]}`, either directly or indirectly? "
                f"Think step-by-step by following the method calls from `{chain[0]}`."
            )
            
            distance = len(chain)
            
            questions_with_distances_and_chains.append((question, distance, chain))
    return questions_with_distances_and_chains

def write_chains_to_file(questions:list, filename:str):
    """Write all chains from the questions list to a file, one per line."""
    with open(filename, 'w') as file:
        for question, dist, chain in questions:
            file.write(" ".join(chain) + '\n')  # Write each chain on a new line

# Generate an experiment with 3 trees of depth 3 ==> 15 methods each, 45 methods in total
generate_exp(exp_name="tree_exp", n_trees=3, tree_depth=3)
# Generate a larger experiment with 3 trees of depth 6 ==> 127 methods each, 381 methods in total
# generate_exp(exp_name="large_tree_exp", n_trees=3, tree_depth=6)