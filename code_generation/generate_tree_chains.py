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
    # print(f"Generating a tree with {n_methods} methods and depth {tree_depth}")
    # print("Method names:", method_names)
    root = method_tree.build_binary_tree(tree_depth, method_names)
    return root

def generate_single_call_tree_from_names(tree_depth:int, method_names:list):
    """Generate a single call tree from a list of method names.

    Args:
        tree_depth (int): The depth of the tree to be generated.
        method_names (list): A list of method names to be used in the tree structure.
    """
    # print(f"Generating a tree with {len(method_names)} methods and depth {tree_depth}")
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
        # print(f"Generating tree {i+1}/{n_trees} with depth {tree_depth}")
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
    
def generate_exp(exp_name:str, n_trees:int, tree_depth:int, max_chain_length:int = None, n_questions:int = 100) -> int:
    """Generate an experiment with multiple trees and save the class to a file.

    Args:
        exp_name (str): The name of the experiment.
        n_trees (int): The number of trees to generate.
        tree_depth (int): The depth of each tree.
    """
    print(f"Generating {n_trees} trees with depth {tree_depth} for experiment {exp_name}")
    trees, method_names = generate_many_call_trees(exp_name, tree_depth, n_trees)
    print(f"Generated {len(trees)} trees")
    _, valid_questions = find_all_valid_chains(trees=trees)
    _, invalid_questions = find_all_invalid_chains(trees=trees)
    
    selection = []
    
    # The maximum length of a chain is deduced from the tree depth
    if max_chain_length is None:
        max_chain_length = (2**(tree_depth + 1) - 1)
                
    for depth in range(max_chain_length + 1):
        # TODO : choose a better number of questions to select (e.g. 100 is kind of arbitrary) 
        selection.extend(gen.select_n_of_distance(valid_questions, depth, n_questions))
        # TODO : see if we can manage to get negative questions for all distances
        selection.extend(gen.select_n_of_distance(invalid_questions, -depth, n_questions))
        
    distance_dict = gen.count_distances(selection)
    
    min_amount_of_questions = n_questions
    
    for value in distance_dict.values():
        if value < min_amount_of_questions:
            min_amount_of_questions = value
            
    selection = []
    
    for depth in range(max_chain_length + 1):
        # TODO : choose a better number of questions to select (e.g. 100 is kind of arbitrary) 
        selection.extend(gen.select_n_of_distance(valid_questions, depth, min_amount_of_questions))
        # TODO : see if we can manage to get negative questions for all distances
        selection.extend(gen.select_n_of_distance(invalid_questions, -depth, min_amount_of_questions))
    
    print(f"Selected {len(selection)} questions")
    print(f"Questions per distance after selection: {gen.count_distances(selection)}")
    
    generate_class_from_multiple_trees(directory=exp_name, class_name="TheClass", trees=trees, method_names=method_names, selection=selection)
    
    return min_amount_of_questions
    
def find_all_valid_chains(trees:list):
    """Find all chains in the trees with a maximum length.

    Args:
        trees (list): A list of trees to search for chains.
    """
    all_valid_chains = []
    for tree in trees:
        chains = method_tree.find_all_valid_chains_depth_first(tree)
        all_valid_chains.extend(chains)
        # print(f"Found {len(chains)} valid chains in tree {trees.index(tree) + 1}")
        # print(f"Chains found in tree {trees.index(tree) + 1}: {chains}")
    
    print(f"Total valid chains found: {len(all_valid_chains)}")
    
    questions_with_distances_and_chains = generate_questions_from_valid_chains(all_valid_chains)
    # print(f"Questions generated from valid chains: {questions_with_distances_and_chains}")
    
    chain_distances = gen.count_distances(questions_with_distances_and_chains)
    print(f"Chain distances: {chain_distances}")
    
    return all_valid_chains, questions_with_distances_and_chains

# ! We should not try to find ALL invalid chains, as it would be too costly in terms of time and resources.
def find_all_invalid_chains(trees:list):
    """Find all invalid chains in the trees.

    Args:
        trees (list): A list of trees to search for invalid chains.
    """
    # There are two ways to find invalid chains:
    # 1. Find the invalid chains within a single tree (when a method is not reachable from another one even though they are on the same tree)
    # 2. Find the invalid chains across all trees (when a method is not reachable from another one because they are on different trees)
    all_invalid_chains = []
    for tree in trees:
        invalid_chains = method_tree.find_all_invalid_chains_depth_first(tree)
        all_invalid_chains.extend(invalid_chains)
        # print(f"Found {len(invalid_chains)} invalid chains in tree {trees.index(tree) + 1}")
        # print(f"Invalid chains found in tree {trees.index(tree) + 1}: {invalid_chains}")
    
    print(f"Total invalid chains found: {len(all_invalid_chains)}")
    
    # print(f"Invalid chains: {all_invalid_chains}")
    
    questions_with_distances_and_chains = generate_questions_from_invalid_chains(all_invalid_chains)
    
    chain_distances = gen.count_distances(questions_with_distances_and_chains)
    print(f"Chain distances: {chain_distances}")
    
    return all_invalid_chains, questions_with_distances_and_chains

def generate_questions_from_valid_chains(chains:list, max_chain_length:int = None):
    """Generate questions from valid chains.

    Args:
        chains (list): A list of valid chains to generate questions from.
        max_chain_length (int): The maximum length of the chains to consider.
    """
    questions_with_distances_and_chains = []
    for item in chains:
        if max_chain_length is None or item["distance"] <= max_chain_length:
            chain = item["chain"]
            question = (
                f"Does `{chain[0]}` call `{chain[-1]}`, either directly or indirectly? "
                f"Think step-by-step by following the method calls from `{chain[0]}`."
            )
            
            distance = item["distance"]
            
            questions_with_distances_and_chains.append((question, distance, chain))
    return questions_with_distances_and_chains

def generate_questions_from_invalid_chains(chains:list, max_chain_length:int = None):
    """Generate questions from invalid chains.

    Args:
        chains (list): A list of invalid chains to generate questions from.
        max_chain_length (int): The maximum length of the chains to consider.
    """
    questions_with_distances_and_chains = []
    
    # TODO : à vérifier
    for item in chains:
        for unreachable in item["unreachable_methods"]:
            if max_chain_length is None or len(item["chain"]) <= max_chain_length:
                question = (
                    f"Does `{item['node']}` call `{unreachable}`, either directly or indirectly? "
                    f"Think step-by-step by following the method calls from `{item['node']}`."
                )
                distance = item["distance"]
                chain = item["chain"]

                questions_with_distances_and_chains.append((question, distance, chain))
        
    return questions_with_distances_and_chains

def write_chains_to_file(questions:list, filename:str):
    """Write all chains from the questions list to a file, one per line."""
    with open(filename, 'w') as file:
        for question, dist, chain in questions:
            file.write(" ".join(chain) + '\n')  # Write each chain on a new line

# Generate an experiment with 3 trees of depth 3 ==> 15 methods each, 45 methods in total
# generate_exp(exp_name="tree_exp_2", n_trees=3, tree_depth=3, n_questions=2)
# Generate a larger experiment with 3 trees of depth 6 ==> 127 methods each, 381 methods in total
# generate_exp(exp_name="experiments/large_tree_exp", n_trees=3, tree_depth=6, n_questions=2)