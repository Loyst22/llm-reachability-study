from pathlib import Path
import random
import generate_chain as gen
import method_tree
import prompts
import comments_generation
import control_flow
from experiment_config import TreeCallExperimentConfig

"""            
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
                                  Chaînes de taille 2 max


Pour une taille de chaîne maximale donnée (noté n) il en découle une taille de contexte
minimale (nombre minimal de fonctions) : 2^{n+1} - 1

──►  Une taille de chaîne maximale de 6 donne un nombre de fonctions égal à 127.

Par ailleurs cela donne 2^n chaînes de taille maximale n 

──►  Dans l'exemple vu plus haut on a bien 2^2=4 chaînes de taille maximale (2) ce qui correspond
     au nombre de feuilles dans un arbre binaire complet

Cela a pour conséquence que la taille du contexte et la taille des chaînes sont liées ou se bornent


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
    root = method_tree.build_binary_tree(tree_depth, method_names)
    return root

def generate_single_call_tree_from_names(tree_depth:int, method_names:list):
    """Generate a single call tree from a list of method names.

    Args:
        tree_depth (int): The depth of the tree to be generated.
        method_names (list): A list of method names to be used in the tree structure.
    """
    # print(f"Generating a tree with {len(method_names)} methods and depth {tree_depth}")
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
        tree, _ = generate_single_call_tree_from_names(tree_depth, method_names[i * (2**(tree_depth + 1) - 1):(i + 1) * (2**(tree_depth + 1) - 1)])
        trees.append(tree)
        
    method_tree.write_trees_to_files(trees, dir)
        
    return trees, method_names

def generate_many_call_trees_v2(dir: str, config: TreeCallExperimentConfig):
    """Generate a list of method bodies that call each other in a tree-like structure.

    Args:
        tree_depth (int): The depth of the tree to be generated.
        n_trees (int): The number of trees to generate.
    """
    method_names = gen.generate_unique_method_names(config.context_size)
        
    max_chain_length = max(config.depths)
    depth = max_chain_length//2 + 2
    
    trees = []
    cmpt = 0
    while method_names:
        if cmpt % 4 == 0:
            root = method_tree.build_comb_tree(depth, max_chain_length, method_names, n_params=config.n_params, n_vars=config.n_vars)
            if root:
                trees.append(root)
        elif cmpt % 4 == 1:
            root = method_tree.build_near_comb_tree(depth, max_chain_length, method_names, n_params=config.n_params, n_vars=config.n_vars)
            if root:
                trees.append(root)
        elif cmpt % 4 == 2:
            root = method_tree.build_unbalanced_binary_tree(max_chain_length, method_names, n_params=config.n_params, n_vars=config.n_vars)
            if root:
                trees.append(root)
        else:
            root = method_tree.build_binary_tree(3, method_names, n_params=config.n_params, n_vars=config.n_vars)
            trees.append(root)
        cmpt += 1
        
    # Verification of unicity of method names across trees:
    all_method_names = []

    for root_it in trees:
        all_method_names.extend(root_it.get_method_names())
    
    if not len(all_method_names) == len(set(all_method_names)):
        raise ValueError(f"Method names not unique across trees for dir {dir}")
        
    method_tree.write_trees_to_files(trees, dir)
        
    return trees, all_method_names

def generate_many_call_trees_v3(dir: str, config: TreeCallExperimentConfig):
    """Generate a list of method bodies that call each other in a tree-like structure.

    Args:
        tree_depth (int): The depth of the tree to be generated.
        n_trees (int): The number of trees to generate.
    """
    method_names = gen.generate_unique_method_names(config.context_size)
    
    max_chain_length = max(config.depths)
    comb_depth = max_chain_length//2 + 2
    size_of_comb = 2*comb_depth + 1
    size_of_four_combs = 8*comb_depth + 2
    size_of_jellyfish = size_of_four_combs + 3
    size_of_double_comb = size_of_four_combs/2 + 1
    max_k_depth = 4 # TODO: Maybe find a better one (that depends on the rest)
    shape = ["left", "right"]
    
    trees = []
    while method_names:
        if len(method_names) >= size_of_jellyfish:
            remaining_size = len(method_names) - size_of_four_combs
            k_depth = 1
            k_size = 2**(k_depth+1) - 1
            while k_size <= remaining_size:
                k_depth += 1
                k_size = 2**(k_depth+1) - 1
                if k_depth > max_k_depth:
                    k_depth = max_k_depth + 1
                    break # Stop the inner while
            k_depth -= 1 
            print("Generating Jellyfish tree")
            root = method_tree.build_jellyfish_tree(k_depth, 
                                                    k_depth, 
                                                    max_chain_length, 
                                                    method_names, 
                                                    n_params=config.n_params, 
                                                    n_vars=config.n_vars, 
                                                    shape=shape)
            
        elif len(method_names) >= size_of_double_comb:
            print("Generating Double-Comb tree")
            root = method_tree.build_double_comb(max_chain_length, 
                                                 method_names,
                                                 n_params=config.n_params,
                                                 n_vars=config.n_vars, 
                                                 shape=shape)
            
        elif len(method_names) >= size_of_comb:
            if random.random() > 0.5:
                print("Generating Comb tree")
                root = method_tree.build_comb_tree(comb_depth, 
                                                   max_chain_length, 
                                                   method_names,
                                                   n_params=config.n_params,
                                                   n_vars=config.n_vars, 
                                                   shape=shape)
                
            else:
                print("Generating Near-Comb tree")
                root = method_tree.build_near_comb_tree(comb_depth, 
                                                        max_chain_length,
                                                        method_names,
                                                        n_params=config.n_params,
                                                        n_vars=config.n_vars, 
                                                        shape=shape)
                
        else:
            remaining_size = len(method_names)
            k_depth = 0
            k_size = 2**(k_depth+1) - 1
            while k_size <= remaining_size:
                k_depth += 1
                k_size = 2**(k_depth+1) - 1
            # k_depth -= 1
            print("Generating Binary tree")
            root = method_tree.build_binary_tree(k_depth, 
                                                 method_names, 
                                                 n_params=config.n_params, 
                                                 n_vars=config.n_vars)
                        
        if root:
            trees.append(root)
        shape = shape[::-1]
    
        print(f"Size of tree generated is: {root.get_subtree_size()}")
        
    # Verification of unicity of method names across trees:
    all_method_names = []

    for root_it in trees:
        all_method_names.extend(root_it.get_method_names())
    
    if not len(all_method_names) == len(set(all_method_names)):
        raise ValueError(f"Method names not unique across trees for dir {dir}")
        
    method_tree.write_trees_to_files(trees, dir)
        
    return trees, all_method_names
    
def generate_tree_method_calls(trees:list, config: TreeCallExperimentConfig = None):
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
        method_bodies += generate_tree_method_calls_rec(tree, config)

    return method_bodies    


def generate_tree_method_calls_rec(tree: method_tree.Node, config: TreeCallExperimentConfig = None):
    """Generate the methods from the tree structure recursively.

    Args:
        tree (Node): A Node object representing a node of the tree.
    """
    if tree is None:
        return []
    
    method_bodies = [generate_single_method_body(tree, config)]
    method_bodies += generate_tree_method_calls_rec(tree.left, config)
    method_bodies += generate_tree_method_calls_rec(tree.right, config)
    
    return method_bodies

def generate_single_method_body(node: method_tree.Node, config: TreeCallExperimentConfig = None):
    param_string = ", ".join([f"{var.var_type} {var.name}" for var in (node.params or [])])
    if config is None: 
        if node.left is None and node.right is None:
            method_body = f"\tpublic void {node.name}({param_string}) {{\n\t\t// End of chain\n\t}}"
        else:
            method_body = f"\tpublic void {node.name}({param_string}) {{\n\t\t{node.left.name}();\n\t\t{node.right.name}();\n\t}}"
        return f"\tpublic void {node.name}() {{\n{method_body}\n\t}}"

    if config.language.lower() != "java":
        raise ValueError("Tree call experiments only supports Java language")
    
    comment = comments_generation.generate_lorem_ipsum_comments(config.n_comment_lines, config.language)
    
    next_methods = []
    
    if node.left is not None:
        var_types = [var.var_type for var in node.left.params or []]
        call_params = control_flow.choose_n_vars_from_types(var_types, node.all_variables)
        next_methods.append(f"{node.left.name}({', '.join([var.name for var in call_params])});")
    if node.right is not None:
        var_types = [var.var_type for var in node.right.params or []]
        call_params = control_flow.choose_n_vars_from_types(var_types, node.all_variables)
        next_methods.append(f"{node.right.name}({', '.join([var.name for var in call_params])});")
    
    if len(next_methods) == 0:
        next_methods = None
    
    method_body = control_flow.generate_method_body(next_methods=next_methods,
                                                   vars=node.variables,
                                                   all_vars=node.all_variables,
                                                   n_loops=config.n_loops,
                                                   n_if=config.n_if)

    comment = "\t" + comment.replace("\n", "\n\t")
    method_body = "\t" + method_body.replace("\n", "\n\t")
    return f"{comment}\n\tpublic void {node.name}({param_string}) {{\n{method_body}\n\t}}"


def generate_class_from_multiple_trees(directory:str, class_name:str, trees:list, method_names:list, selection:list):
    """Generate a class with methods that call each other in a tree-like structure.

    Args:
        directory (str): The directory where the generated files will be saved.
        tree_depth (int): The depth of the tree to be generated.
    """
    
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
    gen.write_prompt_to_file(prompts.in_context_tree_calls, class_body, dir / "system.txt")
    
def generate_exp(exp_name:str, n_trees:int, tree_depth:int, max_chain_length:int = None, n_questions:int = 400) -> int:
    """Generate an experiment with multiple trees and save the class to a file.

    Args:
        exp_name (str): The name of the experiment.
        n_trees (int): The number of trees to generate.
        tree_depth (int): The depth of each tree.
    """
    print(f"Generating {n_trees} trees with depth {tree_depth} for experiment {exp_name}")
    trees, method_names = generate_many_call_trees(exp_name, tree_depth, n_trees)
    print(f"Generated {len(trees)} trees")
    valid_questions = find_all_valid_chains(trees=trees)
    invalid_questions = find_all_invalid_chains(trees=trees)
    
    
    # The maximum length of a chain is deduced from the tree depth
    if max_chain_length is None:
        max_chain_length = (2**(tree_depth + 1) - 1)
        
    selection = []
                
    for depth in range(max_chain_length + 1):
        selection.extend(gen.select_n_of_distance(valid_questions, depth, n_questions))
        selection.extend(gen.select_n_of_distance(invalid_questions, -depth, n_questions))
        
    distance_dict = gen.count_distances(selection)
    
    min_amount_of_questions = n_questions
    
    for value in distance_dict.values():
        if value < min_amount_of_questions:
            min_amount_of_questions = value
            
    selection = []
    
    for depth in range(max_chain_length + 1):
        selection.extend(gen.select_n_of_distance(valid_questions, depth, min_amount_of_questions))
        selection.extend(gen.select_n_of_distance(invalid_questions, -depth, min_amount_of_questions))
    
    print(f"Selected {len(selection)} questions")
    print(f"Questions per distance after selection: {gen.count_distances(selection)}")
    
    generate_class_from_multiple_trees(directory=exp_name, class_name="TheClass", trees=trees, method_names=method_names, selection=selection)
    
    return min_amount_of_questions
    
def find_all_valid_chains(trees:list) -> tuple[list, list]:
    """Find all chains in the trees with a maximum length.

    Args:
        trees (list): A list of trees to search for chains.
    """
    all_valid_chains = []
    for tree in trees:
        chains = method_tree.find_all_valid_chains_depth_first(tree)
        all_valid_chains.extend(chains)
    
    generate_questions_from_valid_chains(all_valid_chains)
    
    # chain_distances = gen.count_distances(all_valid_chains)
    # print(f"Chain distances: {chain_distances}")
    
    return all_valid_chains

# ! We should not try to find ALL invalid chains, as it would be too costly in terms of time and resources.
def find_all_invalid_chains(trees:list) -> tuple[list, list]:
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
    
    all_invalid_chains = generate_questions_from_invalid_chains(all_invalid_chains)
    
    # chain_distances = gen.count_distances(all_invalid_chains)
    # print(f"Chain distances: {chain_distances}")
    
    return all_invalid_chains

def generate_questions_from_valid_chains(chains:list, max_chain_length:int = None):
    """Generate questions from valid chains.

    Args:
        chains (list): A list of valid chains to generate questions from.
        max_chain_length (int): The maximum length of the chains to consider.
    """
    for item in chains:
        if max_chain_length is None or item["distance"] <= max_chain_length:
            chain = item["chain"]
            question = (
                f"Does `{chain[0]}` call `{chain[-1]}`, either directly or indirectly? "
                f"Think step-by-step by following the method calls from `{chain[0]}`."
            )
            item["question"] = question          
            
    return

def generate_questions_from_invalid_chains(chains:list, max_chain_length:int = None) -> list:
    """Generate questions from invalid chains.

    Args:
        chains (list): A list of invalid chains to generate questions from.
        max_chain_length (int): The maximum length of the chains to consider.
    """
    generated_questions = []
    
    for item in chains:
        for unreachable in item["unreachable_methods"]:
            if max_chain_length is None or len(item["chain"]) <= max_chain_length:
                # Make a copy of the item
                new_item = item.copy()
                new_item.pop("unreachable_methods", None)  # Remove the field if it exists
                
                question = (
                    f"Does `{item['node']}` call `{unreachable}`, either directly or indirectly? "
                    f"Think step-by-step by following the method calls from `{item['node']}`."
                )
                
                new_item["question"] = question 
                new_item["target_method"] = unreachable
                
                generated_questions.append(new_item)

    return generated_questions

def write_chains_to_file(questions:list, filename:str):
    """Write all chains from the questions list to a file, one per line."""
    with open(filename, 'w') as file:
        for question, dist, chain in questions:
            file.write(" ".join(chain) + '\n')  # Write each chain on a new line

if __name__ == '__main__':

    # Generate an experiment with 3 trees of depth 3 ==> 15 methods each, 45 methods in total
    # generate_exp(exp_name="experiments/tree_exp_2", n_trees=3, tree_depth=3, n_questions=2)
    # Generate a larger experiment with 3 trees of depth 6 ==> 127 methods each, 381 methods in total
    # generate_exp(exp_name="experiments/large_tree_exp", n_trees=3, tree_depth=6, n_questions=2)
    
    generate_exp("validation_code", 1, 2)