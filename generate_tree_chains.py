from pathlib import Path
import random
import generate_chain as gen
import method_tree

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
    print("Method names:", method_names)
    root = method_tree.build_binary_tree(tree_depth, method_names)
    return root

def generate_single_call_tree_from_names(tree_depth:int, method_names:list):
    """Generate a single call tree from a list of method names.

    Args:
        tree_depth (int): The depth of the tree to be generated.
        method_names (list): A list of method names to be used in the tree structure.
    """
    print(f"Generating a tree with {len(method_names)} methods and depth {tree_depth}")
    print("Method names:", method_names)
    root = method_tree.build_binary_tree(tree_depth, method_names)
    return root

def generate_many_call_trees(tree_depth:int, n_trees:int):
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
        tree.write_tree_to_file(f"tree_exp/tree_structure_{trees.index(tree)}.txt")
        
    return trees
    
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


def generate_class_from_multiple_trees(directory:str, class_name:str, trees:list):
    """Generate a class with methods that call each other in a tree-like structure.

    Args:
        directory (str): The directory where the generated files will be saved.
        tree_depth (int): The depth of the tree to be generated.
    """
    print(f"Generating classes for trees :")
    for tree in trees:
        tree.print_tree()
    
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
    # write_prompt_to_file(p.in_context, the_class, dir / "system.txt")
    # write_questions_to_file(selection, dir / "reachability_questions.txt")
    # write_chains_to_file(selection, dir / "chains.txt")
    # write_methods_to_file(method_names,  dir / "methods.txt")  
    
def generate_exp(exp_name:str, n_trees:int, tree_depth:int):
    """Generate an experiment with multiple trees and save the class to a file.

    Args:
        exp_name (str): The name of the experiment.
        n_trees (int): The number of trees to generate.
        tree_depth (int): The depth of each tree.
    """
    print(f"Generating {n_trees} trees with depth {tree_depth} for experiment {exp_name}")
    trees = generate_many_call_trees(tree_depth, n_trees)
    print(f"Generated {len(trees)} trees")
    generate_class_from_multiple_trees(exp_name, "TheClass", trees)
    
# tree = generate_single_call_tree(5)  # Generate a tree with depth 5
# generate_class_from_multiple_trees("tree_exp", "TheClass", [tree])
# tree.write_tree_to_file("tree_exp/tree_structure.txt")
    
generate_exp("tree_exp", 3, 5)  # Generate an experiment with 3 trees of depth 5
