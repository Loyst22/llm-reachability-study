import copy
from pathlib import Path
import random
import control_flow

class Node:
    """
    A class representing a node in a binary tree.
    Each node actually represents a function.
    
    Attributes:
        name (str): Name of the function/method.
        parent (Node): Parent node in the tree.
        left (Node): Left child node.
        right (Node): Right child node.
        params (list[Variable]): List of parameters for the method.
        variables (list[Variable]): List of variables defined in the body of the method.
        all_variables (list[Variable]): List of variables defined/used in the method.
        var_types (list[str]): List of variable types used in the method. (params + local variables)
        return_variable (Variable): Variable to return at the end of the method.
        return_type (str): Return type of the method.
    """
    def __init__(self, 
                 name: str,
                 n_params: int=0, 
                 n_vars: int=0, 
                 return_value: bool=False, # ! TO CHANGE back to False if set to True 
                 path: list[str]=None, 
                 parent=None, 
                 left=None, 
                 right=None):
        # Name
        self.name = name
        
        # Links
        self.parent = parent
        self.left = left
        self.right = right
        
        # Path (very optional)
        if path is None:
            path = []
        self.path = path.copy()
            
        # Params
        # ! Be careful with building that at init, parents might get added later
        # ! atm there is no issue with that, but it might break in the future
        if parent is None:
            # We can't have parameters if it's the root
            self.params = []
            # But we have to have at least the same amount of variable as of parameters if the other methods have parameters
            if n_vars < n_params:
                n_vars = n_params
                
            # self.params = control_flow.random_variables(n_params)
        else:
            self.params = control_flow.choose_n_vars(n_params, parent.all_variables)
            control_flow.rename_vars(self.params)
        
        # Get names of all param variables to avoid duplication
        param_names = {var.name for var in self.params}
        
        # Variables
        self.variables = []
        
        while len(self.variables) < n_vars:
            additional_vars = control_flow.random_variables(n_vars - len(self.variables))
            self.variables.extend([var for var in additional_vars if var.name not in param_names and var.name not in {v.name for v in self.variables}])
        
        self.all_variables = self.variables + self.params # Add parameters to the list of variables
        self.var_types = [var.var_type for var in self.all_variables]
        
        # Return
        tmp_return = control_flow.choose_n_vars(1, self.all_variables)
        if tmp_return and return_value:
            self.return_variable = tmp_return[0]
            self.return_type = self.return_variable.var_type
        else:
            self.return_variable = None
            self.return_type = "void"
        
            
    def __str__(self):
        parent_name = self.parent.name if self.parent else "None"
        left_name = self.left.name if self.left else "None"
        right_name = self.right.name if self.right else "None"
        param_types = [var.var_type for var in self.params]

        return (
            f"Node(name={self.name}, "
            f"parent={parent_name}, "
            f"left={left_name}, "
            f"right={right_name}, "
            f"path={self.path}, "
            f"variable_types={self.var_types},"
            f"param_types={param_types}, "
            f"variables={self.variables},"
            f"return_type={self.return_type},"
            f"return_variable={self.return_variable})"
        )

    def print_tree(self, indent: str = ""):
        """Prints the subtree to the standard output"""
        if self is None:
            return
        print(indent + self.name)
        if self.left:
            self.left.print_tree(indent + "   L- ")
        if self.right:
            self.right.print_tree(indent + "   R- ")

    def write_tree_to_file(self, file_path: str):
        """Write the subtree to a file"""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        """Write the tree structure to a file in a readable format."""
        def write_node(node, indent=""):
            if node is None:
                return
            f.write(indent + node.name + "\n")
            write_node(node.left, indent + "   L- ")
            write_node(node.right, indent + "   R- ")
            
        with open(file_path, 'w') as f:
            write_node(self)
    
    def get_height(self, height: int = 0) -> int:
        """Get the height of the node"""
        if self.parent is None:
            return height
        return self.parent.get_height(height=height+1)
    
    def get_relative_height(self, relative_parent: "Node", height: int = 0) -> int:
        """Get the height-distance of the node from its relative parent"""
        if self == relative_parent:
            return height
        return self.parent.get_relative_height(relative_parent, height+1)
            
    def get_subtree_size(self):
        """Get the size of the subtree"""
        size = 1
        if self.left is not None:
            size += self.left.get_subtree_size()
        if self.right is not None:
            size += self.right.get_subtree_size()
        return size
    
    def get_list_of_nodes(self):
        """Get the list of nodes of the subtree"""
        nodes = [self]
        if self.left is not None:
            nodes.extend(self.left.get_list_of_nodes())
        if self.right is not None:
            nodes.extend(self.right.get_list_of_nodes())
        return nodes
    
    def get_method_names(self):
        """Get the list of method names that appear in the subtree"""
        methods_names, *rest = depth_first_traversal(self)
        return methods_names
    
    def get_number_of_variables(self):
        """Get the number of variables defined/used in the method."""
        return len(self.all_variables)
                    
def write_trees_to_files(trees: list, dir: str):
    """Write individual tree files and a cumulative file."""
    Path(f"{dir}/tree_structures").mkdir(parents=True, exist_ok=True)
    cumulative_file = f"{dir}/tree_structures/all_tree_structures_0--{len(trees)-1}.txt"

    with open(cumulative_file, 'w') as file:
        for idx, tree in enumerate(trees):
            individual_path = f"{dir}/tree_structures/tree_structure_{idx}.txt"
            tree.write_tree_to_file(individual_path)

            file.write(f"\n{'='*20} Tree {idx} {'='*20}\n\n")
            with open(individual_path, 'r') as tree_f:
                file.write(tree_f.read())

""" Tree generation functions """

def build_binary_tree(depth: int, method_names: list[str], direction: str=None, parent: Node = None, n_params: int=0, n_vars: int=0
                      ) -> Node:
    """Build a binary tree from a list of method names.

    Args:
        depth (int): Depth of the tree.
        method_names (list): A list of method names to be used as node names in the tree.
        direction (int, optional): Used to track the path to the current node in the tree.
        parent (Node, optional): The parent node of the current node. Defaults to None.
        n_params (int, optional): Number of parameters per function. Defaults to 0.
        n_vars (int, optional): Number of variables per function. Defaults to 0.
        
    Returns:
        Node: The root of the binary tree.
    """
    if depth < 0 or not method_names:
        return None
    
    if parent: 
        path = parent.path.copy()
        path.append(direction)
    else:
        path = []

    # Pop the current method name
    node = Node(name=method_names.pop(0), n_params=n_params, n_vars=n_vars, path=path, parent=parent)

    # Build left and right subtrees, consuming names in-place
    node.left = build_binary_tree(depth - 1, method_names, "left", node, n_params, n_vars)
    node.right = build_binary_tree(depth - 1, method_names, "right", node, n_params, n_vars)

    return node

def build_jellyfish_tree(k_depth: int, 
                         k_depth_iter: int,
                         max_distance: int,
                         method_names: list,
                         parent: Node = None,
                         direction: str=None, 
                         n_params: int=0,
                         n_vars: int=0,
                         shape: list=["left", "right"]
                         ) -> Node:
    """Build a jellyfish tree from a list of method names.

    Args:
        k_depth (int): Depth of the complete part of the tree.
        k_depth_iter (int): Depth of the complete part of the tree (used to remember full depth).
        max_distance (int): The maximum distance for which we wish to ask question.
        method_names (list): A list of method names to be used as node names in the tree.
        direction (int, optional): Used to track the path to the current node in the tree.
        parent (Node, optional): The parent node of the current node. Defaults to None.
        n_params (int, optional): Number of parameters per function. Defaults to 0.
        n_vars (int, optional): Number of variables per function. Defaults to 0.
        shape (list, optional): Describes the shape of the comb parts of the tree. Defaults to ["left", "right"].

    Returns:
        Node: The root of the binary tree
    """
    comb_depth = max_distance//2 + 2
    
    # First we build the base of the jellyfish (ie. a complete balanced binary tree)
    root = build_binary_tree(depth=k_depth, method_names=method_names, n_params=n_params, n_vars=n_vars)
    
    # Then we build the comb parts of the tree
    current_node = root
    # Iterate to find the left most node
    while current_node.path.count("left") < k_depth and current_node.left is not None:
        current_node = current_node.left
    if current_node.path.count("left") == k_depth:
        setattr(current_node, shape[0], build_comb_tree(comb_depth, max_distance, method_names, n_params, n_vars, current_node, shape))
        setattr(current_node, shape[1], build_near_comb_tree(comb_depth, max_distance, method_names, n_params, n_vars, current_node, shape))
        
    current_node = root
    
    shape = shape[::-1]
    
    # Iterate so that we find the right most node
    while current_node.path.count("right") < k_depth and current_node.right is not None:
        current_node = current_node.right
        
    if current_node.path.count("right") == k_depth:
        setattr(current_node, shape[0], build_comb_tree(comb_depth, max_distance, method_names, n_params, n_vars, current_node, shape))
        setattr(current_node, shape[1], build_near_comb_tree(comb_depth, max_distance, method_names, n_params, n_vars, current_node, shape))
    
    return root

    while False:
        
        # ! Solution here ? 
        if k_depth_iter < 0 or not method_names:
            return None

        # Build the path list if it's not the root
        if parent: 
            path = parent.path.copy()
            path.append(direction)
        else:
            path = []
        # Create the new node
        node = Node(method_names.pop(0), n_params, n_vars, path, parent)
        
        # If it's the extreme left/right bottom node of the complete part of the tree
        # We insert the comb parts of the tree
        if node.path.count("left") == k_depth-1 and node.path.count("right") == 0:
            print(f"\nHere, left comb\n")
            node.left = build_comb_tree(comb_depth, max_distance, method_names, n_params, n_vars, node, shape)
            node.right = build_jellyfish_tree(k_depth, k_depth_iter-1, max_distance, method_names, node, "right", n_params, n_vars, shape)
            return node
        elif node.path.count("right") == k_depth-1 and node.path.count("left") == 0:
            print(f"\nHere, right comb\n")
            node.left = build_jellyfish_tree(k_depth, k_depth_iter-1, max_distance, method_names, node, "left", n_params, n_vars, shape)
            node.right = build_comb_tree(comb_depth, max_distance, method_names, n_params, n_vars, node, shape[::-1])
            return node
        else:
            # Build left and right subtrees for regular nodes
            node.left = build_jellyfish_tree(k_depth, k_depth_iter-1, max_distance, method_names, node, "left", n_params, n_vars, shape)
            node.right = build_jellyfish_tree(k_depth, k_depth_iter-1, max_distance, method_names, node, "right", n_params, n_vars, shape)

            return node

def build_double_comb(max_distance: int,
                      method_names: list,
                      n_params: int = 0,
                      n_vars: int = 0,
                      parent: Node = None,
                      shape: list[str] = ["left", "right"]):
    """Build an double comb binary tree from a list of method names.

    Args:
        max_distance (int): The maximum distance for which we wish to ask negative questions.
        method_names (list): A list of method names to be used as node names in the tree.
        n_params (int, optional): Number of parameters per function. Defaults to 0.
        n_vars (int, optional): Number of variables per function. Defaults to 0.
        parent (Node, optional): Parent node to link the double comb with.
        shape (list, optional): Describes the shape of the comb parts of the tree. Defaults to ["left", "right"].
        
    Returns:
        Node: The root of the double comb binary tree.
    """
    if not method_names:
        return
    
    comb_depth = max_distance//2 + 2
    
    if parent is not None:
        root = parent
    else:
        root = Node(name=method_names.pop(0), n_params=n_params, n_vars=n_vars)
    
    setattr(root, shape[0], build_comb_tree(comb_depth, max_distance, method_names, n_params, n_vars, root, shape))
    setattr(root, shape[1], build_near_comb_tree(comb_depth, max_distance, method_names, n_params, n_vars, root, shape))
    
    return root

def build_comb_tree(depth: int, 
                    max_distance: int, 
                    method_names: list, 
                    n_params: int=0, 
                    n_vars: int=0, 
                    parent: Node=None,
                    shape: list=["left", "right"]
                    ) -> Node:
    """Build a comb binary tree from a list of method names.

    Args:
        depth (int): Depth of the tree
        max_distance (int): The maximum distance for which we wish to ask negative questions.
        method_names (list): A list of method names to be used as node names in the tree.
        n_params (int, optional): Number of parameters per function. Defaults to 0.
        n_vars (int, optional): Number of variables per function. Defaults to 0.
        parent (Node, optional): Parent node to link the comb with.
        shape (list, optional): Describes the shape of the comb parts of the tree. Defaults to ["left", "right"].
        
    Returns:
        Node: The root of the comb binary tree.
    """
    if not method_names:
        return
    
    
    current_index = 0
    root = Node(name=method_names.pop(0), n_params=n_params, n_vars=n_vars, parent=parent)
    current_index += 1
    current_node = root
    
    depth_iter = depth
    
    while depth_iter > 0:
        if not method_names:
            return root
        
        new_node = Node(name=method_names.pop(0), n_params=n_params, n_vars=n_vars, parent=current_node)
        setattr(current_node, shape[0], new_node) 
        current_index += 1
        current_node = getattr(current_node, shape[0])
        depth_iter -= 1
# if current_index < max_distance:
    current_node = current_node.parent
    while current_index - 4 < max_distance and current_node is not None:
        if not method_names:
            return root
        
        new_node = Node(name=method_names.pop(0), n_params=n_params, n_vars=n_vars, parent=current_node)
        setattr(current_node, shape[1], new_node)
        current_index += 1
        current_node = current_node.parent
    
    if current_node is None:
        return root
    
    # new_node = build_near_comb_tree(depth, max_distance, method_names, n_params, n_vars, root, shape[::-1])
    # setattr(root, shape[1], new_node)
    
    return root

def build_near_comb_tree(depth: int, 
                        max_distance: int, 
                        method_names: list,
                        n_params: int=0, 
                        n_vars: int=0, 
                        parent: Node=None,
                        shape: list=["left", "right"]
                        ) -> Node:
    """Build an near comb binary tree from a list of method names.

    Args:
        depth (int): Depth of the tree.
        max_distance (int): The maximum distance for which we wish to ask negative questions.
        method_names (list): A list of method names to be used as node names in the tree.
        n_params (int): Number of parameters to include for each method node.
        n_vars (int): Number of variables to include for each method node.
        parent (Node, optional): Parent node to link the near comb with.
        shape (list, optional): Describes the shape of the comb parts of the tree. Defaults to ["left", "right"].
        
    Returns:
        Node: The root of the near comb binary tree.
    """
    if not method_names:
        return
    
    current_index = 0
    root = Node(name=method_names.pop(0), n_params=n_params, n_vars=n_vars, parent=parent)
    current_index += 1
    current_node = root
    
    depth_iter = depth
    
    while depth_iter > 0:
        if not method_names:
            return root
    
        new_node = Node(name=method_names.pop(0), n_params=n_params, n_vars=n_vars, parent=current_node)
        setattr(current_node, shape[0], new_node)
        current_index += 1
        current_node = getattr(current_node, shape[0])
        depth_iter -= 1
# if current_index < max_distance + 2:
    current_node = current_node.parent
    current_node = current_node.parent
        
    while current_index - 3 < max_distance and current_node is not None:
        if not method_names:
            return root
        
        new_node = Node(name=method_names.pop(0), n_params=n_params, n_vars=n_vars, parent=current_node)
        setattr(current_node, shape[1], new_node)
        current_index += 1
        current_node = current_node.parent
    
    if current_node is None:
        return root
    
    return root

def build_unbalanced_binary_tree(max_distance: int, method_names: list, n_params: int=0, n_vars: int=0) -> Node:
    """Build an unbalanced binary tree from a list of method names.

    Args:
        max_distance (int): The maximum distance for which we wish to ask negative questions.
        method_names (list): A list of method names to be used as node names in the tree.
        n_params (int, optional): Number of parameters per function. Defaults to 0.
        n_vars (int, optional): Number of variables per function. Defaults to 0.
        
    Returns:
        Node: The root of the binary tree.
    """
    if not method_names:
        return
    
    depth = max_distance
    root = Node(name=method_names.pop(0), n_params=n_params, n_vars=n_vars)
    
    build_left_branch(root, depth, method_names, n_params=n_params, n_vars=n_vars)
    
    return root

def build_left_branch(root: Node, depth: int, method_names: list, n_params: int=0, n_vars: int=0):
    """
    Build a branch (only using the left child of each node)
    
    Args:
        root (Node): Root of the tree, the branch starts here.
        depth (int): Depth of the tree.
        method_names (list): A list of method names to be used as node names in the tree.
        n_params (int): Number of parameters to include for each method node.
        n_vars (int): Number of variables to include for each method node.
    """
    if (depth is not None and depth < 0) or not method_names: 
        return

    root.left = Node(name=method_names.pop(0), parent=root, n_params=n_params, n_vars=n_vars)
    build_left_branch(root.left, depth-1 if depth is not None else depth, method_names, n_params=n_params, n_vars=n_vars)

def build_branch(method_names: list, n_params: int=0, n_vars: int=0) -> Node:
    """
    Build a branch. Used for linear tree calls.
    
    Args:
        depth (int): Depth of the tree.
        method_names (list): A list of method names to be used as node names in the tree.
        n_params (int): Number of parameters to include for each method node.
        n_vars (int): Number of variables to include for each method node.
    
    Returns:
        Node: The root of the branch
    """
    root = Node(name=method_names.pop(0), n_params=n_params, n_vars=n_vars)
    build_left_branch(root, None, method_names, n_params, n_vars)
    return root

def generate_many_branches(chains: list[list[str]], n_params: int, n_vars: int) -> tuple[list[Node], list[list[str]]]:
    """
    Build many branches. Used for linear tree calls.

    Args:
        chains (list[list[str]]): List of chains. Each chain is a list of method names.
        n_params (int): Number of parameters to include for each method node.
        n_vars (int): Number of variables to include for each method node.

    Returns:
        tuple[list[Node], list[list[str]]]: List of trees and chains used to generate them
    """
    ret_chains = copy.deepcopy(chains)
    
    trees = []
    
    for chain in chains:
        trees.append(build_branch(chain, n_params, n_vars))
    
    return trees, ret_chains

def build_diamond_tree(method_names: list, 
                        max_distance: int,
                        direction: str=None, 
                        n_params: int=0, 
                        n_vars: int=0, 
                        parent: Node=None
                        ) -> Node:
    """
    Build a tree shaped like a diamond (broad at half height)
    
    Args:
        method_names (list): A list of method names to be used as node names in the tree.
        max_distance (int): The maximum distance for which we wish to ask negative questions.
        n_params (int): Number of parameters to include for each method node.
        n_vars (int): Number of variables to include for each method node.
        
    Returns:
        Node: The root of the tree
    """
    if not method_names:
        return None
        
    if parent and (parent.path.count("left") >= 5 or parent.path.count("right") >= 5):
        if (parent.path.count("left") < 5 or parent.path.count("right") < 5):
            return None
    if parent: 
        path = parent.path.copy()
        path.append(direction)
    else:
        path = []
    node = Node(name=method_names.pop(0), n_params=n_params, n_vars=n_vars, path=path, parent=parent)
    
    node.left = build_diamond_tree(method_names, max_distance, "left", n_params, n_vars, node)
    if parent and (parent.path.count("left") % 2 == 0 and parent.path.count("right") % 2 == 0):
        return node
    node.right = build_diamond_tree(method_names, max_distance, "right", n_params, n_vars, node)
    
    return node

""" Tree traversal/search functions """

def depth_first_traversal(node: Node) -> tuple[list[str], int, int, int]:
    """
    Perform a depth-first traversal of the tree.

    Args:
        node (Node): The root node of the tree.
        
    Returns:
        tuple[list[str], int, int, int]: List of method names, size of subtree, size of subtree with backtracking, relative height. 
    """
    
    method_names = []
    counter = -1
    counter_with_backtracking = -1
    node_traversal = node
    
    def dft_rec(current_node: Node):
        nonlocal counter, counter_with_backtracking, node_traversal
        if current_node is None:
            return
        
        method_names.append(current_node.name)
        counter += 1
        counter_with_backtracking += 1
        node_traversal = current_node
        
        dft_rec(current_node.left)
        if current_node.left is not None:
            counter_with_backtracking += 1
        
        dft_rec(current_node.right)
        if current_node.right is not None:
            counter_with_backtracking += 1
        

    dft_rec(node)
    
    relative_height = node_traversal.get_relative_height(node)
    
    counter_with_backtracking -= relative_height
    
    # print(f"Depth-first traversal completed. Distance: {counter}. With backtracking: {counter_with_backtracking}. Height: {relative_height}")
    
    return method_names, counter, counter_with_backtracking, relative_height       


def depth_first_search(node: Node, search_node: Node) -> tuple[list[str], int, int, int]:
    """Perform a depth-first search of the tree.
    The search starts from node and stops at search_node
    The point is to get a distances (normal, with backtracking and height)

    Args:
        node (Node): The root node of the tree.
        search_node (Node): The node we're searching for.
            
    Returns:
        tuple[list[str], int, int, int]: List of method names, size of subtree, size of subtree with backtracking, relative height. 
    """
    
    method_names = []
    distance = 0
    distance_with_backtracking = 0
    is_found = False
    
    def dft_rec(current_node: Node):
        nonlocal distance, distance_with_backtracking, is_found
        if current_node is None or is_found:
            return
        
        method_names.append(current_node.name)
        
        if current_node == search_node:
            is_found = True
            return
        
        distance += 1
        distance_with_backtracking += 1
        
        dft_rec(current_node.left)
        if is_found: 
            return
        if current_node.left is not None:
            distance_with_backtracking += 1
            
        
        dft_rec(current_node.right)
        if is_found: 
            return
        if current_node.right is not None:
            distance_with_backtracking += 1
        

    dft_rec(node)
    
    relative_height = search_node.get_relative_height(node)
    
    # print(f"Depth-first traversal completed. Distance: {distance}. With backtracking: {distance_with_backtracking}. Height: {relative_height}")
    
    return method_names, distance, distance_with_backtracking, relative_height


""" Chain finding functions """

def find_all_valid_chains_depth_first(node: Node, chains: list = None) -> list:
    """Find all method chains in a tree using depth-first traversal.
    This function traverses the tree in a depth-first manner and collects the names of the methods in the order they are visited.
    See doc for more details on the amount of such chains.

    Args:
        node (Node): The root node of the tree.

    Returns:
        list: A list of list of strings, each representing a chain of Java methods.
    """
    if chains is None:
        chains = []
        
    if node is None:
        return []
    
    # If the node is a leaf node (no children), return an empty list
    if node.left is None and node.right is None:
        return []
        
    def find_chains_starting_from(root_node: Node, current_node: Node = None) -> list[dict]:
        nonlocal chains
        if current_node is None or root_node is None:
            return
        
        if root_node != current_node:
            chain, distance, distance_with_backtracking, distance_height = depth_first_search(root_node, current_node)
            
            chains.append({
                "chain": chain,
                "distance": distance,
                "distance_with_backtracking": distance_with_backtracking,
                "distance_height": distance_height,
            })    
            
        
        # Traverse left and right children
        find_chains_starting_from(root_node, current_node.left)
        find_chains_starting_from(root_node, current_node.right)
        
    # Start the depth-first traversal from the root node
    find_chains_starting_from(node, node)

    # As we are looking for ALL chains, we need to do the same operation for the subtrees of the root node
    find_all_valid_chains_depth_first(node.left, chains)
    find_all_valid_chains_depth_first(node.right, chains)
    
    # The formula for the number of chains is Σ 2^k x (2^{h+1-k} -2) for k in [0, h-1] where h is the height of the tree.
    # This is because for each level k, we have 2^k nodes, and each node can have 2^{h+1-k} - 2 chains.
    # The total number of chains is the sum of all chains from all levels. 
    
    return chains


def find_all_invalid_chains_depth_first(node: Node, root: Node = None, chains: list = None) -> list:
    """Find all invalid method chains in a tree using depth-first traversal.
    This function finds "invalid" chains, which are just the basis for "NO" questions to ask the LLMs.
    This method looks for invalid chains within a single tree, since all of these trees are perfect binary trees,
    each node has a left and right child, and the tree is balanced.
    This means that with the method the invalid chains will always be of the form: 2^k - 1
    See doc for more details on the amount of such chains.

    Args:
        node (Node): The root node of the tree.

    Returns:
        list: A list of list of strings, each representing a chain of Java methods.
    """
    if chains is None:
        chains = []
    
    if node is None:
        return []
    
    if root is None:
        find_all_invalid_chains_depth_first(node=node.left, root=node, chains=chains)
        find_all_invalid_chains_depth_first(node=node.right, root=node, chains=chains)
        return chains
    
    # Compute the size of the subtree rooted at the current node and get the chain associated with it 
    chain_from_subtree, size_of_subtree, size_of_subtree_with_backtracking, height = depth_first_traversal(node)
    # The distance of the invalid chain is the size of the subtree from this node (negative value)
    # since it is the number of methods that the LLM must check to determine if the chain is valid or not. 
    distance = -size_of_subtree
    distance_with_backtracking = -size_of_subtree_with_backtracking
    distance_height = height
        
    # Start the depth-first traversal from the root node
    unreachable_methods = find_list_of_unreachable_methods(node, root)
    
    if distance != 0:
        chains.append({
            "node": node.name,
            "unreachable_methods": unreachable_methods,
            "distance": distance,
            "distance_with_backtracking": distance_with_backtracking,
            "distance_height": distance_height,
            "chain": chain_from_subtree
        })

    # As we are looking for ALL chains, we need to do the same operation for the subtrees of the root node
    find_all_invalid_chains_depth_first(node=node.left, root=root, chains=chains)
    find_all_invalid_chains_depth_first(node=node.right, root=root, chains=chains)
    
    return chains

def find_list_of_unreachable_methods(node: Node, root: Node) -> list:
    """Build a list of methods that are unreachable from the Node "node".

    Args:
        node (Node): The node for which we are looking for unreachable methods.
        root (Node): The root of the tree in which the node is.

    Returns:
        list: The list of names of unreachable methods.
    """
    unreachable_methods = []
    
    # Traverse the tree from the root node to find methods that are not reachable from the current node
    if root == node or root is None:
        return []
    
    else:   
        unreachable_methods.append(root.name)
        # Traverse the tree in a depth-first manner to find unreachable methods
        unreachable_methods.extend(find_list_of_unreachable_methods(node, root.left))
        unreachable_methods.extend(find_list_of_unreachable_methods(node, root.right))
        
    return unreachable_methods




# Build a binary tree with a depth of 3 and method names

if __name__ == '__main__':
    method_names = [
    "foo", "bar", "baz", "qux", "quux", "corge", "grault", "garply",
    "waldo", "fred", "plugh", "xyzzy", "thud", "zedd", "last"
    ]
    tree = build_binary_tree(3, method_names)
    tree.print_tree()


    print(depth_first_search(tree, tree.right.right.right))


"""
Jellyfish tree                                                                                                                                                                                                                                                                                               
                                                                                                                                    ┌────────────┐                                                                                                                                             
                                                                                                                                    │            │                                                                                                                                             
                                                                                                                                    │            │                                                                                                                                             
                                                                                                                                    │            │                                                                                                                                             
                                                                                                                                    └──┬──────┬──┘                                                                                                                                             
                                                                                                                                       │      │                                                                                                                                                
                                                                                          ┌────────────────────────────────────────────┘      └───────────────────────────────────────────┐                                                                                                    
                                                                                          │                                                                                               │                                                                                                    
                                                                                          ▼                                                                                               ▼                                                                                                    
                                                                                    ┌────────────┐                                                                                 ┌────────────┐                                                                                              
                                                                                    │            │                                                                                 │            │                                                                                              
                                                                                    │            │                                                                                 │            │                                                                                              
                                                                                    │            │                                                                                 │            │                                                                                              
                                                                                    └──┬──────┬──┘                                                                                 └──┬──────┬──┘                                                                                              
                                                                                       │      │                                                                                       │      │                                                                                                 
                                                      ┌────────────────────────────────┘      └───────────────────┐                                                ┌──────────────────┘      └───────────────────────────────────────┐                                                         
                                                      │                                                           │                                                │                                                                 │                                                         
                                                      │                                                           │                                                │                                                                 │                                                         
                                                      ▼                                                           ▼                                                ▼                                                                 ▼                                                         
                                               ┌────────────┐                                               ┌────────────┐                                  ┌────────────┐                                                    ┌────────────┐                                                   
                                               │            │                                               │            │                                  │            │                                                    │            │                                                   
                                               │            │                                               │            │                                  │            │                                                    │            │                                                   
                                               │            │                                               │            │                                  │            │                                                    │            │                                                   
                                               └──┬──────┬──┘                                               └────────────┘                                  └────────────┘                                                    └──┬──────┬──┘                                                   
                                                  │      │                                                                                                                                                                       │      │                                                      
                                  ┌───────────────┘      └───────────────┐                                                                                                                                       ┌───────────────┘      └───────────────┐                                      
                                  │                                      │                                                                                                                                       │                                      │                                      
                                  ▼                                      ▼                                                                                                                                       ▼                                      ▼                                      
                           ┌────────────┐                         ┌────────────┐                                                                                                                          ┌────────────┐                         ┌────────────┐                                
                           │            │                         │            │                                                                                                                          │            │                         │            │                                
                           │            │                         │            │                                                                                                                          │            │                         │            │                                
                           │            │                         │            │                                                                                                                          │            │                         │            │                                
                           └──┬──────┬──┘                         └──┬──────┬──┘                                                                                                                          └──┬──────┬──┘                         └──┬──────┬──┘                                
                        ┌─────┘      └──────┐                  ┌─────┘      └──────┐                                                                                                                   ┌─────┘      └──────┐                  ┌─────┘      └──────┐                            
                        │                   │                  │                   │                                                                                                                   │                   │                  │                   │                            
                        ▼                   ▼                  ▼                   ▼                                                                                                                   ▼                   ▼                  ▼                   ▼                            
                  ┌────────────┐      ┌────────────┐     ┌────────────┐      ┌────────────┐                                                                                                      ┌────────────┐      ┌────────────┐     ┌────────────┐      ┌────────────┐                     
                  │            │      │            │     │            │      │            │                                                                                                      │            │      │            │     │            │      │            │                     
                  │            │      │            │     │            │      │            │                                                                                                      │            │      │            │     │            │      │            │                     
                  │            │      │            │     │            │      │            │                                                                                                      │            │      │            │     │            │      │            │                     
                  └──┬──────┬──┘      └────────────┘     └──┬──────┬──┘      └────────────┘                                                                                                      └────────────┘      └──┬──────┬──┘     └────────────┘      └──┬──────┬──┘                     
               ┌─────┘      └──────┐                  ┌─────┘      └──────┐                                                                                                                                       ┌─────┘      └──────┐                  ┌─────┘      └──────┐                 
               │                   │                  │                   │                                                                                                                                       │                   │                  │                   │                 
               ▼                   ▼                  ▼                   ▼                                                                                                                                       ▼                   ▼                  ▼                   ▼                 
         ┌────────────┐      ┌────────────┐     ┌────────────┐      ┌────────────┐                                                                                                                          ┌────────────┐      ┌────────────┐     ┌────────────┐      ┌────────────┐          
         │            │      │            │     │            │      │            │                                                                                                                          │            │      │            │     │            │      │            │          
         │            │      │            │     │            │      │            │                                                                                                                          │            │      │            │     │            │      │            │          
         │            │      │            │     │            │      │            │                                                                                                                          │            │      │            │     │            │      │            │          
         └──┬──────┬──┘      └────────────┘     └──┬─────────┘      └────────────┘                                                                                                                          └────────────┘      └────────────┘     └────────────┘      └───┬───┬────┘          
      ┌─────┘      └──────┐                  ┌─────┘                                                                                                                                                                                  └──────┐                      ┌──────┘   └───────┐       
      │                   │                  │                                                                                                                                                                                               │                      │                  │       
      ▼                   ▼                  ▼                                                                                                                                                                                               ▼                      ▼                  ▼       
┌────────────┐      ┌────────────┐     ┌────────────┐                                                                                                                                                                                  ┌────────────┐        ┌────────────┐     ┌────────────┐ 
│            │      │            │     │            │                                                                                                                                                                                  │            │        │            │     │            │ 
│            │      │            │     │            │                                                                                                                                                                                  │            │        │            │     │            │ 
│            │      │            │     │            │                                                                                                                                                                                  │            │        │            │     │            │ 
└────────────┘      └────────────┘     └────────────┘                                                                                                                                                                                  └────────────┘        └────────────┘     └────────────┘ 
"""

""" 
Another version of Jellyfish tree
                                                                                           ┌────────────┐                                                                                                                                                                                                     
                                                                                           │            │                                                                                                                                                                                                     
                                                                                           │            │                                                                                                                                                                                                     
                                                                                           │            │                                                                                                                                                                                                     
                                                                                           └──┬──────┬──┘                                                                                                                                                                                                     
                                                                                              │      │                                                                                                                                                                                                        
                                                     ┌────────────────────────────────────────┘      └────────────────────────────────────────┐                                                                                                                                                               
                                                     │                                                                                        │                                                                                                                                                               
                                                     │                                                                                        │                                                                                                                                                               
                                                     ▼                                                                                        ▼                                                                                                                                                               
                                               ┌────────────┐                                                                          ┌────────────┐                                                                                                                                                         
                                               │            │                                                                          │            │                                                                                                                                                         
                                               │            │                                                                          │            │                                                                                                                                                         
                                               │            │                                                                          │            │                                                                                                                                                         
                                               └──┬──────┬──┘                                                                          └──┬──────┬──┘                                                                                                                                                         
                                                  │      │                                                                                │      │                                                                                                                                                            
                                  ┌───────────────┘      └───────────────┐                                                ┌───────────────┘      └───────────────┐                                                                                                                                            
                                  │                                      │                                                │                                      │                                                                                                                                            
                                  │                                      │                                                │                                      │                                                                                                                                            
                                  ▼                                      ▼                                                ▼                                      ▼                                                                                                                                            
                           ┌────────────┐                         ┌────────────┐                                   ┌────────────┐                         ┌────────────┐                                                                                                                                      
                           │            │                         │            │                                   │            │                         │            │                                                                                                                                      
                           │            │                         │            │                                   │            │                         │            │                                                                                                                                      
                           │            │                         │            │                                   │            │                         │            │                                                                                                                                      
                           └──┬──────┬──┘                         └──┬──────┬──┘                                   └──┬──────┬──┘                         └──┬──────┬──┘                                                                                                                                      
                        ┌─────┘      └──────┐                  ┌─────┘      └──────┐                            ┌─────┘      └──────┐                  ┌─────┘      └──────┐                                                                                                                                  
                        │                   │                  │                   │                            │                   │                  │                   │                                                                                                                                  
                        ▼                   ▼                  ▼                   ▼                            ▼                   ▼                  ▼                   ▼                                                                                                                                  
                  ┌────────────┐      ┌────────────┐     ┌────────────┐      ┌────────────┐               ┌────────────┐      ┌────────────┐     ┌────────────┐      ┌────────────┐                                                                                                                           
                  │            │      │            │     │            │      │            │               │            │      │            │     │            │      │            │                                                                                                                           
                  │            │      │            │     │            │      │            │               │            │      │            │     │            │      │            │                                                                                                                           
                  │            │      │            │     │            │      │            │               │            │      │            │     │            │      │            │                                                                                                                           
                  └──┬──────┬──┘      └────────────┘     └──┬──────┬──┘      └────────────┘               └────────────┘      └──┬──────┬──┘     └────────────┘      └──┬──────┬──┘                                                                                                                           
               ┌─────┘      └──────┐                  ┌─────┘      └──────┐                                                ┌─────┘      └──────┐                  ┌─────┘      └──────┐                                                                                                                       
               │                   │                  │                   │                                                │                   │                  │                   │                                                                                                                       
               ▼                   ▼                  ▼                   ▼                                                ▼                   ▼                  ▼                   ▼                                                                                                                       
         ┌────────────┐      ┌────────────┐     ┌────────────┐      ┌────────────┐                                   ┌────────────┐      ┌────────────┐     ┌────────────┐      ┌────────────┐                                                                                                                
         │            │      │            │     │            │      │            │                                   │            │      │            │     │            │      │            │                                                                                                                
         │            │      │            │     │            │      │            │                                   │            │      │            │     │            │      │            │                                                                                                                
         │            │      │            │     │            │      │            │                                   │            │      │            │     │            │      │            │                                                                                                                
         └──┬──────┬──┘      └────────────┘     └──┬─────────┘      └────────────┘                                   └────────────┘      └────────────┘     └────────────┘      └───┬───┬────┘                                                                                                                
      ┌─────┘      └──────┐                  ┌─────┘                                                                                           └──────┐                      ┌──────┘   └───────┐                                                                                                             
      │                   │                  │                                                                                                        │                      │                  │                                                                                                             
      ▼                   ▼                  ▼                                                                                                        ▼                      ▼                  ▼                                                                                                             
┌────────────┐      ┌────────────┐     ┌────────────┐                                                                                           ┌────────────┐        ┌────────────┐     ┌────────────┐                                                                                                       
│            │      │            │     │            │                                                                                           │            │        │            │     │            │                                                                                                       
│            │      │            │     │            │                                                                                           │            │        │            │     │            │                                                                                                       
│            │      │            │     │            │                                                                                           │            │        │            │     │            │                                                                                                       
└────────────┘      └────────────┘     └────────────┘                                                                                           └────────────┘        └────────────┘     └────────────┘                                                                                                       
"""

"""
Double Comb tree

                                               ┌────────────┐
                                               │            │                                                   
                                               │            │
                                               │            │
                                               └──┬──────┬──┘
                                                  │      │
                                  ┌───────────────┘      └───────────────┐
                                  │                                      │
                                  ▼                                      ▼
                           ┌────────────┐                         ┌────────────┐
                           │            │                         │            │
                           │            │                         │            │
                           │            │                         │            │
                           └──┬──────┬──┘                         └──┬──────┬──┘
                        ┌─────┘      └──────┐                  ┌─────┘      └──────┐
                        │                   │                  │                   │
                        ▼                   ▼                  ▼                   ▼
                  ┌────────────┐      ┌────────────┐     ┌────────────┐      ┌────────────┐
                  │            │      │            │     │            │      │            │
                  │            │      │            │     │            │      │            │
                  │            │      │            │     │            │      │            │
                  └──┬──────┬──┘      └────────────┘     └──┬──────┬──┘      └────────────┘
               ┌─────┘      └──────┐                  ┌─────┘      └──────┐
               │                   │                  │                   │
               ▼                   ▼                  ▼                   ▼
         ┌────────────┐      ┌────────────┐     ┌────────────┐      ┌────────────┐
         │            │      │            │     │            │      │            │ 
         │            │      │            │     │            │      │            │ 
         │            │      │            │     │            │      │            │ 
         └──┬──────┬──┘      └────────────┘     └──┬─────────┘      └────────────┘
      ┌─────┘      └──────┐                  ┌─────┘
      │                   │                  │
      ▼                   ▼                  ▼
┌────────────┐      ┌────────────┐     ┌────────────┐
│            │      │            │     │            │
│            │      │            │     │            │
│            │      │            │     │            │
└────────────┘      └────────────┘     └────────────┘
"""