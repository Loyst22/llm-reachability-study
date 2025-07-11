from pathlib import Path
import control_flow

class Node:
    """
    A class representing a node in a binary tree.
    Each node is actually respresenting a function.
    
    Attributes:
        name (str): Name of the function/method.
        parent (Node): Parent node in the tree.
        left (Node): Left child node.
        right (Node): Right child node.
        params (list[Variable]): List of parameters for the method.
        variables (list[Variable]): List of variables defined in the body of the method.
        return_type (str): Return type of the method.
        var_types (list[str]): List of variable types used in the method. (params + local variables)
        all_variables (list[Variable]): List of variables defined/used in the method.
    """
    def __init__(self, name: str, n_params: int = 0, n_vars: int = 0, parent=None, left=None, right=None):
        self.name = name
        self.parent = parent
        self.left = left
        self.right = right
        
        # ! Be careful with building that at init, parents might get added later
        # ! atm there is no issue with that, but it might break in the future
        if parent is None:
            self.params = control_flow.random_variables(n_params)
        else:
            self.params = control_flow.choose_n_vars(n_params, parent.all_variables)
        
        # Get names of all param variables to avoid duplication
        param_names = {var.name for var in self.params}
        self.variables = []
        
        while len(self.variables) < n_vars:
            additional_vars = control_flow.random_variables(n_vars - len(self.variables))
            self.variables.extend([var for var in additional_vars if var.name not in param_names and var.name not in {v.name for v in self.variables}])
        
        self.all_variables = self.variables + self.params # Add parameters to the list of variables
        
        self.var_types = [var.var_type for var in self.variables]
        
        # Unused atm
        """ 
        if return_type is None:
            self.return_type = "void"    
        else:
            self.return_type = return_type  # Return type of the method (if any)
        """
        
    def __str__(self):
        parent_name = self.parent.name if self.parent else "None"
        left_name = self.left.name if self.left else "None"
        right_name = self.right.name if self.right else "None"
        param_types = [var.var_type for var in self.params]
        variable_types = [var.var_type for var in self.variables]

        return (
            f"Node(name={self.name}, "
            f"parent={parent_name}, "
            f"left={left_name}, "
            f"right={right_name}, "
            f"param_types={param_types}, "
            f"variable_types={variable_types}),"
            f"variables={self.variables}"
        )

    def print_tree(self, indent: str = ""):
        if self is None:
            return
        print(indent + self.name)
        if self.left:
            self.left.print_tree(indent + "   L- ")
        if self.right:
            self.right.print_tree(indent + "   R- ")

    def write_tree_to_file(self, file_path: str):
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
            
    def get_subtree_size(self):
        size = 1
        if self.left is not None:
            size += self.left.get_subtree_size()
        if self.right is not None:
            size += self.right.get_subtree_size()
        return size
    
    def get_list_of_nodes(self):
        nodes = [self]
        if self.left is not None:
            nodes.extend(self.left.get_list_of_nodes())
        if self.right is not None:
            nodes.extend(self.right.get_list_of_nodes())
        return nodes
    
    def get_method_names(self):
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

def build_binary_tree(depth: int, method_names: list[str], parent: Node = None) -> tuple[Node, list[str]]:
    """Build a binary tree from a list of method names.

    Args:
        depth (int): Depth of the tree.
        method_names (list): A list of method names to be used as node names in the tree.
        index (int, optional): Used to track the current index in the method_names list while building the tree. Defaults to 0.
        parent (Node, optional): The parent node of the current node. Defaults to None        
        
    Returns:
        tuple: The root of the binary tree, and the list of remaining method names
    """
    if depth < 0 or not method_names:
        return None

    # Pop the current method name
    name = method_names.pop(0)
    node = Node(name=name, parent=parent)

    # Build left and right subtrees, consuming names in-place
    node.left = build_binary_tree(depth - 1, method_names, node)
    node.right = build_binary_tree(depth - 1, method_names, node)

    return node

def build_unbalanced_binary_tree(max_distance: int, method_names: list) -> Node:
    """Build an unbalanced binary tree from a list of method names.

    Args:
        max_distance (int): The maximum distance for which we wish to ask negative questions.
        method_names (list): A list of method names to be used as node names in the tree.
        
    Returns:
        Node: The root of the binary tree
    """
    if not method_names:
        return
    
    depth = max_distance
    root = Node(method_names.pop(0))
    
    build_left_branch(root, depth, method_names)
    
    # root.right = build_binary_tree(depth-1, method_names)
    # root.right.parent = root
    
    return root


def build_unbalanced_binary_tree_v2(depth: int, max_distance: int, method_names: list) -> Node:
    """Build an unbalanced binary tree from a list of method names.

    Args:
        depth (int): Depth of the tree
        max_distance (int): The maximum distance for which we wish to ask negative questions.
        method_names (list): A list of method names to be used as node names in the tree.
        
    Returns:
        Node: The root of the binary tree
    """
    if not method_names:
        return
        
    current_index = 0
    root = Node(method_names.pop(0))
    current_index += 1
    current_node = root
    
    depth_iter = depth
    
    while depth_iter > 0:
        if not method_names:
            return root
        
        current_node.left = Node(method_names.pop(0), current_node)
        current_index += 1
        current_node = current_node.left
        depth_iter -= 1
    if current_index < max_distance:
        current_node = current_node.parent
    while current_index - 1 < max_distance and current_node is not None:
        if not method_names:
            return root
        
        current_node.right = Node(method_names.pop(0), current_node)
        current_index += 1
        current_node = current_node.parent
    
    if current_node is None:
        return root
    
    # else:
    #     current_node.right = build_binary_tree(depth, method_names[current_index:])
    #     current_node.right.parent = current_node
        
    
    return root

def build_unbalanced_binary_tree_v3(depth: int, max_distance: int, method_names: list) -> Node:
    """Build an unbalanced binary tree from a list of method names.

    Args:
        depth (int): Depth of the tree
        max_distance (int): The maximum distance for which we wish to ask negative questions.
        method_names (list): A list of method names to be used as node names in the tree.
        
    Returns:
        Node: The root of the binary tree
    """
    if not method_names:
        return
    
    current_index = 0
    root = Node(method_names.pop(0))
    current_index += 1
    current_node = root
    
    depth_iter = depth
    
    while depth_iter > 0:
        if not method_names:
            return root
    
        current_node.left = Node(method_names.pop(0), current_node)
        current_index += 1
        current_node = current_node.left
        depth_iter -= 1
    if current_index < max_distance + 2:
        current_node = current_node.parent
        current_node = current_node.parent
        
    while current_index < max_distance + 2 and current_node is not None:
        if not method_names:
            return root
        
        current_node.right = Node(method_names.pop(0), current_node)
        current_index += 1
        current_node = current_node.parent
    
    if current_node is None:
        return root
    # else:
    #     root.right = build_binary_tree(depth, method_names[current_index:])
    #     root.right.parent = root
        
    
    return root

    
def build_left_branch(root: Node, depth: int, method_names: list):
    if depth < 0 or not method_names: 
        return

    root.left = Node(method_names.pop(0), root)
    build_left_branch(root.left, depth-1, method_names)

def depth_first_traversal(node: Node):
    """Perform a depth-first traversal of the tree.

    Args:
        node (Node): The root node of the tree.
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
    
    relative_height = get_relative_height(node_traversal, node)
    
    counter_with_backtracking -= relative_height
    
    # print(f"Depth-first traversal completed. Distance: {counter}. With backtracking: {counter_with_backtracking}. Height: {relative_height}")
    
    return method_names, counter, counter_with_backtracking, relative_height       

def depth_first_search(node: Node, search_node: Node):
    """Perform a depth-first search of the tree.
    The search starts from node and stops at search_node
    The point is to get a distances (normal, with backtracking and height)

    Args:
        node (Node): The root node of the tree.
        search_node (Node): The node we're searching for
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
    
    relative_height = get_relative_height(search_node, node)
    
    # distance_with_backtracking -= relative_height
    
    # print(f"Depth-first traversal completed. Distance: {distance}. With backtracking: {distance_with_backtracking}. Height: {relative_height}")
    
    return method_names, distance, distance_with_backtracking, relative_height


def get_height(node: Node, height: int = 0) -> int:
    if node.parent is None:
        return height
    return get_height(node=node.parent, height=height+1)

def get_relative_height(node: Node, relative_parent: Node, height: int = 0) -> int:
    if node == relative_parent:
        return height
    return get_relative_height(node.parent, relative_parent, height+1)

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
        
    def find_chains_starting_from(root_node: Node, current_node: Node = None):
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

"""
method_names = [
"foo", "bar", "baz", "qux", "quux", "corge", "grault", "garply",
"waldo", "fred", "plugh", "xyzzy", "thud", "zedd", "last"
]
tree = build_binary_tree(3, method_names)
tree.print_tree()


print(depth_first_search(tree, tree.right.right.right))
"""