from pathlib import Path
class Node:
    def __init__(self, name, left=None, right=None):
        self.name = name      # Name of java method
        self.left = left      # Left child node
        self.right = right    # Right child node

    def print_tree(self, indent: str = ""):
        if self is None:
            return
        print(indent + self.name)
        if self.left or self.right:
            self.left.print_tree(indent + "   L- ")
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

def build_binary_tree(depth: int, method_names: list, index: int = 0) -> Node:
    """Build a binary tree from a list of method names.

    Args:
        depth (int): Description of the depth of the tree.
        method_names (list): A list of method names to be used as node names in the tree.
        index (int, optional): Used to track the current index in the method_names list while building the tree. Defaults to 0.

    Returns:
        Node: _description_
    """
    if depth < 0 or index >= len(method_names):
        return None
    name = method_names[index]
    left = build_binary_tree(depth - 1, method_names, 2 * index + 1)
    right = build_binary_tree(depth - 1, method_names, 2 * index + 2)
    return Node(name, left, right)

def depth_first_traversal(node: Node):
    """Perform a depth-first traversal of the tree.

    Args:
        node (Node): The root node of the tree.
    """
    
    method_names = []
    compteur = 0
    
    def dft_rec(current_node: Node):
        nonlocal compteur
        if current_node is None:
            return
        method_names.append(current_node.name)
        compteur += 1
        dft_rec(current_node.left)
        dft_rec(current_node.right)
    
    dft_rec(node)
    print(f"Depth-first traversal completed. Total methods: {compteur}")
    print("Method names:", method_names)
    return method_names, compteur        
        

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
    
    # If the node is a leaf node (no children), return an empty list
    if node.left is None and node.right is None:
        return []
        
    def dft_rec(current_node: Node, current_chain: list=[]):
        nonlocal chains
        if current_node is None:
            return

        if node.left is None and node.right is None:
            return
        
        # Add the current method name to the chain
        current_chain.append(current_node.name)
        
        # As we want to get all chains, we append the current chain to the chains list
        # Avoid adding single method chains
        if len(current_chain) > 1: 
            chains.append(current_chain.copy())
        
        
        # Traverse left and right children
        dft_rec(current_node.left, current_chain)
        dft_rec(current_node.right, current_chain)
        
        # # Backtrack to explore other branches
        # current_chain.pop()
        
    # Start the depth-first traversal from the root node
    dft_rec(node)

    # As we are looking for ALL chains, we need to do the same operation for the subtrees of the root node
    find_all_valid_chains_depth_first(node.left, chains)
    find_all_valid_chains_depth_first(node.right, chains)
    
    # The formula for the number of chains is Σ 2^k x (2^{h+1-k} -2) for k in [0, h-1] where h is the height of the tree.
    # This is because for each level k, we have 2^k nodes, and each node can have 2^{h+1-k} - 2 chains.
    # The total number of chains is the sum of all chains from all levels. 
    
    return chains
    