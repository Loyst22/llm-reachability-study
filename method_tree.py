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
