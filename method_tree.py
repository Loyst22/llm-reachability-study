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
            self.left.print_tree(indent + "  L- ")
            self.right.print_tree(indent + "  R- ")


def build_binary_tree(depth: int, method_names: list, index: int = 0) -> Node:
    if depth < 0 or index >= len(method_names):
        return None
    name = method_names[index]
    left = build_binary_tree(depth - 1, method_names, 2 * index + 1)
    right = build_binary_tree(depth - 1, method_names, 2 * index + 2)
    return Node(name, left, right)

