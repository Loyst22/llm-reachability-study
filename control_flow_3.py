import random
import generate_chain

# Simple data structure to hold variable information
class Variable:
    def __init__(self, name, var_type, value):
        self.name = name
        self.var_type = var_type
        self.value = value
    
    def __repr__(self):
        return f"{self.name} ({self.var_type} = {self.value})"

""" Random generation functions for control flow in Java methods. """

def random_variable_name() -> str:
    """Randomly pick a variable name."""
    return random.choice(["x", "y", "z", "counter", "flag"])

def random_variable() -> Variable:
    """Generate a random variable with a type and value."""
    var_name = random_variable_name()
    var_type = random.choice(["int", "boolean", "double"])
    
    if var_type == "int":
        value = random.randint(1, 10)
    elif var_type == "boolean":
        value = random.choice([True, False])
    else:  # double
        value = round(random.uniform(1.0, 10.0), 2)
    
    return Variable(var_name, var_type, value)

def random_condition(variables: list) -> list:
    """Generate a simple condition that always evaluates to True using declared variables."""
    # Choose a random variable
    var = random.choice(variables)
    
    # Generate conditions based on variable type, ensuring they are always true
    if var.var_type == "int":
        # For int, ensure the condition is true based on the value of the variable
        if var.value <= 5:
            return f"{var.name} <= {var.value}"  # x <= 5 where x is <= 5
        else:
            return f"{var.name} >= {var.value}"  # x >= 5 where x is > 5
    elif var.var_type == "boolean":
        # For boolean, create a condition that evaluates to true
        return f"{var.name} == {var.value}"
    elif var.var_type == "double":
        # For double, ensure the condition is true based on the value of the variable
        if var.value <= 5.0:
            return f"{var.name} <= {var.value}"  # z <= value where z <= value
        else:
            return f"{var.name} >= {var.value}"  # z >= value where z > value
    
def random_method_call(called_method: str) -> str:
    """Generate the next method call using the given method name."""
    return f"\t{called_method}();"

def random_loop(next_method: str) -> str:
    """Generate a random for or while loop."""
    loop_type = random.choice(["for", "while"])
    if loop_type == "for":
        return f"\tfor (int i = 0; i < {random.randint(1, 5)}; i++) {{\n\t{random_method_call(next_method)}\n\t}}"
    else:  # while loop
        return f"\twhile (counter < 5) {{\n\t{random_method_call(next_method)}\n\t\tcounter++;\n\t}}"

""" Method body generation functions. """

def generate_method_body(called_method: str, variables: list, next_method: str) -> str:
    # TODO : this generation method is not correct, it creates recursive functions since called_method calls itself
    """Generate a method body with simple control flow, declarations, and method calls.
    
    Args: 
        called_method (str): name of the method being called
        variables (list) : list of the variables being used
        next_method (str) : name of the method being called
        
    Returns: 
        str: Java method body with variable declarations, conditions, and method calls
    """
    body = []
    
    # Declare random variables (can be empty)
    for _ in range(random.randint(1, 3)):  # Random number of variables to declare
        var = random_variable()
        variables.append(var)  # Store the variable for later use in conditions
        body.append(f"\t{var.var_type} {var.name} = {var.value};")
    
    # Add an optional condition check (like an 'if' statement)
    if random.choice([True, False]):  # Decide if we add a condition or not
        condition = random_condition(variables)
        body.append(f"\tif ({condition}) {{\n\t{random_method_call(next_method)}\n\t}}")
    
    # Add an optional loop (either a for loop or a while loop)
    if random.choice([True, False]):
        body.append(random_loop(next_method))
    
    # Add the actual method call (to the next method in the chain)
    body.append(f"\t{called_method}();")
    
    return "\n".join(body)

def generate_method_bodies(method_names: list) -> list:
    """Generate random method bodies for a list of method names.

    Args:
        method_names (list): list of method names to generate bodies for

    Returns:
        list: list of Java method bodies as strings
    """
    bodies = []
    variables = []  # List to store declared variables
    
    # Iterate over the method names and create a body for each
    for i, method_name in enumerate(method_names):
        # Each method calls the next one in the list, or wraps around to the first
        next_method = method_names[(i + 1) % len(method_names)]
        body = generate_method_body(method_name, variables, next_method)
        bodies.append(f"public void {method_name}() {{\n{body}\n}}")
    
    return bodies


def generate_full_class(nb_methods: int=15, nb_loops: int=None, nb_if: int=None, nb_chains: int=1):
    method_names = generate_chain.generate_unique_method_names(nb_methods)
    method_bodies = generate_method_bodies(method_names)
    
    for body in method_bodies:
        print(body)
        print()

""" Test """

generate_full_class()
