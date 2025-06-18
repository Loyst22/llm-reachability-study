import random
import generate_chain as generate_chain

# Simple data structure to hold variable information
class Variable:
    def __init__(self, name, var_type, value):
        self.name = name
        self.var_type = var_type
        self.value = value
    
    def __repr__(self):
        if self.var_type == "boolean":
            java_value = str(self.value).lower()
            return f"{self.name} ({self.var_type} = {java_value})"
        else:
            return f"{self.name} ({self.var_type} = {self.value})"

""" Random generation functions for control flow in Java methods. """

def random_variable_name() -> str:
    """Randomly pick a variable name."""
    return random.choice(["x", "y", "z", "var", "cpt", "flag"])

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
        java_value = str(var.value).lower()
        return f"{var.name} == {java_value}"
    elif var.var_type == "double":
        # For double, ensure the condition is true based on the value of the variable
        if var.value <= 5.0:
            return f"{var.name} <= {var.value}"  # z <= value where z <= value
        else:
            return f"{var.name} >= {var.value}"  # z >= value where z > value
    
def method_call(called_method: str) -> str:
    """Generate the next method call using the given method name."""
    if called_method is None:
        return f"\t// End of chain"
    else:
        return f"\t{called_method}();"

def random_loop(next_method: str = None) -> str:
    """Generate a random for or while loop."""
    loop_type = random.choice(["for", "while"])
    if loop_type == "for":
        if next_method is None:
            return f"\tfor (int i = 0; i < {random.randint(1, 5)}; i++) {{\n\t\tSystem.out.println(i);\n\t}}"
        else:
            return f"\tfor (int i = 0; i < {random.randint(1, 5)}; i++) {{\n\t{method_call(next_method)}\n\t}}"
    else:  # while loop
        if next_method is None:
            return f"\tint counter = 0;\n\twhile (counter < 5) {{\n\t\tSystem.out.println(counter);\n\t\tcounter++;\n\t}}"
        else:
            return f"\tint counter = 0;\n\twhile (counter < 5) {{\n\t{method_call(next_method)}\n\t\tcounter++;\n\t}}"

""" Method body generation functions. """

def generate_method_body(next_method: str = None,
                        n_vars: int = 0,
                        n_loops: int = 0,
                        n_if: int = 0) -> str:
    """Generate a method body with simple control flow, declarations, and method calls.
    
    Args: 
        called_method (str): name of the method being called
        variables (list) : list of the variables being used
        next_method (str) : name of the method being called
        
    Returns: 
        str: Java method body with variable declarations, conditions, and method calls
    """
    body = []
    variables = []
    control_flow = []
    
    # Declare random variables (can be empty)
    for _ in range(n_vars):  # Random number of variables to declare
        var = random_variable()
        variables.append(var)  # Store the variable for later use in conditions
        if var.var_type == "boolean":
            java_value = str(var.value).lower()
            body.append(f"\t{var.var_type} {var.name} = {java_value};")
        else:    
            body.append(f"\t{var.var_type} {var.name} = {var.value};")
    
    # Add optional condition checks (like 'if' statements)
    for _ in range(n_if):  # Decide if we add a condition or not
        condition = random_condition(variables)
        control_flow.append(f"\tif ({condition}) {{\n\t{method_call(next_method)}\n\t}}")
    
    # Add an optional loop (either a for loop or a while loop)
    for _ in range(n_loops):
        # next_method may be None of valid here : both work
        control_flow.append(random_loop(next_method))
    
    random.shuffle(control_flow)
    body.extend(control_flow)
    
    # Add the actual method call (to the next method in the chain)
    # ! Invalid version, it should call next_method or else it is recursive
    # body.append(f"\t{called_method}();")
    if next_method != None :
        body.append(f"\t{next_method}();")
    else :
        body.append(f"\t // End of chain")
    
    return "\n".join(body)

def generate_method_bodies(method_names: list) -> list:
    """Generate random method bodies for a list of method names.
        They all call each other and it wraps around

    Args:
        method_names (list): list of method names to generate bodies for

    Returns:
        list: list of Java method bodies as strings
    """
    bodies = []
    
    # Iterate over the method names and create a body for each
    for i, method_name in enumerate(method_names):
        # Each method calls the next one in the list, or wraps around to the first
        next_method = method_names[(i + 1) % len(method_names)]
        body = generate_method_body(method_name, next_method)
        bodies.append(f"public void {method_name}() {{\n{body}\n}}")
    
    return bodies

def generate_chained_method_calls(method_names: list) -> list:
    """Generate a series of method bodies where each method calls the next one in the list.
        The last method does not call anything (end of chain).
    
    Args:
        method_names (list): list of method names to generate bodies for
    
    Returns:
        list: list of Java method bodies as strings, where each method calls the next
    """
    method_bodies = []

    # Loop through the list of method names
    for i, method in enumerate(method_names):
        # Check if this is the last method in the list
        if i < len(method_names) - 1:
            # Call the next method in the list
            next_method = method_names[i + 1]

            method_body = generate_method(method, next_method, 3)
        else:
            # Last method, no call to the next method
            method_body = f"public void {method}() {{\n    // End of chain\n}}"
        
        # Append to the list of method bodies
        method_bodies.append(method_body)
    
    return method_bodies

def generate_method(caller_method: str,
                    called_method: str = None,
                    n_vars: int = 0,
                    n_loops: int = 0,
                    n_if: int = 0) -> str:
    """Generate a method that calls another method with a specified number of variables.

    Args:
        caller_method (str): name of the caller method being generated
        called_method (str): name of the method being called
        nvars (int): number of variables to declare in the method body

    Returns:
        str: Java method definition with a body that includes variable declarations
    """
    body = generate_method_body(called_method, n_vars, n_loops, n_if)
    return f"public void {caller_method}() {{\n{body}\n}}"

def generate_full_class(nb_methods: int=15, n_loops: int=None, n_if: int=None, nb_chains: int=1):
    method_names = generate_chain.generate_unique_method_names(nb_methods)
    method_bodies = generate_method_bodies(method_names)
    
    for body in method_bodies:
        print(body)
        print()

""" Test """

# generate_full_class()
