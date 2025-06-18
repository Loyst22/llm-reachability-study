import random
import generate_chain as generate_chain

# restructure to pass a function to generate bodies to generate_chain

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
    """Randomly pick a variable name among a predefined set."""
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

def random_condition(variables) -> str:
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
    """Generate the next method call."""
    return f"\t{called_method}();"

def random_loop() -> str:
    """Generate a random for or while loop."""
    loop_type = random.choice(["for", "while"])
    if loop_type == "for":
        #return f"    for (int i = 0; i < {random.randint(1, 5)}; i++) {{ {random_method_call('methodTwo')} }}\n"
        return f"\tfor (int i = 0; i < {random.randint(1, 5)}; i++) {{\n\t\tint blah = i;\n\t}}"
    else:  # while loop
        return f"\twhile (counter < 5) {{\n\t\tcounter++;\n\t}}"

""" Method body generation functions. """

def generate_method_body(called_method: str, n_vars: int) -> str:
    """Generate a method body with simple control flow, declarations, and method calls.

    Args:
        called_method (str): name of the method being called
        n_vars (int): number of variables to declare in the method body

    Returns:
        str: Java method body with variable declarations, conditions, and method calls
    """

    body = []
    variables = []
    
    # Declare random variables (can be empty)
    for _ in range(n_vars):  # Random number of variables to declare
        var = random_variable()
        variables.append(var)  # Store the variable for later use in conditions
        body.append(f"\t{var.var_type} {var.name} = {var.value};")
    
    # Add an optional condition check (like an 'if' statement)
    if random.choice([True, False]):  # Decide if we add a condition or not
        condition = random_condition(variables)
        body.append(f"\tif ({condition}) {{\n\t{random_method_call(called_method)} \n\t}}")
    else: 
        body.append(random_loop())
        body.append(f"\t{called_method}();")
    # commenting out extra complexity, add back later
    # Add an optional loop (either a for loop or a while loop)
    #if random.choice([True, False]):
    #    
    
    # Add the actual method call (to the next method in the chain)
    #body.append(f"\t{called_method}();")
    
    return "\n".join(body)

def generate_method(caller: str, called: str, nvars: int) -> str:
    """Generate a method that calls another method with a specified number of variables.

    Args:
        caller (str): name of the caller method being generated
        called (str): name of the method being called
        nvars (int): number of variables to declare in the method body

    Returns:
        str: Java method definition with a body that includes variable declarations
    """
    body = generate_method_body(called, nvars)
    return f"public void {caller}() {{\n{body}\n}}"

def generate_method_bodies(method_names: list) -> list:
    """Generate random method bodies for a list of method names.

    Args:
        method_names (list): list of method names to generate bodies for

    Returns:
        list: list of Java method bodies as strings
    """
    bodies = []
    # variables = []  # List to store declared variables
    
    for i, method_name in enumerate(method_names):
        # Each method calls the next one in the list, or wraps around to the first
        called_method = method_names[(i + 1) % len(method_names)]
        
        # ! weird : shouldn't work to pass variables, since it is a list not a number
        # body = generate_method_body(called_method, variables)
        body = generate_method_body(called_method, 3) 
        bodies.append(f"public void {method_name}() {{\n{body}\n}}")
    
    return bodies


def generate_chained_method_calls(method_names: list) -> list:
    """Generate a series of method bodies where each method calls the next one in the list.
    
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

""" Test """

# Example usage
method_names = generate_chain.generate_unique_method_names(15)
method_bodies = generate_method_bodies(method_names)

# Print the generated method bodies
for body in method_bodies:
   print(body)
   print()
