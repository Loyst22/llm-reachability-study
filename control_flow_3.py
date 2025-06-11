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

def random_variable_name():
    """Randomly pick a variable name."""
    return random.choice(["x", "y", "z", "counter", "flag"])

def random_variable():
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

def random_condition(variables):
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
    
def random_method_call(called_method):
    """Generate the next method call using the given method name."""
    return f"    {called_method}();"

def random_loop():
    """Generate a random for or while loop."""
    loop_type = random.choice(["for", "while"])
    if loop_type == "for":
        return f"    for (int i = 0; i < {random.randint(1, 5)}; i++) {{ {random_method_call('nextMethod')} }}\n"
    else:  # while loop
        return f"    while (counter < 5) {{ {random_method_call('nextMethod')} counter++; }}\n"

def generate_method_body(called_method, variables, next_method):
    """Generate a method body with simple control flow, declarations, and method calls."""
    body = []
    
    # Declare random variables (can be empty)
    for _ in range(random.randint(1, 3)):  # Random number of variables to declare
        var = random_variable()
        variables.append(var)  # Store the variable for later use in conditions
        body.append(f"    {var.var_type} {var.name} = {var.value};")
    
    # Add an optional condition check (like an 'if' statement)
    if random.choice([True, False]):  # Decide if we add a condition or not
        condition = random_condition(variables)
        body.append(f"    if ({condition}) {{ {random_method_call(next_method)} }}")
    
    # Add an optional loop (either a for loop or a while loop)
    if random.choice([True, False]):
        body.append(random_loop())
    
    # Add the actual method call (to the next method in the chain)
    body.append(f"    {called_method}();")
    
    return "\n".join(body)

def generate_method_bodies(method_names):
    """Generate random method bodies for a list of method names."""
    bodies = []
    variables = []  # List to store declared variables
    
    # Iterate over the method names and create a body for each
    for i, method_name in enumerate(method_names):
        # Each method calls the next one in the list, or wraps around to the first
        next_method = method_names[(i + 1) % len(method_names)]
        body = generate_method_body(method_name, variables, next_method)
        bodies.append(f"public void {method_name}() {{\n{body}\n}}")
    
    return bodies

# Example usage
method_names = generate_chain.generate_unique_method_names(15)
method_bodies = generate_method_bodies(method_names)

# Print the generated method bodies
for body in method_bodies:
    print(body)
    print()
