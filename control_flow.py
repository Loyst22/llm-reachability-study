import random
import generate_chain

def random_variable():
    """Randomly pick a variable name."""
    return random.choice(["x", "y", "z", "counter", "flag"])

def random_declaration():
    """Generate a random variable declaration."""
    var_type = random.choice(["int", "boolean", "double"])
    var_name = random_variable()
    if var_type == "int":
        return f"    int {var_name} = {random.randint(1, 10)};"
    elif var_type == "boolean":
        return f"    boolean {var_name} = {random.choice([True, False])};"
    else:  # double
        return f"    double {var_name} = {random.uniform(1.0, 10.0)};"

def random_condition():
    """Generate a simple condition involving declared variables."""
    var = random_variable()
    return f"{var} > {random.randint(5, 20)}"

def random_method_call(called_method):
    """Generate the next method call."""
    return f"    {called_method}();"

def random_loop():
    """Generate a random for or while loop."""
    loop_type = random.choice(["for", "while"])
    if loop_type == "for":
        return f"    for (int i = 0; i < {random.randint(1, 5)}; i++) {{ {random_method_call('methodTwo')} }}\n"
    else:  # while loop
        return f"    while (counter < 5) {{ {random_method_call('methodThree')} counter++; }}\n"

def generate_method_body(called_method):
    """Generate a method body with simple control flow, declarations, and method calls."""
    body = []
    
    # Declare random variables (can be empty)
    for _ in range(random.randint(0, 2)):  # Random number of variables to declare
        body.append(random_declaration())
    
    # Add an optional condition check (like an 'if' statement)
    if random.choice([True, False]):  # Decide if we add a condition or not
        condition = random_condition()
        body.append(f"    if ({condition}) {{ {random_method_call(called_method)} }}")
    
    # Add an optional loop (either a for loop or a while loop)
    if random.choice([True, False]):
        body.append(random_loop())
    
    # Add the actual method call (to the next method in the chain)
    body.append(f"    {called_method}();")
    
    return "\n".join(body)

def generate_method_bodies(method_names):
    """Generate random method bodies for a list of method names."""
    bodies = []
    for i, method_name in enumerate(method_names):
        # Each method calls the next one in the list, or wraps around to the first
        called_method = method_names[(i + 1) % len(method_names)]
        body = generate_method_body(called_method)
        bodies.append(f"public void {method_name}() {{\n{body}\n}}")
    return bodies

# Example usage
method_names = generate_chain.generate_unique_method_names(10)
method_bodies = generate_method_bodies(method_names)

# Print the generated method bodies
for body in method_bodies:
    print(body)
    print()

