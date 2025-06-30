from math import ceil
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

    @staticmethod
    def random_variable_name() -> str:
        """Randomly pick a variable name."""
        return random.choice(["x", "y", "z", "var", "cpt", "flag", "temp", "data", "result", "value",
                              "input", "output", "index", "count", "total", "sum", "avg", "num", "max",
                              "min", "length", "size", "height", "width", "depth", "name", "id", "key",
                              "item", "node", "list", "array", "map", "dict", "buffer", "record", "line",
                              "text", "path", "file", "error", "status", "response",
                              "user", "message", "token", "config", "option", "mode"])
    
""" Random generation functions for control flow in Java methods. """

def is_var_in_list(var: Variable, list: list) -> bool:
    """Check if a variable with the same name exists in the list."""
    return any(v.name == var.name for v in list)

def random_variable(var_type: str = None) -> Variable:
    """Generate a random variable with a type and value."""
    var_name = Variable.random_variable_name()
    if var_type is None or var_type not in ["int", "long", "boolean", "double"]:
        var_type = random.choice(["int", "long", "boolean", "double"])
    
    if var_type == "int" or var_type == "long":
        value = random.randint(1, 10)
    elif var_type == "boolean":
        value = random.choice([True, False])
    else:  # double
        value = round(random.uniform(1.0, 10.0), 2)
    
    return Variable(var_name, var_type, value)


def random_true_condition(variables: list, complexity: int = 0) -> str:
    """Generate a simple condition that always evaluates to True using declared variables."""
    # Choose a random variable
    var = random.choice(variables)
    
    if complexity == 3:
        rand = random.random()
        if rand < 1/4:
            return "(" + random_true_condition(variables, 1) + ") && (" + random_true_condition(variables, 1) + ")"
        elif rand < 1/2:
            return "(" + random_true_condition(variables, 1) + ") || (" + random_true_condition(variables, 1) + ")"
        elif rand < 3/4:
            return "(" + random_true_condition(variables, 1) + ") || (" + random_false_condition(variables, 1) + ")"
        else:
            return "(" + random_false_condition(variables, 1) + ") || (" + random_true_condition(variables, 1) + ")"
    
    if complexity == 2:
        rand = random.random()
        if rand < 1/8:
            return "(" + random_true_condition(variables, 1) + ") && " + random_true_condition(variables, 0)
        elif rand < 1/4:
            return random_true_condition(variables, 0) + " && (" + random_true_condition(variables, 1) + ")"
        elif rand < 3/8:
            return "(" + random_true_condition(variables, 1) + ") || " + random_true_condition(variables, 0)
        elif rand < 1/2:
            return random_true_condition(variables, 0) + " || (" + random_true_condition(variables, 1) + ")"
        elif rand < 5/8:
            return "(" + random_true_condition(variables, 1) + ") || " + random_false_condition(variables, 0)
        elif rand < 3/4:
            return random_true_condition(variables, 0) + " || (" + random_false_condition(variables, 1) + ")"
        elif rand < 7/8:
            return "(" + random_false_condition(variables, 1) + ") || " + random_true_condition(variables, 0)
        else:
            return random_false_condition(variables, 0) + " || (" + random_true_condition(variables, 1) + ")"
            
            
    
    if complexity == 1:
        rand = random.random()
        if rand < 1/4:
            return random_true_condition(variables, 0) + " && " + random_true_condition(variables, 0)
        elif rand < 1/2:
            return random_true_condition(variables, 0) + " || " + random_true_condition(variables, 0)
        elif rand < 3/4:
            return random_true_condition(variables, 0) + " || " + random_false_condition(variables, 0)
        else:
            return random_false_condition(variables, 0) + " || " + random_true_condition(variables, 0)
        
    
    # Generate conditions based on variable type, ensuring they are always true
    if complexity == 0:
        match var.var_type:
            case "int" | "long":
                # For int, ensure the condition is true based on the value of the variable
                delta = random.randint(1, 5) # To avoid situation like 5 <= 5...
                if var.value <= 5:
                    return f"{var.name} <= {var.value + delta}"  # x <= x + delta where x is <= value
                else:
                    return f"{var.name} >= {var.value - delta}"  # x >= x - delta where x is > value

            case "boolean":
                # For boolean, create a condition that evaluates to true
                if random.random() < 0.5:
                    java_value = str(var.value).lower()
                    return f"{var.name} == {java_value}"
                else:
                    not_java_value = str(not var.value).lower()
                    return f"{var.name} != {not_java_value}"

            case "double":
                # For double, ensure the condition is true based on the value of the variable
                delta = random.uniform(1, 5)
                if var.value <= 5.0:
                    return f"{var.name} <= {round(var.value + delta, 2)}"  # z <= z + delta where z <= 5
                else:
                    return f"{var.name} >= {round(var.value - delta, 2)}"  # z >= z - delta where z > 5
            

def random_false_condition(variables: list, complexity: int = 0) -> str:
    """Generate a simple condition that always evaluates to False using declared variables."""
    # Choose a random variable
    var = random.choice(variables)
    
    if complexity == 3:
        rand = random.random()
        if rand < 1/4:
            return "(" + random_false_condition(variables, 1) + ") && (" + random_false_condition(variables, 1) + ")"
        elif rand < 1/2:
            return "(" + random_false_condition(variables, 1) + ") && (" + random_true_condition(variables, 1) + ")"
        elif rand < 3/4:
            return "(" + random_true_condition(variables, 1) + ") && (" + random_false_condition(variables, 1) + ")"
        else:
            return "(" + random_false_condition(variables, 1) + ") || (" + random_false_condition(variables, 1) + ")"
    
    if complexity == 2:
        rand = random.random()
        if rand < 1/8:
            return "(" + random_false_condition(variables, 1) + ") && " + random_false_condition(variables, 0)
        elif rand < 1/4:
            return random_false_condition(variables, 0) + " && (" + random_false_condition(variables, 1) + ")"
        elif rand < 3/8:
            return "(" + random_false_condition(variables, 1) + ") && " + random_true_condition(variables, 0)
        elif rand < 1/2:
            return random_false_condition(variables, 0) + " && (" + random_true_condition(variables, 1) + ")"
        elif rand < 5/8:
            return "(" + random_true_condition(variables, 1) + ") && " + random_false_condition(variables, 0)
        elif rand < 3/4:
            return random_true_condition(variables, 0) + " && (" + random_false_condition(variables, 1) + ")"
        elif rand < 7/8:
            return "(" + random_false_condition(variables, 1) + ") || " + random_false_condition(variables, 0)
        else:
            return random_false_condition(variables, 0) + " || (" + random_false_condition(variables, 1) + ")"
    
    if complexity == 1:
        rand = random.random()
        if rand < 1/4:
            return random_false_condition(variables, 0) + " && " + random_false_condition(variables, 0)
        elif rand < 1/2:
            return random_false_condition(variables, 0) + " && " + random_true_condition(variables, 0)
        elif rand < 3/4:
            return random_true_condition(variables, 0) + " && " + random_false_condition(variables, 0)
        else:
            return random_false_condition(variables, 0) + " || " + random_false_condition(variables, 0)
    
    if complexity == 0:
        match var.var_type:
            case "int" | "long":
                # For int, ensure the condition is false based on the value of the variable
                delta = random.randint(1, 5) 
                if var.value <= 5:
                    return f"{var.name} <= {var.value - delta}"  # x <= x - delta where x is <= 5
                else:
                    return f"{var.name} >= {var.value + delta}"  # x >= x + delta where x is > 5
            
            case "boolean":
                # For boolean, create a condition that evaluates to false
                if random.random() < 0.5:
                    not_java_value = str(not var.value).lower()
                    return f"{var.name} == {not_java_value}"
                else:
                    java_value = str(var.value).lower()
                    return f"{var.name} != {java_value}"
            
            case "double":
                # For double, ensure the condition is false based on the value of the variable
                delta = random.uniform(1, 5)
                if var.value <= 5.0:
                    return f"{var.name} <= {round(var.value - delta, 2)}"  # z <= z - delta where z <= 5
                else:
                    return f"{var.name} >= {round(var.value + delta, 2)}"  # z >= z + delta where z > 5
                
        
def random_true_if(variables: list, next_method: str = None) -> str:
    """Generate a random if statement"""
    condition = random_true_condition(variables)
    if next_method is not None:
        return f"\tif ({condition}) {{\n\t{method_call(next_method)}\n\t}}"
    else:
        var = random.choice(variables)
        return f"\tif ({condition}) {{\n\t\tSystem.out.println({var.name});\n\t}}"
    
def random_false_if(variables: list, useless_method: str = None) -> str:
    """Generate a random if statement"""
    condition = random_false_condition(variables)
    if useless_method is not None:
        return f"\tif ({condition}) {{\n\t{method_call(useless_method)}\n\t}}"
    else:
        var = random.choice(variables)
        return f"\tif ({condition}) {{\n\t\tSystem.out.println({var.name});\n\t}}"
    
def method_call(called_method: str) -> str:
    """Generate the next method call using the given method name."""
    if called_method is None:
        return f"\t// End of chain"
    else:
        return f"\t{called_method}();"

def random_loop(next_method: str = None, nb_while: int = 0) -> tuple[str, str]:
    """Generate a random for or while loop."""
    loop_type = random.choice(["for", "while"])
    if loop_type == "for":
        if next_method is None:
            return f"\tfor (int i = 0; i < {random.randint(1, 5)}; i++) {{\n\t\tSystem.out.println(i);\n\t}}", "for"
        else:
            return f"\tfor (int i = 0; i < {random.randint(1, 5)}; i++) {{\n\t{method_call(next_method)}\n\t}}", "for"
    else:  # while loop
        if nb_while == 0:
            if next_method is None:
                return f"\tint counter = 0;\n\twhile (counter < {random.randint(1, 5)}) {{\n\t\tSystem.out.println(counter);\n\t\tcounter++;\n\t}}", "while"
            else:
                return f"\tint counter = 0;\n\twhile (counter < {random.randint(1, 5)}) {{\n\t{method_call(next_method)}\n\t\tcounter++;\n\t}}", "while"
        else:
            if next_method is None:
                return f"\tcounter = 0;\n\twhile (counter < {random.randint(1, 5)}) {{\n\t\tSystem.out.println(counter);\n\t\tcounter++;\n\t}}", "while"
            else:
                return f"\tcounter = 0;\n\twhile (counter < {random.randint(1, 5)}) {{\n\t{method_call(next_method)}\n\t\tcounter++;\n\t}}", "while"                
            
""" Method body generation functions. """

def generate_method_body(next_methods: list = None, n_vars: int = 0, n_loops: int = 0, n_if: int = 0) -> str:
    """Generate a method body with simple control flow, declarations, and method calls.
    
    Args: 
        next_methods (list): list of names of methods being called
        n_vars (int): number of variables to declare
        n_loops (int): number of loops to include
        n_if (int): number of if statements to include
        
    Returns: 
        str: Java method body with variable declarations, conditions, and method calls
    """
    body = []
    variables = []
    control_flow = []
    
    if next_methods is None:
        next_methods = []
        
    # Declare random variables (can be empty)
    for _ in range(n_vars):  # Random number of variables to declare
        var = random_variable()
        while is_var_in_list(var=var, list=variables):
            var = random_variable()
        variables.append(var)  # Store the variable for later use in conditions
        if var.var_type == "boolean":
            java_value = str(var.value).lower()
            body.append(f"\t{var.var_type} {var.name} = {java_value};")
        else:    
            body.append(f"\t{var.var_type} {var.name} = {var.value};")
    
    end_of_chain = not next_methods
    
    # We define the types of control flow blocks to add to the method
    # And we shuffle them to avoid having all if statement separated from the loops 
    total_methods = len(next_methods)
    total_blocks = n_if + n_loops
    calls_in_if = 0
    calls_in_loop = 0
    calls_in_plain = 0

    if total_blocks > 0 and total_methods > 0:
        # Allocate proportionally
        calls_in_if = ceil((n_if / total_blocks) * total_methods)
        calls_in_loop = ceil((n_loops / total_blocks) * total_methods)
        calls_in_plain = total_methods - (calls_in_if + calls_in_loop)
    else:
        # If no blocks, all method calls are plain
        calls_in_plain = total_methods
    
    control_flow_types = []
    
    # Add if statement blocks (either with method calls or without)
    for _ in range(calls_in_if):
        control_flow_types.append(("if", True))
    for _ in range(n_if - calls_in_if):
        control_flow_types.append(("if", False))
    # Add loops - while and for (either with method calls or without)
    for _ in range(calls_in_loop):
        control_flow_types.append(("loop", True))
    for _ in range(n_loops - calls_in_loop):
        control_flow_types.append(("loop", False))
    # Add plain method calls if necessary
    for _ in range(calls_in_plain):
        control_flow_types.append(("plain", True))
        
    random.shuffle(control_flow_types)
    
    # Used to declare counter when necessary and to avoid doing it when not necessary
    nb_while = 0
    
    for block_type, has_call in control_flow_types:
        # Generate a if statement
        if block_type == "if":
            if next_methods and has_call:
                next_method = next_methods.pop(0)
                control_flow.append(random_true_if(variables, next_method))
            else:
                control_flow.append(random_true_if(variables))
        
        # Generate a loop (while or for)
        if block_type == "loop":
            if next_methods and has_call:
                next_method = next_methods.pop(0)
                loop_code, loop_type = random_loop(next_method, nb_while)
            else:
                loop_code, loop_type = random_loop(None, nb_while)
            
            # Next_method may be None of valid here : both work
            control_flow.append(loop_code)
            
            if loop_type == "while":
                nb_while += 1
            
        # Add simple method calls (when no more control flow)
        if block_type == "plain":
            if next_methods and has_call:
                next_method = next_methods.pop(0)
                control_flow.append(f"\t{next_method}();")
        
        
    body.extend(control_flow)
    
    # If it's the end of chain we inform the LLM
    # ? is it actually necessary ?
    if end_of_chain:
        body.append(f"\t// End of chain")
    
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

def generate_method(caller_method: str, called_methods: list = None, n_vars: int = 0, n_loops: int = 0, n_if: int = 0) -> str:
    """Generate a method that calls another method with a specified number of variables.

    Args:
        caller_method (str): name of the caller method being generated
        called_methods (list): list of names of methods being called
        nvars (int): number of variables to declare in the method body

    Returns:
        str: Java method definition with a body that includes variable declarations
    """
    body = generate_method_body(called_methods, n_vars, n_loops, n_if)
    return f"public void {caller_method}() {{\n{body}\n}}"

def generate_full_class(nb_methods: int=15, n_loops: int=None, n_if: int=None, nb_chains: int=1):
    method_names = generate_chain.generate_unique_method_names(nb_methods)
    method_bodies = generate_method_bodies(method_names)
    
    for body in method_bodies:
        print(body)
        print()

""" Test """

# generate_full_class()

if __name__ == '__main__':
    a = random_variable("int")
    b = random_variable("int")
    c = random_variable("int")
    d = random_variable("int")
    
    
    print("First variable:", a)
    print("Second variable:", b)
    print("Third variable:", c)
    print("Fourth variable:", d)
    
    print()
    
    print("True condition:", random_true_condition([a, b, c, d], 3))
    print("False condition:", random_false_condition([a, b, c, d], 3))
    
    