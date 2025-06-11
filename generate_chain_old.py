import random
import string
import prompts as p
from pathlib import Path


def generate_random_java_method_name():
    prefixes = [
        "get", "set", "is", "calculate", "process", "fetch", "update", "create", "delete", 
        "find", "check", "load", "save", "reset", "clear", "validate", "initialize", 
        "convert", "apply", "enable", "disable", "sort", "merge", "copy", "generate", 
        "retrieve", "parse", "extract", "compare", "build", "register", "unregister",
        "sync", "execute", "dispatch", "resolve", "filter", "log"
    ]
    
    verbs = [
        "Data", "Item", "Value", "State", "Config", "Status", "Object", "Parameter", "Setting", 
        "Resource", "Detail", "Info", "Message", "Handler", "Element", "Connection", "Index", 
        "Entry", "Key", "Session", "Metric", "Field", "Action", "Notification", "Instance", 
        "Node", "Task", "Job", "Event", "Request", "Response", "Flag", "File", "Directory", 
        "Path", "Buffer", "User", "Account", "Transaction", "Cache", "Result", "List", 
        "Map", "Queue", "Stack", "Collection", "Component", "Service", "Manager"
    ]
    
    nouns = [
        "ById", "ForUser", "WithFilter", "InCache", "FromDatabase", "FromFile", "ToJson", 
        "FromXml", "IfAvailable", "OrDefault", "AsString", "FromUrl", "OnClick", "InMemory", 
        "FromApi", "ForSession", "WithTimeout", "ForRequest", "FromResponse", "AtIndex", 
        "WithKey", "WithIndex", "ForTransaction", "IfValid", "OnInit", "AsList", "ForRole", 
        "ToBuffer", "ForMapping", "OnComplete", "AtPosition", "ToSet", "AsMap", "AsQueue", 
        "WithLimit", "ToCollection", "ForEach", "IfEnabled", "WithPolicy", "InThread", 
        "ForExecution", "InParallel", "AsObservable", "IfExists", "WithRetries"
    ]

    prefix = random.choice(prefixes)
    verb = random.choice(verbs)
    noun = random.choice(nouns)

    return prefix + verb + noun

def generate_random_method_name(length):
    # Generate a random string of specified length with uppercase, lowercase letters, and digits
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


def generate_unique_method_names(n):
    unique_names = set()
    while len(unique_names) < n:
        method_name = generate_random_java_method_name()
        unique_names.add(method_name)
    
    return list(unique_names)

def generate_chained_method_calls(method_names):
    method_bodies = []

    # Loop through the list of method names
    for i, method in enumerate(method_names):
        # Check if this is the last method in the list
        if i < len(method_names) - 1:
            # Call the next method in the list
            next_method = method_names[i + 1]
            method_body = f"public void {method}() {{\n    {next_method}();\n}}"
        else:
            # Last method, no call to the next method
            method_body = f"public void {method}() {{\n    // End of chain\n}}"
        
        # Append to the list of method bodies
        method_bodies.append(method_body)
    
    return method_bodies

# Example usage
#method_names = ["methodOne", "methodTwo", "methodThree", "methodFour"]
#method_bodies = generate_chained_method_calls(method_names)

#for body in method_bodies:
#    print(body + "\n")


def generate_unique_names(n):
     # Generate unique random method names
    method_names = set()
    while len(method_names) < n:
        method_names.add(generate_random_java_method_name())
    return list(method_names)


def generate_class_with_chained_methods(class_name, n, method_length=10):
    method_names = generate_unique_method_names(n)

    # Generate the chain of method calls
    method_bodies = generate_chained_method_calls(method_names)

    # Shuffle the method bodies to create random order in the class
    random.shuffle(method_bodies)

    # Construct the class with shuffled method bodies
    class_body = f"public class {class_name} {{\n"
    class_body += "\n\n".join(method_bodies)
    class_body += "\n}"

    return class_body

def generate_class_with_chained_methods_names(class_name, method_names):
    # Generate the chain of method calls
    method_bodies = generate_chained_method_calls(method_names)

    # Shuffle the method bodies to create random order in the class
    random.shuffle(method_bodies)

    # Construct the class with shuffled method bodies
    class_body = f"public class {class_name} {{\n"
    class_body += "\n\n".join(method_bodies)
    class_body += "\n}"

    return class_body

def flatten_list(lst):
    return [item for sublist in lst for item in sublist]


def generate_class_with_multiple_chains(class_name, chains):
    # Generate the chain of method calls
    method_bodies = []
    for c in chains:
        method_bodies.append(generate_chained_method_calls(c))

    method_bodies = flatten_list(method_bodies)
    # Shuffle the method bodies to create random order in the class
    random.shuffle(method_bodies)

    # Construct the class with shuffled method bodies
    class_body = f"public class {class_name} {{\n"
    class_body += "\n\n".join(method_bodies)
    class_body += "\n}"

    return class_body

def generate_call_dependency_matrix(method_names):
    # Initialize the matrix as a list of lists (n x n), filled with -1 (no calls)
    n = len(method_names)
    matrix = [[-1] * n for _ in range(n)]
    
    # Distance of 0 for each method calling itself
    for i in range(n):
        matrix[i][i] = 0
    
    # Distance of 1 for direct calls (i calls i+1)
    for i in range(n - 1):
        matrix[i][i + 1] = 1
    
    # Apply the Floyd-Warshall algorithm to compute the shortest distance between methods
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if matrix[i][k] != -1 and matrix[k][j] != -1:  # Check if there is a path through k
                    # Update distance if a shorter path is found
                    if matrix[i][j] == -1 or matrix[i][j] > matrix[i][k] + matrix[k][j]:
                        matrix[i][j] = matrix[i][k] + matrix[k][j]
    
    return matrix

def generate_call_dependency_matrix_negative(method_names):
    #matrix = generate_call_dependency_matrix(method_names)
    n = len(method_names)
    # Initialize the matrix as a list of lists (n x n), filled with -1 (no calls)
    n = len(method_names)
    matrix = [[-1] * n for _ in range(n)]
    
    # Distance of 0 for each method calling itself
    for i in range(n):
        matrix[i][i] = 0
    
    # Distance of 1 for direct calls (i calls i+1)
    for i in range(n - 1):
        matrix[i][i + 1] = 1

    return floyd_warshall_with_arbitrary_negative_distances(matrix)

def floyd_warshall_with_arbitrary_negative_distances(matrix):
    """Modified Floyd-Warshall to track arbitrary negative distances."""
    n = len(matrix)  # Number of methods (nodes)
    distance_matrix = [row[:] for row in matrix]  # Deep copy of the matrix
    
    # First, propagate direct and indirect distances (like Floyd-Warshall)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if distance_matrix[i][k] >= 0 and distance_matrix[k][j] >= 0:
                    if distance_matrix[i][j] == -1:  # No path yet
                        distance_matrix[i][j] = distance_matrix[i][k] + 1  # Indirect call
                    elif distance_matrix[i][j] > distance_matrix[i][k] + 1:
                        distance_matrix[i][j] = distance_matrix[i][k] + 1  # Shorten path if possible

    # Now propagate negative distances for unreachable nodes
    # Start with -1 for unreachable nodes and propagate negative distances further
    changed = True
    while changed:
        changed = False
        for i in range(n):
            for j in range(n):
                if distance_matrix[i][j] == -1:  # No call path found
                    for k in range(n):
                        if distance_matrix[i][k] < 0 and distance_matrix[k][j] < 0:
                            # Increment negative distance if both are unreachable
                            new_distance = min(distance_matrix[i][k] - 1, distance_matrix[k][j] - 1)
                            if distance_matrix[i][j] == -1 or distance_matrix[i][j] > new_distance:
                                distance_matrix[i][j] = new_distance
                                changed = True

    return distance_matrix

def transform_negative_cells(matrix, add_value):
    """
    Adds a fixed number to all negative cells in a 2D matrix and then reverses their sign.
    
    Parameters:
    matrix (list of lists of int/float): The input 2D matrix.
    add_value (int/float): The fixed number to add to each negative cell.
    
    Returns:
    list of lists of int/float: The transformed 2D matrix.
    """
    # Create a transformed matrix
    transformed_matrix = []
    
    # Process each cell in the matrix
    for row in matrix:
        transformed_row = []
        for cell in row:
            # Check if the cell is negative
            if cell < 0:
                # Add the fixed number and reverse the sign
                transformed_cell = -(cell + add_value)
            else:
                # Keep positive values as they are
                transformed_cell = cell
            # Append the transformed cell to the transformed row
            transformed_row.append(transformed_cell)
        # Append the transformed row to the transformed matrix
        transformed_matrix.append(transformed_row)
    
    return transformed_matrix



def generate_call_questions_with_distances(method_names, distance_matrix):
    """Generate questions and distances for all pairs of methods."""
    questions_with_distances = []
    num_methods = len(method_names)

    for i in range(num_methods):
        for j in range(num_methods):
            if i != j:
                question = f"Does `{method_names[i]}` call `{method_names[j]}`, either directly or indirectly? Think step-by-step by following the method calls from `{method_names[i]}.`."
                distance = distance_matrix[i][j]
                questions_with_distances.append((question, distance))
    
    return questions_with_distances

def generate_call_questions_with_distances_and_chains(method_names, distance_matrix):
    """Generate questions, distances, and chains for all pairs of methods."""
    questions_with_distances_and_chains = []
    num_methods = len(method_names)

    for i in range(num_methods):
        for j in range(num_methods):
            if i != j:
                # Generate the question
                question = (
                    f"Does `{method_names[i]}` call `{method_names[j]}`, either directly or indirectly? "
                    f"Think step-by-step by following the method calls from `{method_names[i]}.`"
                )
                distance = distance_matrix[i][j]
                
                # Generate the chain from i to j if i < j, or from i to the end if i > j
                if i < j:
                    chain = method_names[i:j + 1]
                    distance = len(chain) - 1
                else:
                    chain = method_names[i:]
                    distance = - (len(chain) - 1)
                start_back_chain = max(0, i - len(chain))
                end_back_chain = i
                back_chain = method_names[start_back_chain: end_back_chain]
                
                # Add the question, distance, and chain to the output list
                questions_with_distances_and_chains.append((question, distance, chain, back_chain))
    
    return questions_with_distances_and_chains


def print_matrix(matrix, method_names):
    # Print the matrix with the method names as headers
    print("        " + "  ".join(method_names))
    for i, row in enumerate(matrix):
        print(f"{method_names[i]}  " + "  ".join([str(val) for val in row])) 


def select_n_of_distance(questions_with_distances_maybe_chain, distance, n):
    """Select up to `n` questions with a specified `distance`."""
    # Filter questions by the specified distance
    filtered_questions = list(
        filter(lambda qd: qd[1] == distance, questions_with_distances_maybe_chain)
    )
    # Randomly sample up to `n` questions from the filtered list
    return random.sample(filtered_questions, min(n, len(filtered_questions)))

def select_n_for_all_distances(questions_with_distances_maybe_chain, n):
    """Select up to `n` questions for each unique distance value."""
    # Group questions by distance
    distance_groups = {}
    for question in questions_with_distances_maybe_chain:
        dist = question[1]
        if dist not in distance_groups:
            distance_groups[dist] = []
        distance_groups[dist].append(question)
    
    # Select up to `n` questions for each distance
    selected_questions = []
    for dist, questions in distance_groups.items():
        selected_questions.extend(random.sample(questions, min(n, len(questions))))
    
    return selected_questions


def select_n_up_to_max_distance(questions_with_distances_maybe_chain, max_distance, n):
    """Select up to `n` questions with distances less than or equal to `max_distance`, positive or negative."""
    def dist_check(n):
        return n != 0 and abs(n) <= max_distance
     # Filter questions by the specified distance
    filtered_questions = list(
        filter(lambda qd: dist_check(qd[1]), questions_with_distances_maybe_chain)
    )

    # Randomly sample up to `n` questions from the filtered list
    return select_n_for_all_distances(filtered_questions, n)


# Example usage
#class_body = generate_class_with_chained_methods("MyRandomClass", 15, 6)
#print(class_body)

# Generate a list of 10 unique method names
#method_names = generate_unique_method_names(10)
#for name in method_names:
#    print(name)


def divide_list_into_sublists(lst, m):
    """
    Divides a list into sublists of a specified size.
    
    Parameters:
    lst (list): The input list to be divided.
    m (int): The number of elements each sublist should have.
    
    Returns:
    list of lists: A list containing the sublists.
    """
    # Create sublists of m elements each
    return [lst[i:i + m] for i in range(0, len(lst), m)]

def divide_list_with_max(lst, max_size):
    result = []
    start_index = 0
    sublist_length = 1
    
    # Keep creating sublists until we've processed the entire list
    while start_index < len(lst):
        if sublist_length < max_size:
            # Create sublists of increasing size
            end_index = start_index + sublist_length
            result.append(lst[start_index:end_index])
            start_index = end_index
            sublist_length += 1
        else:
            # Once we reach max_size, create one big list with the rest of the elements
            result.append(lst[start_index:])
            break
    
    return result

def sum_consecutive_integers(n):
    return n * (n + 1) // 2

from collections import defaultdict

def count_distances(tuples_list):
    count_dict = defaultdict(int)
    
    # Iterate over each tuple in the list
    for item in tuples_list:
        # Access the last element of the tuple (item[-1])
        dist = item[1]
        # Increment the count for this last item
        count_dict[dist] += 1
    
    return count_dict

# we will lose some questions with a short distance: we will have at most n_chains negative questions in a given run
def generate_class(n_method, max_dist, q_per_dist):
    # Generate unique random method names
    #if (n_method < (max_dist * q_per_dist)):
    #        n_method = (max_dist * q_per_dist)
    n_method = max(n_method, sum_consecutive_integers(q_per_dist+2))
    method_names = set()
    while len(method_names) < n_method:
        #method_names.add(generate_random_method_name(method_length))
        method_names.add(generate_random_java_method_name())
    method_names = list(method_names)
    # we need n chains so that there n "ends of chains"
    #all_chains = divide_list_into_sublists(method_names, q_per_dist)
    all_chains = divide_list_with_max(method_names, q_per_dist + 1)
    the_class = generate_class_with_multiple_chains("MyClass", all_chains)
    print(the_class)
    print("\n\n\n\n\n")
    all_qs = []
    for chain_names in all_chains:
        matrix = generate_call_dependency_matrix_negative(chain_names)
        #print_matrix(matrix, chain_names)
        matrix = transform_negative_cells(matrix, len(chain_names))
        print_matrix(matrix, chain_names)
        print("\n\n\n\n\n")
        questions = generate_call_questions_with_distances_and_chains(chain_names, matrix)
        all_qs.append(questions)
    #for q in questions:
    #    print(q)
    #print("\n\n\n\n\n")

    all_qs = flatten_list(all_qs)

    selection = select_n_up_to_max_distance(all_qs, max_dist, q_per_dist)
    for q in selection:
        print(q)

    print(len(selection))
    print(count_distances(selection))
    print(n_method)
    #write_prompt_to_file(prompt_start, the_class, prompt_end, "../llama.cpp-master/system.txt")
    write_class_to_file(the_class, "../llama.cpp-master/theClass.java")
    write_prompt_to_file(p.in_context, the_class, "../llama.cpp-master/system.txt")
    write_questions_to_file(selection, "../llama.cpp-master/reachability_questions.txt")
    write_chains_to_file(selection,  "../llama.cpp-master/chains.txt")
    write_methods_to_file(method_names,  "../llama.cpp-master/methods.txt")
    print("./reachability-bench -m ../models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf -ns 60 -np 12 -b 20000 -c 20000 -f reachability_questions.txt")




def write_questions_to_file(questions_with_distances, filename):
    """Writes the questions to a file, one per line."""
    with open(filename, 'w') as f:
        for question, dist, *rest in questions_with_distances:
            # Write only the question (ignoring the distance) to the file
            f.write(f"{dist}\t{question}\n")

def write_class_to_file(body, filename):
    """Writes the questions to a file, one per line."""
    with open(filename, 'w') as f:
        f.write(body)


def write_prompt_to_file(prompt, body, filename):
    """Writes the questions to a file, one per line."""
    with open(filename, 'w') as f:
        f.write(prompt["start"])
        f.write(body)
        f.write(prompt["end"])

def write_chains_to_file(questions, filename):
    """Write all chains from the questions list to a file, one per line."""
    with open(filename, 'w') as file:
        for question, dist, chain, back_chain in questions:
            file.write(" ".join(chain) + '\t' + " ".join(back_chain) + '\n')  # Write each chain pair on a new line, tab separated

def write_methods_to_file(methods, filename):
    """Write all chains from the questions list to a file, one per line."""
    with open(filename, 'w') as file:
        file.write(" ".join(methods))  # Write each chain pair on a new line, tab separated


### code is buggy, this is way too big ...

### read and integrate the discussion about kv size and continous batching:
# https://github.com/ggerganov/llama.cpp/discussions/4130


def generate_all(exp_name, context_size, depths, n_questions, n_pad):
    chain_size = max(depths) + n_pad
    n_methods_needed = chain_size * n_questions
    print("number of methods needed: ", n_methods_needed)
    n_chains_in_context = context_size // chain_size
    n_questions_left = n_questions
    dir = Path(exp_name)
    dirs = []
    while n_questions_left > 0:
        exp_dir = dir / f"{n_questions - n_questions_left}--{n_questions - n_questions_left + n_chains_in_context}"
        if not exp_dir.exists():
            exp_dir.mkdir(parents=True, exist_ok=True) 
        # generate n_chains_in_context questions per depth
        n_qs = n_chains_in_context if n_chains_in_context < n_questions_left else n_questions_left
        print("generate a context size of ", context_size, " for ", n_qs * len(depths), " questions, with ", n_chains_in_context, " chains")
        generate_class_new(exp_dir, context_size, n_chains_in_context, chain_size,  depths, n_qs) # + chain_size + exp name
        dirs.append(exp_dir)
        n_questions_left -= n_chains_in_context
    return dirs


# we will lose some questions with a short distance: we will have at most n_chains negative questions in a given run
def generate_class_new(directory, context_size, n_chains, chain_size, depths, n_questions):
    # Generate unique random method names
    print("generating a class with:")
    print("methods: ", context_size)
    print("how many chains: ", n_chains)
    print("number of questions per depth: ", n_questions)
    print("expected total number of questions:", n_questions * len(depths))
    method_names = generate_unique_method_names(context_size)
    # we need n chains so that there n "ends of chains"
    #all_chains = divide_list_into_sublists(method_names, q_per_dist)
    all_chains = divide_list_into_sublists(method_names, chain_size)
    print("actual number of chains: ", len(all_chains))
    the_class = generate_class_with_multiple_chains("MyClass", all_chains)
    #print(the_class)
    #print("\n\n\n\n\n")
    all_qs = []
    for chain_names in all_chains:
        questions = generate_call_questions_with_distances_and_chains_new(chain_names)
        all_qs.append(questions)
    #for q in questions:
    #    print(q)
    #print("\n\n\n\n\n")

    all_qs = flatten_list(all_qs)

    selection = []
    for d in depths:
        selection.extend(select_n_of_distance(all_qs, d, n_questions))

    #for q in selection:
    #    print(q)

    print("actual number of questions: ", len(selection))
    print(count_distances(selection))
    #print(n_method)
    dir = Path(directory)


    #write_prompt_to_file(prompt_start, the_class, prompt_end, "../llama.cpp-master/system.txt")
    write_class_to_file(the_class,  dir / "theClass.java")
    write_prompt_to_file(p.in_context, the_class, dir / "system.txt")
    write_questions_to_file(selection, dir / "reachability_questions.txt")
    write_chains_to_file(selection, dir / "chains.txt")
    write_methods_to_file(method_names,  dir / "methods.txt")
    print("./reachability-bench -m ../models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf -ns 60 -np 12 -b 20000 -c 20000 -f reachability_questions.txt")


def generate_call_questions_with_distances_and_chains_new(method_names):
    """Generate questions, distances, and chains for all pairs of methods."""
    questions_with_distances_and_chains = []
    num_methods = len(method_names)

    for i in range(num_methods):
        for j in range(num_methods):
            if i != j:
                # Generate the question
                question = (
                    f"Does `{method_names[i]}` call `{method_names[j]}`, either directly or indirectly? "
                    f"Think step-by-step by following the method calls from `{method_names[i]}.`"
                )

                # Generate the chain from i to j if i < j, or from i to the end if i > j
                if i < j:
                    chain = method_names[i:j + 1]
                    distance = len(chain) - 1
                else:
                    chain = method_names[i:]
                    distance = - (len(chain) - 1)
                start_back_chain = max(0, i - len(chain))
                end_back_chain = i
                back_chain = method_names[start_back_chain: end_back_chain]
                
                # Add the question, distance, and chain to the output list
                questions_with_distances_and_chains.append((question, distance, chain, back_chain))
    
    return questions_with_distances_and_chains

dirs = generate_all("foo", 500, range(1,50), 50, 2)


# we need chains of at least length n, and we need at least n_questions of them for distance 1
# chain of length 10:
# Positive questions:
# - 1 q of length 10
# - 2 qs of length 9
# - 3 qs of length 8 
# ...
# - 9 qs of length 1
# negative questions
# - 1 q of each length!

# so we need n_questions times n chains
# given a context size of n, we can compute the number of chains that fit in 1 "batch"
# else we generate more batches.

# if n * n_q > context
# then we generate more batches, that's it?
# we want to accomodate some backwards calls as well

# there is some "Critical values" when the number of questions is high, context size is low
# in that case it's better to just generate individual contexts, and not a full shared body?