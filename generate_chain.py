import random
import string
import prompts as p
from pathlib import Path
from collections import defaultdict
import control_flow_2
import big_comments


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

def flatten_list(lst):
    return [item for sublist in lst for item in sublist]


def generate_class_with_multiple_chains(class_name, chains, chain_generator):
    # Generate the chain of method calls
    method_bodies = []
    for c in chains:
        #method_bodies.append(big_comments.generate_chained_method_calls(c))
        method_bodies.append(chain_generator(c))

    method_bodies = flatten_list(method_bodies)
    # Shuffle the method bodies to create random order in the class
    random.shuffle(method_bodies)

    # Construct the class with shuffled method bodies
    class_body = f"public class {class_name} {{\n"
    class_body += "\n\n".join(method_bodies)
    class_body += "\n}"

    return class_body

def select_n_of_distance(questions_with_distances_maybe_chain, distance, n):
    """Select up to `n` questions with a specified `distance`."""
    # Filter questions by the specified distance
    filtered_questions = list(
        filter(lambda qd: qd[1] == distance, questions_with_distances_maybe_chain)
    )
    # Randomly sample up to `n` questions from the filtered list
    return random.sample(filtered_questions, min(n, len(filtered_questions)))

def divide_list_into_sublists(lst, m):
    """Divides a list into sublists of a specified size."""
    # Create sublists of m elements each
    return [lst[i:i + m] for i in range(0, len(lst), m)]

def count_distances(tuples_list):
    count_dict = defaultdict(int)
    
    # Iterate over each tuple in the list
    for item in tuples_list:
        # Access the last element of the tuple (item[-1])
        dist = item[1]
        # Increment the count for this last item
        count_dict[dist] += 1
    
    return count_dict

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

def generate_class_new(directory, context_size, n_chains, chain_size, depths, n_questions, chain_generator):
    # Generate unique random method names
    print("generating a class with:")
    print("methods: ", context_size)
    print("how many chains: ", n_chains)
    print("number of questions per depth: ", n_questions)
    print("expected total number of questions:", n_questions * len(depths))

    method_names = generate_unique_method_names(context_size)
    all_chains = divide_list_into_sublists(method_names, chain_size)
    print("actual number of chains: ", len(all_chains))
    the_class = generate_class_with_multiple_chains("MyClass", all_chains, chain_generator)
    
    all_qs = []
    for chain_names in all_chains:
        questions = generate_call_questions_with_distances_and_chains_new(chain_names)
        all_qs.append(questions)
    all_qs = flatten_list(all_qs)
    selection = []
    for d in depths:
        selection.extend(select_n_of_distance(all_qs, d, n_questions))
        selection.extend(select_n_of_distance(all_qs, -d, n_questions))
    print("actual number of questions: ", len(selection))
    print(count_distances(selection))

    dir = Path(directory)
    write_class_to_file(the_class,  dir / "theClass.java")
    write_prompt_to_file(p.in_context, the_class, dir / "system.txt")
    write_questions_to_file(selection, dir / "reachability_questions.txt")
    write_chains_to_file(selection, dir / "chains.txt")
    write_methods_to_file(method_names,  dir / "methods.txt")


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

                # Recover the chain from i to j if i < j, or from i to the end if i > j
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

def method_generator(c):
    big_comments.generate_chained_method_calls(c)

def generate_all(exp_name, context_size, depths, n_questions, n_pad, n_comment_lines=0):
    chain_size = max(depths) + n_pad
    n_methods_needed = chain_size * n_questions
    print("number of methods needed: ", n_methods_needed)
    n_chains_in_context = context_size // chain_size
    n_questions_left = n_questions
    dir = Path(exp_name)
    dirs = []
    while n_questions_left > 0:
        depth_str = f"{depths[0]}--{depths[-1]}" if len(depths) > 8 else "_".join(str(num) for num in depths)
        q_start = (n_questions - n_questions_left) * len(depths) * 2
        # this index is wrong, it's one full step when for the last iteration it should be partial
        q_end = (n_questions - n_questions_left + n_chains_in_context)  * len(depths) * 2
        exp_dir = dir / f"ctx_{context_size}_depths_{depth_str}_com_{n_comment_lines}_qs_{q_start}--{q_end}"
        if not exp_dir.exists():
            exp_dir.mkdir(parents=True, exist_ok=True) 
        # generate n_chains_in_context questions per depth
        n_qs = n_chains_in_context if n_chains_in_context < n_questions_left else n_questions_left
        print("generate a context size of ", context_size, " for ", n_qs * len(depths), " questions, with ", n_chains_in_context, " chains")
        print("in: ", exp_dir)
        chain_generator = lambda c: big_comments.generate_chained_method_calls(c, n_comment_lines)
        generate_class_new(exp_dir, context_size, n_chains_in_context, chain_size,  depths, n_qs, chain_generator) # + chain_size + exp name
        dirs.append(exp_dir)
        n_questions_left -= n_chains_in_context
    return dirs

def write_slurm(name, path, time):
    contents = f"""#! /usr/bin/bash
#SBATCH --job-name=reachability-{name}
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --constraint=h100
#SBATCH --ntasks-per-node=1
#SBATCH --account=spk@h100
#SBATCH --time={time}
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=romain.robbes@labri.fr

module load arch/h100
module load cuda/12.8.0
cd $WORK
python run_experiment.py {path}"""
    with open(f'reachability-{name}.slurm', 'w') as file:
        file.write(contents)

def generate_really_all():
    def write_exps(context_ranges, n_comments):
        for context_size in context_ranges:
            depth_ranges = range(1,11) #1 to 10
            n_questions = 200
            n_padding = 0
            name = f'context_{context_size}_comments_{n_comments}' 
            exp_path = f'xps/{name}'
            generate_all(exp_path, context_size, depth_ranges, n_questions, n_padding, n_comments)
            write_slurm(name, exp_path, '1:00:00')

    ranges_0_2 = [50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000]
    ranges_4 = [50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    ranges_7 = [50, 75, 100, 150, 200, 250, 300, 350]
    ranges_12 = [50, 75, 100, 150, 200]
    ranges_24 = [100]
    write_exps(ranges_24,2)




"""
dirs = generate_all("small-ext", 30, range(8,13), 100, 2)
dirs = generate_all("smallish-ext", 50, range(8,13), 100, 2)
dirs = generate_all("medium-ext", 75, range(8,13), 100, 2)
dirs = generate_all("medium-plus-ext", 100, range(8,13), 100, 2)
dirs = generate_all("medium-plus-plus-ext", 125, range(8,13), 100, 2)
dirs = generate_all("medium-large-ext", 150, range(8,13), 100, 2)
dirs = generate_all("medium-large-plus-ext", 175, range(8,13), 100, 2)
dirs = generate_all("largish-ext", 200, range(8,13), 100, 2)
dirs = generate_all("largish-plus-ext", 250, range(8,13), 100, 2)
dirs = generate_all("very-large-ext", 300, range(8,13), 100, 2)
dirs = generate_all("very-very-large-ext", 350, range(8,13), 100, 2)
dirs = generate_all("huge-ext", 400, range(8,13), 100, 2)
"""

"""
dirs = generate_all("small-flow-ext", 30, range(8,13), 100, 2)
dirs = generate_all("smallish-flow-ext", 50, range(8,13), 100, 2)
dirs = generate_all("medium-flow-ext", 75, range(8,13), 100, 2)
dirs = generate_all("medium-plus-flow-ext", 100, range(8,13), 100, 2)
dirs = generate_all("medium-plus-plus-flow-ext", 125, range(8,13), 100, 2)
dirs = generate_all("medium-large-flow-ext", 150, range(8,13), 100, 2)
dirs = generate_all("medium-large-plus-flow-ext", 175, range(8,13), 100, 2)
dirs = generate_all("largish-flow-ext", 200, range(8,13), 100, 2)
dirs = generate_all("largish-plus-flow-ext", 250, range(8,13), 100, 2)
dirs = generate_all("very-large-flow-ext", 300, range(8,13), 100, 2)
dirs = generate_all("very-very-large-flow-ext", 350, range(8,13), 100, 2)
dirs = generate_all("huge-flow-ext", 400, range(8,13), 100, 2)
dirs = generate_all("medium-large-plus-flow", 175, range(1, 8), 100, 2)
dirs = generate_all("largish-flow", 200, range(1, 8), 100, 2)
dirs = generate_all("largish-plus-flow", 250, range(1, 8), 100, 2)
dirs = generate_all("very-large-flow", 300, range(1, 8), 100, 2)
dirs = generate_all("very-very-large-flow", 350, range(1, 8), 100, 2)
dirs = generate_all("huge-flow", 400, range(1, 8), 100, 2)
"""
#dirs = generate_all("test/ctx-80", 80, range(1, 11), 200, 0, 0)

generate_really_all()

#generate_all("small-comment", 30, range(1,8), 100, 2)
#generate_all("smallish-comment", 50, range(1,8), 100, 2)
#generate_all("medium-comment", 75, range(1,8), 100, 2)
#generate_all("medium-plus-comment", 100, range(1,8), 100, 2)
#generate_all("medium-plus-plus-comment", 125, range(1,8), 100, 2)
#generate_all("medium-large-comment", 125, range(1,8), 100, 2)




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


### read and integrate the discussion about kv size and continous batching:
# https://github.com/ggerganov/llama.cpp/discussions/4130
