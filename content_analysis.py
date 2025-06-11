import os
import sys
from pathlib import Path
from collections import defaultdict
import shutil


def is_camel_case(s):
    # Matches a string that starts with a lowercase letter and contains at least one uppercase letter
    return bool(re.match(r'^[a-z]+[A-Za-z]*[A-Z]+[A-Za-z]*$', s))

def split_camel_case(s):
    """Split camel-case words into lowercase subwords, filtering out common prepositions."""
    common_prepositions = {"in", "on", "at", "to", "for", "by", "with", "about", "against", 
                           "between", "into", "through", "during", "before", "after", "above", 
                           "below", "from", "up", "down", "over", "under", "again", "further", 
                           "then", "once", "of", "off"}
    #print("split camel case of ", s)
    words = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z])|[A-Z]+$', s)
    filtered_words = {word.lower() for word in words if word.lower() not in common_prepositions}
    
    return filtered_words

def camel_case_intersection(single_word, word_list):
    # Split the single camel case word
    #print("single word is ", single_word)
    #print("word list is ", word_list)
    subwords = set(split_camel_case(single_word))
    
    # Split each word in word_list and flatten to get all subwords
    list_subwords = set()
    for word in word_list:
        list_subwords.update(split_camel_case(word))
    
    # Find the intersection of subwords
    intersection = subwords.intersection(list_subwords)
    
    return list(intersection)


def associate_files_with_lines(results_directory, chains_file):
    """Associate each result file with a line from the other file based on 'x' in the filename."""
    
    # Read all lines from the other file
    with open(chains_file, 'r') as f:
        chains_file_lines = f.readlines()

    # Dictionary to store the associations
    associations = {}

    print("Results dir: ", results_directory)
    print("Chains file dir: ", chains_file)
    # Loop through the files in the results directory
    for filename in os.listdir(results_directory):
        if filename.endswith(".txt"):  # Only consider .txt files
            # Extract 'x' from the filename (before the first underscore)
            x_value = int(filename.split('_')[0].replace('result', ''))  # Extract 'x' value
            
            # Ensure there are enough lines in the other file to match 'x'
            if 0 <= x_value < len(chains_file_lines):
                # Get the corresponding line from the other file
                line_from_other_file = chains_file_lines[x_value].strip()  # Line is 1-based
                # Associate the filename with the line
                associations[filename] = line_from_other_file
            else:
                print(f"Warning: No matching line for {filename} (x = {x_value})")
    
    return associations

import re

def analyze_words_in_file(file_path, right_words, back_words, all_words, physical_methods):
    # Step 1: Remove the right words from the list of all words to get the wrong words
    wrong_words = [word for word in all_words if word not in right_words]

    # Compile a regular expression pattern to remove unwanted punctuation
    # This pattern removes any non-alphanumeric character (e.g., quotes, backquotes, parentheses, etc.)
    pattern = re.compile(r'[^\w\s]')

    # Step 2: Read the content of the file
    try:
        with open(file_path, 'r') as file:
            content = file.readlines()[1:] # omit first line
            print("\n".join(content))
            content = " ".join(content).split()  # Split the content into words

        # Step 3: Clean and count the number of right words in the file content
        right = []
        wrong = []
        back = []
        hallucinated = []
        intersection = []

        # Step 4: Iterate through each word in the content
        for word in content:
            # Clean up the word by removing unwanted punctuation
            cleaned_word = pattern.sub('', word) # Remove punctuation
            if is_camel_case(cleaned_word):

                # Count the right words
                if cleaned_word in right_words:
                    right.append(cleaned_word)
                    print("RIGHT: ", cleaned_word)
                elif cleaned_word in back_words:
                    back.append(cleaned_word)
                    print("BACK: ", cleaned_word)
                # Count the wrong words
                elif cleaned_word in wrong_words:
                    wrong.append(cleaned_word)
                    print("WRONG: ", cleaned_word)
                    print("DISTANCE: ", min_word_distance(physical_methods, cleaned_word, right_words))
                    inter = camel_case_intersection(cleaned_word, right_words)
                    if inter:
                        print("+ INTERSECTION:", inter)
                        intersection.extend(inter)
                else: 
                    hallucinated.append(cleaned_word)
                    print("HALLUCINATE: ", cleaned_word)
                    inter = camel_case_intersection(cleaned_word, right_words)
                    if inter:
                        print("+ INTERSECTION:", inter)
                        intersection.extend(inter)

            #else: 
            #    print("not camel case: ", word)

        # Step 5: Return the counts
        return len(right), len(wrong), len(back), len(hallucinated), len(intersection)

    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return 0, 0
    
def remove_consecutive_duplicates(lst):
    # Create an empty list to store the result
    result = []
    
    # Iterate over the list and add the first element to the result
    for i in range(len(lst)):
        if i == 0 or lst[i] != lst[i-1]:  # Check if the current element is different from the previous
            result.append(lst[i])
    
    return result

def convert_to_consecutive_pairs(lst): return list(zip(lst, lst[1:]))

def extract_answer(content):
    last_20_chars = content[-20:] if len(content) > 20 else content


    # Determine the file's status based on the content
    if "FINAL ANSWER: YES" in content or "YES" in last_20_chars:
        return "YES"
    elif "FINAL ANSWER: NO" in content or "NO" in last_20_chars:
        return "NO"
    else:
        return "NOT FOUND"

def find_legit_call(calls, origin):
    """Finds the first tuple in a list where the first element matches the target.
    
    Args:
        lst (list of tuple): The list of tuples to search.
        target: The value to match with the first element of each tuple.
    
    Returns:
        tuple: The first matching tuple, or None if no match is found.
    """
    #print("find legit calls ", calls, " origin: ", origin)
    for call in calls:
        if call[0] == origin:
            #print("find legit calls returning: ", call[1])
            return call[1]
    #print("find legit calls returning NONE")
    return None


legit = "GOOD"
expected = "EXPECTED" 
repeat = "REPEATS OLD CALL"  
backwards = "BACKWARDS CALL"
backtrack_first = "BACKTRACK FIRST CALL"
backtrack_other = "BACKTRACK OTHER"
backwards_first = "BACKWARDS (FIRST)"
backwards_last = "BACKWARDS (LAST)"
off1 = "OFF BY ONE"
near = "NEAR MISS"
far = "FAR MISS"
extra = "EXTRA CALL"
bad = "BAD"
badback = "BACK SIMILAR:"
hallucinated = "HALLUCINATED"
similar = "~= "
close_back = "CLOSE_BACK"
far_back = "FAR_BACK"
first = " 1st!"


def analyze_reasoning(file_path, right_chain, back_chain, all_methods, physical_methods):
    # Compile a regular expression pattern to remove unwanted punctuation
    # This pattern removes any non-alphanumeric character (e.g., quotes, backquotes, parentheses, etc.)
    pattern = re.compile(r'[^\w\s]')
    print(file_path)

    # Step 2: Read the content of the file
    try:
        with open(file_path, 'r',  encoding='latin-1') as file:
            content = file.readlines() # omit first line
            question = content[0]
            content = " ".join(content[1:])
            answer = extract_answer(content)

        camel_q = []
         # Step 4: Iterate through each word in the content
        for word in question.split():
            # Clean up the word by removing unwanted punctuation
            cleaned = pattern.sub('', word) # Remove punctuation
            if is_camel_case(cleaned):
                        camel_q.append(cleaned)
        start = camel_q[0]
        end = camel_q[1]
        words = content.split()  # Split the content into words
        camels = []
        # Step 4: Iterate through each word in the content
        for word in words:
            # Clean up the word by removing unwanted punctuation
            cleaned_word = pattern.sub('', word) # Remove punctuation
            if is_camel_case(cleaned_word):
                camels.append(cleaned_word)
        camels = remove_consecutive_duplicates(camels)
        calls = convert_to_consecutive_pairs(camels)
        all_calls = convert_to_consecutive_pairs(all_methods)
        call_chain = convert_to_consecutive_pairs(right_chain)
        past_calls = []
        results = []
        # we remove the target call from the back chain
        back_calls = convert_to_consecutive_pairs(all_methods[::-1])
        # add the start method as starting point, then reverse
        close_back_calls = convert_to_consecutive_pairs((back_chain + [right_chain[0]])[::-1])
        final_calls = [(start, end),(end, start)] if len(right_chain) > 2 else []
        for call in calls:
            if call in final_calls:
                continue
            res = []
            if call in all_calls:
                # this call exists
                res.append(legit)
                if call_chain and call == call_chain[0]:
                    # this is the call we expect
                    res.append(expected)
                    # we expect the next one in the future
                    past_calls.append(call_chain.pop(0))
                elif call in past_calls:
                    # this is a call in a call chain, but we have already encountered it
                    res.append(repeat)
                elif call[0] in back_chain and call[1] in right_chain:
                    # dunno if we should be more precise here
                    res.append(backtrack_first)
                elif call[0] in back_chain or call[1] in back_chain:
                    res.append(backtrack_other)
                else: 
                    # this is a call to the class, but how far is it from the expected one?
                    if call_chain:
                        target = call[1]
                        actual = call_chain[0][1]
                        distance = word_distance(physical_methods, target, actual)
                        if distance == 1:
                            res.append(off1)
                        elif distance <= 3:
                            res.append(near)
                        else:
                            inter = camel_case_intersection(target, [actual])
                            if inter:
                                res.append(similar + " ".join(inter))
                                if first_in_list(physical_methods, target, actual):
                                    res[-1]+= first
                            else:
                                res.append(far)
                                if first_in_list(physical_methods, target, actual):
                                    res[-1]+= first
                    else:
                        # looks like we went through all the legit calls and still have some?
                        res.append(extra)
            elif call in back_calls:
                if first_in_list(physical_methods, call[0], call[1]):
                    res.append(backwards_first)
                else:
                    res.append(backwards_last)
                if call in close_back_calls:
                    res.append(close_back)
                else:
                    res.append(far_back)
            else: 
                # this call does not exist
                res.append(bad)
                
                if call[0] not in all_methods or call[1] not in all_methods:
                    # non-existent call
                    res.append(hallucinated)
                #elif call in back_calls:
                    # dunno if we should be more precise here
                #    res.append(backwards)
                else: 
                    actual = find_legit_call(all_calls, call[0])
                    backtual = find_legit_call(back_calls, call[0])
                    # this is a call to the class, but how far is it from the expected one?
                    #if call_chain:
                    target = call[1]
                    #    actual = call_chain[0][1]
                    distance = word_distance(physical_methods, target, actual)
                    distanceback = word_distance(physical_methods, target, actual)
                    if distance == 1:
                        res.append(off1)
                    elif distance <= 3:
                        res.append(near)
                    else:
                        inter = camel_case_intersection(target, [actual])
                        if inter:
                            res.append(similar + " ".join(inter))
                            if first_in_list(physical_methods, target, actual):
                                res[-1]+= first
                        else:
                            if backtual:
                                interback = camel_case_intersection(target, [backtual])
                                if interback:
                                    res.append(badback + " ".join(interback))
                                    if first_in_list(physical_methods, target, backtual):
                                        res[-1]+= first
                                else:
                                    res.append(far)
                            else:
                                res.append(far)
                    #else:
                        # looks like we went through all the legit calls and still have some?
                    #    res.append(extra)
                # - hallucinated?
                # - near-miss?
                # - big miss?
            #res.append(f"\t\t{call[0]}\t\t -> \t\t{call[1]}")
            res.append(call)
            results.append(res)
        return question, answer, results, content, right_chain
        
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return "", "", [], ""


def extract_class_definition(java_file, class_name):
    """
    Extracts a Java class definition from a string containing Java code.

    Args:
        java_code (str): The full Java code as a string.
        class_name (str): The name of the class to extract.

    Returns:
        str: The extracted class definition, or an empty string if not found.
    """
    with open(java_file, 'r') as f:
        java_code = f.read()
    # Search for the class declaration
    pattern = rf"\bclass\s+{class_name}\b"
    match = re.search(pattern, java_code)

    if not match:
        return ""  # Class not found
    
    # Find the starting index of the class declaration
    class_start = match.start()

    # Now, find the balance of the braces starting from the first '{'
    # Initialize counters for braces
    open_braces = 0
    class_code = ""

    # Search for the opening brace after the class declaration
    i = class_start
    while i < len(java_code) and java_code[i] != '{':
        i += 1

    if i == len(java_code):  # No opening brace found
        return ""

    # Now we are at the '{', begin balancing braces
    open_braces = 1  # We've seen one opening brace

    # Append to the class_code the character starting at the class declaration
    class_code += java_code[i]

    i += 1
    while i < len(java_code) and open_braces > 0:
        if java_code[i] == '{':
            open_braces += 1
        elif java_code[i] == '}':
            open_braces -= 1
        class_code += java_code[i]
        i += 1

    # Return the full class definition, with balanced braces
    return class_code.strip()

def extract_methods(java_code):
    """
    Extracts all the methods defined in a Java class, in order of appearance.
    
    Args:
        java_code (str): The Java code as a string.
    
    Returns:
        list: A list of method names, in the order they appear in the code.
    """
    # Regular expression to match method signatures
    # This matches method declarations (including access modifiers and return types)
    method_pattern = re.compile(r'\b\w+\s+\w+\s+(\w+)\s*\(.*\)\s*{')
    
    # Find all method names using the regex
    methods = method_pattern.findall(java_code)
    
    return methods

# should be renamed, it works on more than words
def word_distance(word_list, word1, word2):
    """Calculates the distance between two words in a list.
    
    Args:
        word_list (list of str): The list of words.
        word1 (str): The first word.
        word2 (str): The second word.
    
    Returns:
        int: The distance between the first occurrences of word1 and word2.
             Returns -1 if either word is not found in the list.
    """
    try:
        index1 = word_list.index(word1)
        index2 = word_list.index(word2)
        return abs(index1 - index2)
    except ValueError:
        # Return -1 if either word is not in the list
        return -1
    

def first_in_list(lst, item1, item2):
    """Returns the first of the two items that appears in the list.
    
    Args:
        lst (list): The list to search.
        item1: The first item to check.
        item2: The second item to check.
    
    Returns:
        The item that appears first in the list, or None if neither is found.
    """
    # Get the index of each item if they exist in the list, otherwise set to a large number
    index1 = lst.index(item1) if item1 in lst else float('inf')
    index2 = lst.index(item2) if item2 in lst else float('inf')

    # Determine which index is smaller, return corresponding item, or None if both are inf
    if index1 < index2:
        return item1
    elif index2 < index1:
        return item2
    else:
        return None

    
    
def min_word_distance(word_list, first_word, second_word_list):
    res = []
    for w in second_word_list:
        res.append(word_distance(word_list, first_word, w))
    return min(res)
    

def analysis(directory, chains, methods, physical_methods):
    with open(methods, 'r') as f:
        all_methods = f.read().split()
        associations = associate_files_with_lines(directory, chains)
        results = []
        for filename, associated_line in list(associations.items())[:]:
            rep = associated_line.replace('\t', '\t\t back chain:')
            print(f"File {filename} is associated with line: {rep}")
            chains = associated_line.split('\t')
            # weird ... should not happen but oh well
            if len(chains) < 2: chains.append("")
            chain, back_chain = chains
            chain_methods = chain.split()
            back_chain_methods = back_chain.split()
            result = analyze_words_in_file(os.path.join(directory, filename), chain_methods, back_chain_methods, all_methods, physical_methods)
            results.append(result)
            right, wrong, back, hallucinated, intersection = result
            print(f"Right words count: {right}")
            print(f"Back words count: {back}")
            print(f"Wrong words count: {wrong}")
            print(f"Hallucinated words count: {hallucinated}")
            print("\n\n\n\n\n\n")
        return results
    
   
def is_right_answer_for_dist(answer, dist): 
    return (dist > 0 and answer == "YES") or (dist < 0 and answer == "NO")

    
def answer_distance(filename):
    y_value = int(filename.split('_')[1].split('.')[0])
    return y_value

def check_answer(filename, answer):
    return is_right_answer_for_dist(answer, answer_distance(filename))
    # Extract the 'y' value from the filename (the second number in 'resultx_y.txt')
    #y_value = int(filename.split('_')[1].split('.')[0])  # Extract the second number from filename
    
    # Return True or False based on the conditions
    #if (y_value > 0 and answer == "YES") or \
    #   (y_value < 0 and answer == "NO"):
    #    return True
    #else:
    #    return False
 

def detailed_analysis(directory, chains, methods, physical_methods):
    with open(methods, 'r') as f:
        all_methods = f.read().split()
        associations = associate_files_with_lines(directory, chains)
        all_results = []
        for filename, associated_line in list(associations.items())[:]:
            chains = associated_line.split('\t')
            # weird ... should not happen but oh well
            if len(chains) < 2: chains.append("")
            chain, back_chain = chains
            chain_methods = chain.split()
            back_chain_methods = back_chain.split()
            result = analyze_reasoning(os.path.join(directory, filename), chain_methods, back_chain_methods, all_methods, physical_methods)
            question, answer, results, content, right_chain = result
            
            rep = associated_line.replace('\t', '\t\t back chain:')
            if False:
                print("\n\n\n\n\n==================")
                print(f"File {filename} is associated with line: {rep}")
                print(f"Question: {question}")
                print(f"Answer: {answer}")
                print("RIGHT" if check_answer(filename, answer) else "WRONG")
                print("=================")
                print(f"Reasoning:")
                for step in results:
                    print(f"{step[0]}\t{step[1]}\t{step[2]}")
                print("=================")
                print(content)
            augmented_result = answer_distance(filename), question, answer, results, content, right_chain 
            all_results.append(augmented_result)
        return all_results

def count_files_with_categories(results):
    """Count how many files have words in 'back' and 'wrong' categories."""
    back_count = 0
    wrong_count = 0
    all_right = 0
    hallucinated_count = 0
    intersection_count = 0

    for result in results:
        _, wrong, back, hallucinated, intersection = result  # Unpack the result tuple
        if back > 0:
            back_count += 1
        elif hallucinated > 0:
            hallucinated_count += 1
            if intersection > 0:
                intersection_count += 1
        elif wrong > 0:
            wrong_count += 1
            if intersection > 0:
                intersection_count += 1
        else:
            all_right += 1

    return all_right, back_count, wrong_count, hallucinated_count, intersection_count



directory = "/Users/rrobbes/Projects/reachability/llama.cpp-master/output/reachability_questions.txt_2024-11-07_14-59-24"

# analyses the output of 1 model
def detail_analysis_dir(experiment_dir, model_name):
    print("experiment dir: ", experiment_dir)
    batch_dirs = [d for d in experiment_dir.iterdir() if d.is_dir()]
    all_results = []
    batch_num = 0
    for batch_dir in batch_dirs:
        batch_num += 1
        chains = batch_dir / "chains.txt"
        methods = batch_dir / "methods.txt"
        class_def = batch_dir / "system.txt"
        class_file = batch_dir / "theClass.java"
        physical_methods = extract_methods(extract_class_definition(class_def, "MyClass"))
        subdirs = [d for d in batch_dir.iterdir() if d.is_dir()]
        for sub in subdirs:
            if model_name in str(sub):
                #print(f"match in {sub}")
                #print(chains)
                #
                results = detailed_analysis(sub, chains, methods, physical_methods)
                results_with_batch = [r + (batch_num, class_file) for r in results]
                all_results.extend(results_with_batch)
    return all_results
    



if len(sys.argv) > 1:
    directory = sys.argv[1]
else:
    print("default dir")


# make the files dependent on the dir 
directory = "../llama.cpp-master/output/reachability_questions.txt_2024-11-08_14-51-13"
chains = "../llama.cpp-master/chains.txt"
methods = "../llama.cpp-master/methods.txt"
class_def = "../llama.cpp-master/system.txt"

# then do something that generates outputs and aggregate them from subdirs to main dirs?





def filtered_fraction(list_a, num, predicate):
    # Filter list_a based on the predicate
    #print("len results", len(list_a))
    filtered_a = [element for element in list_a if predicate(element)]
    #print("len filtered", len(filtered_a))
    # Compute the fraction of the filtered list compared to the size of list_b
    if num == 0:
        return 0  # Avoid division by zero if list_b is empty
    fraction = len(filtered_a) / num
    #print("fraction: ", fraction)
    return fraction

def expected_call(r): return r[1] == expected
def good_call(r): return r[0] == legit
def bad_call(r): return r[0] == bad
def unexpected_call(r): return good_call(r) and not expected_call(r)
def hallucinated_call(r): return r[1] == hallucinated
def back_call(r): return "BACKWARDS" in r[1]
#def good_back_call(r): return good_call(r) and back_call(r)
def similar_call(r): return "SIMILAR" in r[1]
def far_miss_call(r): return "FAR_MISS" in r[1]
def near_miss_call(r): return r[1] == near or r[1] == off1
def first_call(r): return first in r[1]


def print_result(r):
    distance, question, answer, results, contents, call_chain = r
    print("distance: ", distance)
    print("question: ", question)
    print("answer: ", answer)
    print("results: ", results)
    print("right call chain: ", call_chain)
    amount_expected = filtered_fraction(results, abs(distance), lambda r: r [1] == expected)
    print("\namount expected: ", amount_expected)
    print("\n\ncontents: ", contents)

def write_result_to(result, filename):
    distance, question, answer, results, contents, call_chain = result
    amount_expected = filtered_fraction(results, abs(distance), lambda r: r [1] == expected)
    with open(filename, "w") as file:
        file.write("===========================\n")
        file.write(f"Question: {question}\n")
        file.write(f"Answer: {answer}\n")
        file.write(f"Distance: {distance}\n")
        file.write(f"Call chain: {call_chain}\n")
        file.write(f"Amount expected: {amount_expected}\n")
        file.write("===========================\n")
        file.write("Results:\n")
        for r in results:
            file.write(f"\t\t{r}\n")
        file.write("===========================\n")
        file.write(f"Contents: {contents}\n")

defright = "DEFINITELY_RIGHT"
probright = "PROBABLY_RIGHT"
border = "BORDERLINE"
probwrong = "PROBABLY_WRONG"
rightwrong = "RIGHT_FOR_WRONG_REASONS"
lost_track = "LOST_TRACK"

def classify_right_for_wrong_reasons(r):
    return "RIGHT FOR WRONG REASON"

def classify_right(r):
    distance, question, answer, results, contents, call_chain = r
    correct_steps = [r for r in results if expected_call(r)]
    extra_steps = len([r for r in results if not expected_call(r)])
    missing_steps = len(call_chain) -1 - len(correct_steps)
    if missing_steps == 0 and extra_steps <= 2:
        return "COMPLETELY RIGHT"
    if missing_steps == 0:
        return "RIGHT EXTRA STEPS"
    if missing_steps <= 2 and extra_steps <= 4:
        return "ALMOST RIGHT"
    # problem of too many right for wrong reason is here
    #if missing_steps <= 1 and extra_steps <= 10:
    #    return "MANY EXTRA STEPS"
    return classify_right_for_wrong_reasons(r)

def classify_reasoning(r):
    distance, question, answer, results, contents, call_chain = r
    correct_steps = [r for r in results if expected_call(r)]
    if len(correct_steps) / abs(distance) > 0.8:
        return "ALMOST THERE"
    if len(correct_steps) / abs(distance) > 0.6:
        return "ON TRACK"
    elif len(correct_steps) / abs(distance) > 0.3:
        return "LOST TRACK"
    return "WRONG TRACK"

def classify_mistakes(r):
    distance, question, answer, results, contents, call_chain = r
    good_calls = [r for r in results if good_call(r)]
    bad_calls = [r for r in results if bad_call(r)]
    if bad_calls:
        # the model made at least one bad call
        hallus = [r for r in bad_calls if hallucinated_call(r)]
        if hallus:
            return "HALLUCINATED CALLS"
        similar = [r for r in bad_calls if similar_call(r)]
        if similar:
            return "BAD SIMILAR CALL"
        near_misses = [r for r in bad_calls if near_miss_call(r)]
        if near_misses:
            return "BAD NEAR MISS"
        far_misses = [r for r in bad_calls if far_miss_call(r)]
        if far_misses:
            return "BAD FAR MISS"
        return "OTHER BAD CALL"
        
    else:
        # the model made only good calls
        back_calls = [r for r in good_calls if back_call(r)]
        if back_calls:
            return "BACKWARDS CALL"
        similar = [r for r in good_calls if similar_call(r)]
        if similar:
            return "GOOD SIMILAR CALL"
        near_misses = [r for r in good_calls if near_miss_call(r)]
        if near_misses:
            return "GOOD NEAR MISS"
        far_misses = [r for r in good_calls if far_miss_call(r)]
        if far_misses:
            return "GOOD FAR MISS"
        return "OTHER GOOD CALL"
        
        

#def classify_wrong(r):


def categorize_result(rr):
    r = rr[:-2] # removing the 2 extra things we added
    distance, question, answer, results, contents, call_chain = r
    if is_right_answer_for_dist(answer, distance):
        return "RIGHT", classify_right(r), classify_mistakes(r)
    elif answer == "NOT FOUND":
        return "CUT_OFF", classify_reasoning(r), classify_mistakes(r)
    else:
        return "WRONG", classify_reasoning(r), classify_mistakes(r)
    
#    distance, question, answer, results, contents, call_chain = r

#    amount_expected = filtered_fraction(results, abs(distance), lambda r: r [1] == expected)
#    print("amount expected: ", amount_expected)
#    if is_right_answer_for_dist(answer, distance):
#        if amount_expected > 0.9:
#            return defright
#        elif amount_expected > 0.6:
#            return probright
#        elif amount_expected > 0.4:
#            return border
#        elif amount_expected > 0.2:
#            return probwrong
#        else:
#            return rightwrong
#    else: 
#        if amount_expected > 0.4:
#            return lost_track
#        return "WRONG"


# Step 2: Categorize objects into a three-level hierarchy
def categorize_objects(objects):
    hierarchy = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for obj in objects:
        category, subcategory, sub_subcategory = categorize_result(obj)
        hierarchy[category][subcategory][sub_subcategory].append(obj)
    
    return hierarchy

# Step 3: Calculate and print statistics with super-category-relative percentages
def print_statistics(hierarchy):
       # Calculate the total number of objects for percentage calculations
    total_objects = sum(
        len(items) for categories in hierarchy.values()
        for subcategories in categories.values()
        for items in subcategories.values()
    )
    
    for category, subcategories in hierarchy.items():
        # Calculate total for each category
        category_count = sum(
            len(items) for sub_subcategories in subcategories.values()
            for items in sub_subcategories.values()
        )
        category_percentage = (category_count / total_objects) * 100
        print(f"'{category}': {category_count} ({category_percentage:.2f}%)")
       
        for subcategory, sub_subcategories in subcategories.items():
            # Total count for each subcategory within the current category
            subcategory_count = sum(len(items) for items in sub_subcategories.values())
            subcategory_percentage = (subcategory_count / category_count) * 100
            print(f"      '{subcategory}': {subcategory_count} ({subcategory_percentage:.2f}% of '{category}')")
            
            for sub_subcategory, items in sub_subcategories.items():
                # Count for each sub-subcategory within the current subcategory
                sub_subcategory_percentage = (len(items) / subcategory_count) * 100
                print(f"            '{sub_subcategory}': {len(items)}  ({sub_subcategory_percentage:.2f}% of '{subcategory}')")


def display_list_lengths(dict_of_lists):
    for key, value in dict_of_lists.items():
        print(f"{key}: {len(value)}")


def sort_results(results):
    hierarchy = categorize_objects(results)
    print_statistics(hierarchy)
    return hierarchy
    #for r in sorted[probright]:
    #    print_result(r)
    #    print("==============================")
    #    print("==============================")
    #    print("==============================")

def write_files(exp_dir, output_dir, write_dir, model_name, hierarchy):
       # Calculate the total number of objects for percentage calculations
    total_objects = sum(
        len(items) for categories in hierarchy.values()
        for subcategories in categories.values()
        for items in subcategories.values()
    )

    base_dir = exp_dir / output_dir / model_name
    write_base_dir = write_dir / exp_dir / model_name
    write_base_dir_code = write_dir / exp_dir 
    
    for category, subcategories in hierarchy.items():
        print(category)
        print(subcategories)
        # Calculate total for each category
        category_count = sum(
            len(items) for sub_subcategories in subcategories.values()
            for items in sub_subcategories.values()
        )
        category_percentage = (category_count / total_objects) * 100
        print(f"'{category}': {category_count} ({category_percentage:.2f}%)")
        catdir = f'{category.replace(" ","_")}-{category_count} ({category_percentage:.0f})'
       
        for subcategory, sub_subcategories in subcategories.items():
            # Total count for each subcategory within the current category
            subcategory_count = sum(len(items) for items in sub_subcategories.values())
            subcategory_percentage = (subcategory_count / category_count) * 100
            subcatdir = f'{subcategory.replace(" ","_")}-{subcategory_count} ({subcategory_percentage:.0f})'

            for sub_subcategory, items in sub_subcategories.items():
                # Count for each sub-subcategory within the current subcategory
                sub_subcategory_percentage = (len(items) / subcategory_count) * 100
                subsubcatcatdirdir = f'{sub_subcategory.replace(" ","_")}-{len(items)} ({sub_subcategory_percentage:.0f})'
                dir = write_base_dir / catdir / subcatdir / subsubcatcatdirdir
                dir.mkdir(parents=True, exist_ok=True)
                print("dir will be: ", dir)
                seen = 0
                for res in items:
                    seen += 1
                    # Write the text representation to a file
                    # value[0] is the distance
                    *result, batch_num, code_file_path = res
                    batch_code_file_name = f"theClass-{batch_num}.java"
                    batch_code_path = write_base_dir_code / batch_code_file_name
                    print("base code path is: ", batch_code_path)
                    print("code file path is: ", code_file_path)
                    if not batch_code_path.exists():
                        # copying source file to the path
                        shutil.copy2(code_file_path, batch_code_path)
                    result_file_name = f"result{seen}-{batch_num}-({result[0]}).txt"
                    file_path = dir / result_file_name
                    write_result_to(result, file_path)
                    print("        writing file: ", file_path)

                #dir.mkdir(parents=True, exist_ok=True)
                #print(f"            '{sub_subcategory}': {len(items)}  ({sub_subcategory_percentage:.2f}% of '{subcategory}')")


#    distance, question, answer, results, contents, call_chain = r

#physical_methods = extract_methods(extract_class_definition(class_def, "MyClass"))
#res = detailed_analysis(directory, chains, methods, physical_methods)


def analyse_experiment(experiment, model):
    all_results = detail_analysis_dir(experiment, model)
    print(f"total results: {len(all_results)}")
    hier = sort_results(all_results)
    out_dir = "output" #experiment / "output" / model
    write_dir = "exp_out"
    write_files(experiment, out_dir, write_dir, model, hier)

#models = ["coder-14b", "coder-32b", "Coder-7B", "Coder-1.5B", "Llama-3.1"]
models = ["Mistral-Small"]

def analyze_experiments(xps):
    for m in models:
        for xp in xps:
            xp_path = Path(xp)
            analyse_experiment(xp_path, m)

experiments = sys.argv[1:]
#model = sys.argv[-1]
print(experiments)
analyze_experiments(experiments)
#print(model)
#right_files, back_files, wrong_files, hallucinated_files, interaction_files = count_files_with_categories(res)
#print(f"Files with only correct words: {right_files}")
#print(f"Files with words in 'back': {back_files}")
#print(f"Files with words in 'wrong': {wrong_files}")
#print(f"Files with words hallucinated: {hallucinated_files}")
#print(f"Files with subwords intersection: {interaction_files}")

# next steps: 
# remove the "bad calls" at the beginning, when the model is restating the instruction
# also maybe some at the end
# especially from starting point to end

# look at all the backwards call, not only the chain? => done
# special class, not bad? => done 
# look for similarity with backwards too
# look for cases where it overlooks empty methods
# 173 -3 
# 239 -5 : check legitimate backwards call, that's not it .... more like a backtrack??


# if backward, check if it is the first occurence of the method or not!!!
# if we start with the call, and we have the definition later, is it more likely to mistake one for the other?
# ==> for llama3-8B, this is systematic!!!!

# some of the far miss could be nearby if we look at the method body, or the other end of the chain ...


# we still have the problem of duplicate questions negative that are short ... need something else there


# hypothesis: most of the wrong picks come from similar names, but that come earlier in the file
# this can be when looking for the actual call, but also when looking for another call when one is already in the wrong track
# so we need to check several items
# the actual call
# the caller/callees of the current method
# the backward ones?

"""
example 87 3

=================
Reasoning:
BAD CALL	SIMILAR: process 1st!			processKeyFromResponse		 -> 		processValueById
BAD CALL	SIMILAR: process 1st!			processValueById		 -> 		processActionIfExists
LEGITIMATE	FAR MISS 1st!			processActionIfExists		 -> 		enableInfoForRole
LEGITIMATE	FAR MISS 1st!			enableInfoForRole		 -> 		fetchJobAtPosition
LEGITIMATE	FAR MISS 1st!			fetchJobAtPosition		 -> 		extractDetailForSession
LEGITIMATE	FAR MISS 1st!			extractDetailForSession		 -> 		generateComponentWithLimit
LEGITIMATE	FAR MISS 1st!			generateComponentWithLimit		 -> 		buildStatusAsString
LEGITIMATE	FAR MISS 1st!			buildStatusAsString		 -> 		disableStatusWithLimit
LEGITIMATE	NEAR MISS			disableStatusWithLimit		 -> 		loadTransactionOnClick
BAD CALL	FAR MISS 1st!			loadTransactionOnClick		 -> 		filterMetricInParallel
LEGITIMATE	FAR MISS 1st!			filterMetricInParallel		 -> 		generateElementWithIndex
LEGITIMATE	FAR MISS 1st!			generateElementWithIndex		 -> 		enableNotificationForTransaction
BAD CALL	SIMILAR: process 1st!			enableNotificationForTransaction		 -> 		processKeyFromResponse
=================
A:

FAR MISS is not a good diagnostic here?
the model is "just following" a reasoning chain that makes sense
the FAR MISS, etc, is just for when we have made a bad call
the second bad jump is also because it's a similar name that shows up earlier

LEGITIMATE should be split into
"ON_TRACK"
and "OFF TRACK"?
"""

# still need to solve the issue of chain generation
# for small negative questions we have a lot of repeat otherwise
# this is related to the amount of possible questions to ask
# especially at small context sizes

# experiences
# fix question length, vary context size
# length 5, context: 10, 20, 40, 50, ...
# try higher context size, but ordered, not shuffled
# try higher context with more dissimilar names
# try smaller context with "similar" words
# before and after the words in questions

# questions: if a generation is cut off, how likely is it that it was on the right track