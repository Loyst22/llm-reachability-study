import json
import os
import sys
import re
from pathlib import Path
from collections import defaultdict
import shutil


""" Utils functions """

def element_distance(element_list: list, element1: str, element2: str) -> int:
    """Calculates the distance between two elements in a list.
    
    Args:
        element_list (list of str): The list of elements.
        element1 (str): The first element.
        element2 (str): The second element.
    
    Returns:
        int: The distance between the first occurrences of element1 and element2.
             Returns -1 if either element is not found in the list.
    """    
    try:
        index1 = element_list.index(element1)
        index2 = element_list.index(element2)
        return abs(index1 - index2)
    except ValueError:
        # Return -1 if either element is not in the list
        return -1
    

def first_in_list(lst: list, item1, item2):
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

    
def min_element_distance(element_list: list, first_element, second_element_list: list) -> int:
    """
    Calculates the minimum distance between a target element and any element from a list of other elements.

    Args:
        element_list (list): The list in which to search for elements.
        first_element: The reference element to measure distances from.
        second_element_list (list): A list of elements to compare distances to.

    Returns:
        int: The smallest distance (in list positions) between the first occurrence of first_element 
             and the first occurrences of any elements in second_element_list.
             Returns -1 if any of the elements are not found in the list.
    """
    res = []
    for w in second_element_list:
        res.append(element_distance(element_list, first_element, w))
    return min(res)

def filtered_fraction(list_a, num, predicate):
    """
    Calculates the fraction of elements in `list_a` that satisfy a given predicate,
    relative to a provided denominator `num`.

    Args:
        list_a (list): The list of elements to filter.
        num (int or float): The denominator used to compute the fraction.
        predicate (callable): A function that takes an element of `list_a` and returns True or False.

    Returns:
        float: The fraction of elements in `list_a` that satisfy the predicate divided by `num`.
               Returns 0 if `num` is zero to avoid division by zero.
    """
    filtered_a = [element for element in list_a if predicate(element)]
    if num == 0:
        return 0
    fraction = len(filtered_a) / num
    return fraction

def remove_consecutive_duplicates(lst: list) -> list:
    """
    Removes consecutive duplicate elements from a list.

    Args:
        lst (list): The input list.

    Returns:
        list: A new list with consecutive duplicates removed, while preserving the original order of elements.
    """
    # Create an empty list to store the result
    result = []
    
    # Iterate over the list and add the first element to the result
    for i in range(len(lst)):
        if i == 0 or lst[i] != lst[i-1]:  # Check if the current element is different from the previous
            result.append(lst[i])
    
    return result

def convert_to_consecutive_pairs(lst: list) -> list:
    """
    Converts a list into a list of consecutive element pairs.

    Args:
        lst (list): The input list.

    Returns:
        list: A list of tuples, where each tuple contains two consecutive elements from the input list.

    Example:
        convert_to_consecutive_pairs(['a', 'b', 'c', 'd']) 
        # Returns: [('a', 'b'), ('b', 'c'), ('c', 'd')]
    """
    return list(zip(lst, lst[1:]))

def find_first_match(lst: list, target: str):
    """
    Finds the first tuple in a list where the first element matches the target.
    
    Args:
        calls (list of tuple): The list of tuples to search.
        target (str): The value to match with the first element of each tuple.
    
    Returns:
        element: The second element of the first matching tuple, or None if no match is found.
    """
    # print("find first match ", calls, " target: ", target)
    for tuple in lst:
        if tuple[0] == target:
            # print("find first match returning: ", tuple[1])
            return tuple[1]
    # print("find first match returning NONE")
    return None


""" File extraction function """

def extract_class_definition(java_file: Path, class_name: str) -> str:
    """
    Extracts a Java class definition from a string containing Java code.

    Args:
        java_file (Path): The path to the Java code.
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

def extract_methods(java_code: str) -> list:
    """
    Extracts all the methods defined in a Java class, in order of appearance.
    
    Args:
        java_code (str): The whole Java code as a string.
    
    Returns:
        list: A list of method names, in the order they appear in the code.
    """
    # Regular expression to match method signatures
    # This matches method declarations (including access modifiers and return types)
    method_pattern = re.compile(r'\b\w+\s+\w+\s+(\w+)\s*\(.*\)\s*{')
    
    # Find all method names using the regex
    methods = method_pattern.findall(java_code)
    
    return methods

def extract_answer(content: str) -> str:
    """
    Extracts the answer from the LLM's response

    Args:
        content (str): The content of the LLM's response

    Returns:
        str: YES/NO/NOT FOUND based on the content
    """   
    last_20_chars = content[-20:] if len(content) > 20 else content
    last_20_chars_lower = last_20_chars.lower()
    
    # Case-insensitive search for 'Final Answer: YES' or 'Final Answer: NO'
    yes_match = re.search(r'final answer\s*:\s*yes', content, re.IGNORECASE)
    no_match = re.search(r'final answer\s*:\s*no', content, re.IGNORECASE)

    if yes_match or "yes" in last_20_chars_lower:
        return "YES"
    elif no_match or "no" in last_20_chars_lower:
        return "NO"
    else:
        return "NOT FOUND" 
        
    """ Deprecated: not robuste enough for "Final Answer: YES"...
    last_20_chars = content[-20:] if len(content) > 20 else content

    # Determine the file's status based on the content
    if "FINAL ANSWER: YES" in content or "YES" in last_20_chars:
        return "YES"
    elif "FINAL ANSWER: NO" in content or "NO" in last_20_chars:
        return "NO"
    else:
        return "NOT FOUND"
    """
    
def keep_relevant_output(content: str) -> str:
    """
    Extract the relevant content from a LLM's response

    Args:
        content (str): Content of a LLM's response to a prompt

    Returns:
        str: The relevant output (only the lines containing info about the call chains)
    """    
    # Split into lines
    lines = content.strip().split('\n')
    relevant_lines = []
    
    for line in lines:
        # Keep only lines that start with a number + dot + space (like '1. ')
        if re.match(r'^\s*\d+\.\s', line):
            lower_line = line.lower()
            # ! might be a bad idea but it seems to avoid falsely classifing calls as invalid
            if 'does not call' not in lower_line and 'is not called' not in lower_line and 'is not directly called' not in lower_line:
                relevant_lines.append(line)

    # Join them into a single string separated by newlines
    return '\n'.join(relevant_lines)

def associate_files_with_lines(results_directory: Path, chains_file: Path) -> dict:
    """
    Associate each result file with a line from the other file based on 'x' in the filename.

    Args:
        results_directory (Path): Name of the directory containing the result files
        chains_file (Path): File containing the chains

    Returns:
        dict: Dictionary containing the result_file/chain associations 
    """
        
    # Read all lines from the other file
    with open(chains_file, 'r') as f:
        chains_file_lines = f.readlines()

    # Dictionary to store the associations
    associations = {}

    # print("Results dir: ", results_directory)
    # print("Chains file dir: ", chains_file)
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

""" File writing and printing functions """

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
    with open(filename, "w", encoding="utf-8") as file:
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
        
def write_result_to_json(result, filename, batch_info):
    distance, question, answer, results, contents, call_chain = result
    amount_expected = filtered_fraction(results, abs(distance), lambda r: r[1] == expected)

    # Prepare a dictionary with all the relevant info
    data = {
        "Question": question,
        "Answer": answer,
        "Distance": distance,
        "Context": batch_info["context"],
        "Variables": batch_info["var"],
        "Loops": batch_info["loop"],
        "IfStatements": batch_info["if"],
        "Language": batch_info["language"],
        "CallChain": call_chain,
        "AmountExpected": amount_expected,
        "Results": results,
        "Contents": contents
    }

    # Write dictionary as pretty-formatted JSON
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

def write_files(exp_dir, output_dir, write_dir, batch_infos, model_name, hierarchy):
       # Calculate the total number of objects for percentage calculations
    total_objects = sum(
        len(items) for categories in hierarchy.values()
        for subcategories in categories.values()
        for items in subcategories.values()
    )

    base_dir = exp_dir / output_dir / model_name
    write_base_dirs = []
    write_base_dirs_code = []
    for batch in batch_infos:
        batch_string = f"ctx-{batch['context']}_com-{batch['comments']}_var-{batch['var']}_loop-{batch['loop']}_if-{batch['if']}_{batch['language']}_{batch['structure']}"
        write_base_dirs.append(write_dir / exp_dir / batch_string / model_name)
        write_base_dirs_code.append(write_dir / exp_dir / batch_string)
    
    for category, subcategories in hierarchy.items():
        # print(category)
        # print(subcategories)
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
                seen = 0
                for res in items:
                    seen += 1
                    # Write the text representation to a file
                    # value[0] is the distance
                    *result, batch_num, code_file_path = res
                    dir = write_base_dirs[batch_num-1] / catdir / subcatdir / subsubcatcatdirdir
                    dir.mkdir(parents=True, exist_ok=True)
                    batch_code_file_name = f"TheClass-{batch_num}.java"
                    batch_code_path = write_base_dirs_code[batch_num-1] / batch_code_file_name
                    # print("base code path is: ", batch_code_path)
                    # print("code file path is: ", code_file_path)
                    if not batch_code_path.exists():
                        # copying source file to the path
                        shutil.copy2(code_file_path, batch_code_path)
                    result_file_name = f"result{seen}-{batch_num}-({result[0]}).txt"
                    result_file_name_json = f"result{seen}-{batch_num}-({result[0]}).json"
                    file_path = dir / result_file_name
                    json_path = dir / result_file_name_json
                    write_result_to(result, file_path)
                    write_result_to_json(result, json_path, batch_infos[batch_num-1])
                    print("        writing file: ", file_path)

                #dir.mkdir(parents=True, exist_ok=True)
                #print(f"            '{sub_subcategory}': {len(items)}  ({sub_subcategory_percentage:.2f}% of '{subcategory}')")

""" Camel case checking functions """

def is_camel_case(string: str) -> bool:
    """Matches a string that starts with a lowercase letter and contains at least one uppercase letter"""
    return bool(re.match(r'^[a-z]+[A-Za-z]*[A-Z]+[A-Za-z]*$', string))

def split_camel_case(string: str):
    """Split camel-case words into lowercase subwords, filtering out common prepositions."""
    common_prepositions = {"in", "on", "at", "to", "for", "by", "with", "about", "against", 
                           "between", "into", "through", "during", "before", "after", "above", 
                           "below", "from", "up", "down", "over", "under", "again", "further", 
                           "then", "once", "of", "off"}
    words = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z])|[A-Z]+$', string)
    filtered_words = {word.lower() for word in words if word.lower() not in common_prepositions}
    
    return filtered_words

def camel_case_intersection(single_word: str, word_list: list):
    """
    Finds the common subwords between a single camel case word and a list of camel case words.

    Args:
        single_word (str): A camel case word to compare.
        word_list (list of str): A list of camel case words.

    Returns:
        list: A list of subwords that are present both in the single_word and in any of the words in word_list.
    """
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

""" Call analysis functions """

""" Aliases to classify/qualify method calls """

legit = "GOOD"
expected = "EXPECTED" 
repeat = "REPEATS OLD CALL"  
backtrack_first = "BACKTRACK FIRST CALL"
backtrack_other = "BACKTRACK OTHER"

bad = "BAD"
badback = "BACK SIMILAR:"
hallucinated = "HALLUCINATED"

off1 = "OFF BY ONE"
near = "NEAR MISS"
similar = "~= "
far = "FAR MISS"
extra = "EXTRA CALL"
first = " 1st!"

backwards = "BACKWARDS CALL" # unused
backwards_first = "BACKWARDS (FIRST)"
backwards_last = "BACKWARDS (LAST)"
close_back = "CLOSE_BACK"
far_back = "FAR_BACK"

def find_legit_call(calls: list, target: str): return find_first_match(calls, target)

def detailed_analysis(directory: Path, chains: Path, methods: Path, physical_methods: list) -> list:
    """
    Performs a detailed step-by-step analysis of method call chains extracted from files 
    and compares them to the expected method call chains.

    For each file in the provided directory that matches the chains, the function:
    - Reads the associated question and LLM-generated answer.
    - Extracts method names and call chains.
    - Analyzes whether the reasoning in the LLM output correctly follows the expected method chain.
    - Produces a detailed result including reasoning steps and verification status.

    Notes:
    - The function uses `right_chain` from `analyze_reasoning` in the results.
    - The original `chain_methods` and `back_chain_methods` passed to `analyze_reasoning` are not modified. 
      Only internal copies are manipulated within `analyze_reasoning`.

    Args:
        directory (Path): Path to the directory containing the result files.
        chains (Path): Path to the file associating each result file with its corresponding method chains.
        methods (Path): Path to the file containing the list of all known method names.
        physical_methods (list of str): The ordered list of physical method names used to compute distances.

    Returns:
        list: A list of detailed analysis results, where each entry is a tuple containing:
            - The distance between the expected and generated answers.
            - The original question.
            - The LLM-generated answer.
            - A detailed step-by-step reasoning trace (list of steps with labels).
            - The raw content of the LLM output.
            - The expected method chain (right_chain), which remains unmodified from its original input.
    """
    with open(methods, 'r') as f:
        all_methods = f.read().split()
        associations = associate_files_with_lines(directory, chains)
        all_results = []
        for filename, associated_line in list(associations.items())[:]:
            chains = associated_line.split('\t')
            # weird ... should not happen but oh well
            # update: should happen for tree chains as we didn't define any back chain
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

def analyze_reasoning(file_path: str, right_chain: list, back_chain: list, all_methods: list, physical_methods: list) -> tuple:
    """
    Analyzes the reasoning steps of an LLM-generated file to verify the accuracy of method call chains.

    This function:
    - Reads the question and the generated answer from a result file.
    - Extracts camelCase method names from the question and the LLM's output.
    - Identifies the method call sequences claimed by the LLM.
    - Compares these claimed sequences against the expected method call chain (`right_chain`), backward chains, and all known methods.
    - Labels each identified method call based on its correctness, position in the sequence, proximity to expected methods, or if it is hallucinated.

    Args:
        file_path (str): Path to the file containing the LLM's output and associated question.
        right_chain (list of str): The expected sequence of method calls.
        back_chain (list of str): The backward method call sequence.
        all_methods (list of str): A list of all known methods in the system.
        physical_methods (list of str): A list of method names that are really in the java file.

    Returns:
        tuple:
            question (str): The question asked to the LLM.
            answer (str): The LLM's extracted final answer.
            results (list): A list of lists, each describing a method call and its associated labels (e.g., correct, wrong, backtrack, hallucinated).
            content (str): The full text content of the LLM's response.
            right_chain (list of str): The actual chain (should remain the same).

    Notes:
        - The function heavily relies on string processing and camelCase extraction to follow the flow of reasoning.
        - It handles edge cases like extra calls, backtracking calls, hallucinated calls, and near-miss calls.
        - The function assumes a specific file structure: the first line is the question, the remaining lines are the LLM's output.
    """
    print("Analysing file:", file_path)
    
    # Compile a regular expression pattern to remove unwanted punctuation
    # This pattern removes any non-alphanumeric character (e.g., quotes, backquotes, parentheses, etc.)
    pattern = re.compile(r'[^\w\s]')

    # Step 2: Read the content of the file
    try:
        with open(file_path, 'r',  encoding='latin-1') as file:
            # TODO : will depend on the way we made the result files
            content = file.readlines() 
            question = content[0] # The first line is supposed to be the question
            content = " ".join(content[1:]) # The rest is the actual LLM content
            answer = extract_answer(content)

        # We extract the method names from the question
        camel_words = []
        # Step 4: Iterate through each word in the content
        for word in question.split():
            # Clean up the word by removing unwanted punctuation
            cleaned = pattern.sub('', word) # Remove punctuation
            if is_camel_case(cleaned):
                camel_words.append(cleaned)
        start = camel_words[0]
        end = camel_words[1]
        
        # We extract the method names from the LLM's output
        camels_llm = []
        # Extract only the relevant lines from the LLM's output
        # Otherwise we end up extracting calls that were not mentioned by the LLM
        # This is what was already kind of taken into account with final_calls, perhaps not perfectly
        # There was some issue because the last method name from the relevant content and the first
        # one from the conclusion was then considered a method call
        # Similar issue for chains of distance 1 (only 2 methods)
        relevant_content = keep_relevant_output(content)
        # Step 4: Iterate through each word in the content
        for word in relevant_content.split(): # Split the AI's output content into words
            # Clean up the word by removing unwanted punctuation
            cleaned_word = pattern.sub('', word) # Remove punctuation
            if is_camel_case(cleaned_word):
                camels_llm.append(cleaned_word)
        camels_llm = remove_consecutive_duplicates(camels_llm)
        calls_llm = convert_to_consecutive_pairs(camels_llm)
        
        all_calls = convert_to_consecutive_pairs(all_methods)
        call_chain = convert_to_consecutive_pairs(right_chain)
        if str(file_path).__contains__("result3_3"):
            print("Actual call chain:", call_chain)
            print("Call chain from LLM:", calls_llm)
        past_calls = []
        results = []
        # we remove the target call from the back chain
        back_calls = convert_to_consecutive_pairs(all_methods[::-1])
        # add the start method as starting point, then reverse
        close_back_calls = convert_to_consecutive_pairs((back_chain + [right_chain[0]])[::-1])
        final_calls = [(start, end),(end, start)] if len(right_chain) > 2 else []
        for call in calls_llm:
            if str(file_path).__contains__("result11_-6"):
                print("AI call:", call)
            # Disregard calls that are just final calls
            if call in final_calls:
                continue
            res = []
            if call in all_calls:
                # this call exists
                res.append(legit)
                if str(file_path).__contains__("result3_3"):
                    print("AI call:", call)
                if call_chain and call == call_chain[0]:
                    # this is the call we expect
                    res.append(expected)
                    # we expect the next one in the future
                    past_calls.append(call_chain.pop(0))
                elif call in past_calls:
                    # this is a call in a past call chain, but we have already encountered it
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
                        distance = element_distance(physical_methods, target, actual)
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
                    distance = element_distance(physical_methods, target, actual)
                    distanceback = element_distance(physical_methods, target, actual)
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


def analysis(directory: str, chains: str, methods: str, physical_methods: list):
    """
    Analyzes the association between method call chains and AI-generated output files.

    This function:
    - Associates result files with specific method chains.
    - Parses these chains into forward and backward method call sequences.
    - Analyzes each result file to check if the methods mentioned align with the expected chains.
    - Collects statistics such as correctly identified methods, incorrect methods, back calls, hallucinated methods, and method intersections.

    Args:
        directory (str): Path to the directory containing the LLM result files.
        chains (str): Path to the file that contains method chains.
        methods (str): Path to the file containing the list of all known methods.
        physical_methods (list): A list of method names that are really in the java file.

    Returns:
        list: A list of tuples, each containing:
            - Number of correctly identified methods (int)
            - Number of incorrectly identified methods (int)
            - Number of back call methods found (int)
            - Number of hallucinated methods (int)
            - List of intersecting methods (list)
    """
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
                    print("DISTANCE: ", min_element_distance(physical_methods, cleaned_word, right_words))
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
    
   
def is_right_answer_for_dist(answer: str, dist: int) -> bool: 
    """ Checks the validity of the answer given by the LLM wrt the distance of the questions """
    return (dist > 0 and answer == "YES") or (dist < 0 and answer == "NO")

    
def answer_distance(filename: str) -> int:
    """ Extract the 'y' value from the filename (the second number in 'resultx_y.txt') """
    y_value = int(filename.split('_')[1].split('.')[0]) # Extract the second number from filename
    return y_value

def check_answer(filename: str, answer: str) -> bool:
    return is_right_answer_for_dist(answer, answer_distance(filename))
 

def count_files_with_categories(results: list) -> tuple:
    """
    Counts files based on the presence of words in specific categories within each result.

    Args:
        results (list): A list of tuples, each containing counts of different categories
                        in the format (_, wrong, back, hallucinated, intersection).

    Returns:
        tuple: A 5-element tuple with counts of files categorized as:
            - all_right: Files with no 'back', 'wrong', or 'hallucinated' words.
            - back_count: Files containing one or more 'back' words.
            - wrong_count: Files containing one or more 'wrong' words (and no 'back' words).
            - hallucinated_count: Files containing one or more 'hallucinated' words (and no 'back' words).
            - intersection_count: Files with an intersection count greater than zero and either 'wrong' or 'hallucinated'.
    """
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


# analyses the output of 1 model
def detail_analysis_dir(experiment_dir: Path, model_name: str) -> list:
    """
    Runs detailed analysis across multiple batches and subdirectories within an experiment directory 
    for a specific model.

    For each batch directory inside `experiment_dir`:
    - Reads the associated chains, methods, and system definition files.
    - Extracts the physical methods from the class definition.
    - Iterates over subdirectories containing the specified `model_name`.
    - Runs `detailed_analysis` on each relevant subdirectory.
    - Appends batch metadata (batch number and class file path) to each analysis result.
    - Collects and returns all results across batches.

    Args:
        experiment_dir (Path): Path object pointing to the root directory containing batch subdirectories.
        model_name (str): Substring to identify relevant model-specific subdirectories within each batch.

    Returns:
        list of tuples: A list where each element corresponds to an analysis result from `detailed_analysis`, 
        extended with two additional elements:
            - batch_num (int): The index of the batch directory (starting at 1).
            - class_file (Path): Path object pointing to the class definition file used in the batch.

    Notes:
        - Assumes each batch directory contains `chains.txt`, `methods.txt`, and `system.txt` files.
        - Extracts physical methods from the Java class definition named "TheClass".
        - Only subdirectories whose names include `model_name` are analyzed.
    """
    pattern = (
    r'ctx[-_](?P<context>\d+)'  # ctx_30 or ctx-30
    r'_depths[-_](?P<depths>(?:\d+[-_]?)+)'  # supports depths like 8--12 or 8_9_10
    r'(?:_com[-_](?P<comments>\d+))?'        # optional _com
    r'(?:_var[-_](?P<var>\d+))?'             # optional _var
    r'(?:_loop[-_](?P<loop>\d+))?'           # optional _loop
    r'(?:_if[-_](?P<if>\d+))?'               # optional _if
    r'_qs[-_](?P<qs1>\d+)(?:--(?P<qs2>\d+))?'  # required _qs start, optional end
    r'(?:_(?P<language>[a-zA-Z0-9]+))?'      # optional language
    r'(?:_(?P<structure>linear|tree))?'      # optional structure
    )


    all_results = []
    batch_infos = []
    batch_num = 0
    visited = set()

    for root, dirs, _ in os.walk(experiment_dir):
        root_path = Path(root)

        if root_path in visited:
            continue

        match = re.match(pattern, root_path.name)
        rel_path = os.path.relpath(root_path, experiment_dir)
        print(rel_path)

        if match is None:
            print(f"No match for directory: {rel_path}")
            continue

        # Prevent walking deeper into this matched batch directory
        dirs.clear()
        visited.add(root_path)

        batch_infos.append(match.groupdict())
        batch_num += 1

        chains = root_path / "chains.txt"
        methods = root_path / "methods.txt"
        class_def = root_path / "system.txt"
        class_file = root_path / "TheClass.java"

        physical_methods = extract_methods(extract_class_definition(class_def, "TheClass"))

        for subdir in root_path.iterdir():
            if subdir.is_dir() and model_name in str(subdir):
                results = detailed_analysis(subdir, chains, methods, physical_methods)
                results_with_batch = [r + (batch_num, class_file) for r in results]
                all_results.extend(results_with_batch)

    return all_results, batch_infos
    

def detail_analysis_dir_v2(experiment_dir: Path, model_name: str) -> list:
    """
    Traverses experiment_dir to find context directories, then analyses batch directories inside them.
    """

    # Pattern for outer "context" dirs
    context_pattern = re.compile(
        r'context[-_](?P<context>\d+)' +
        r'_comment[-_](?P<comments>\d+)' +
        r'_var[-_](?P<var>\d+)' +
        r'_loop[-_](?P<loop>\d+)' +
        r'_if[-_](?P<if>\d+)' +
        r'(?:_(?P<language>[a-zA-Z0-9]+))?' +
        r'(?:_(?P<structure>linear|tree))?'
    )

    # Pattern for inner batch dirs
    batch_pattern = re.compile(
        r'ctx[-_](?P<context>\d+)' +
        r'_depths[-_](?P<depths>(?:\d+[-_]?)+)' +
        r'_qs[-_](?P<qs1>\d+)(?:--(?P<qs2>\d+))?'
    )

    all_results = []
    batch_infos = []
    batch_num = 0

    for context_dir in experiment_dir.iterdir():
        if not context_dir.is_dir():
            continue

        match = context_pattern.match(context_dir.name)
        if not match:
            continue

        batch_info = match.groupdict()

        for batch_dir in context_dir.iterdir():
            if not batch_dir.is_dir():
                continue
            if not batch_pattern.match(batch_dir.name):
                continue

            chains = batch_dir / "chains.txt"
            methods = batch_dir / "methods.txt"
            class_def = batch_dir / "system.txt"
            class_file = batch_dir / "TheClass.java"

            if not (chains.exists() and methods.exists() and class_def.exists()):
                print(f"Missing files in {batch_dir}, skipping.")
                continue

            try:
                physical_methods = extract_methods(extract_class_definition(class_def, "TheClass"))
            except Exception as e:
                print(f"Failed to extract methods from {class_def}: {e}")
                continue

            for subdir in batch_dir.iterdir():
                if subdir.is_dir() and model_name in str(subdir):
                    results = detailed_analysis(subdir, chains, methods, physical_methods)
                    results_with_batch = [r + (batch_num + 1, class_file) for r in results]
                    all_results.extend(results_with_batch)

            batch_infos.append(batch_info)
            batch_num += 1

    return all_results, batch_infos

# if len(sys.argv) > 1:
#     directory = sys.argv[1]
# else:
#     print("default dir")


# # make the files dependent on the dir 
# directory = "../llama.cpp-master/output/reachability_questions.txt_2024-11-08_14-51-13"
# chains = "../llama.cpp-master/chains.txt"
# methods = "../llama.cpp-master/methods.txt"
# class_def = "../llama.cpp-master/system.txt"

# directory = "/Users/rrobbes/Projects/reachability/llama.cpp-master/output/reachability_questions.txt_2024-11-07_14-59-24"


# then do something that generates outputs and aggregate them from subdirs to main dirs?


""" Result classification functions """

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


defright = "DEFINITELY_RIGHT"
probright = "PROBABLY_RIGHT"
border = "BORDERLINE"
probwrong = "PROBABLY_WRONG"
rightwrong = "RIGHT_FOR_WRONG_REASONS"
lost_track = "LOST_TRACK"


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

def classify_right_for_wrong_reasons(r):
    return "RIGHT FOR WRONG REASON"

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
        

# def classify_wrong(r):


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


def display_list_lengths(dict_of_lists: dict) -> None:
    for key, value in dict_of_lists.items():
        print(f"{key}: {len(value)}")


def sort_results(results):
    hierarchy = categorize_objects(results)
    print_statistics(hierarchy)
    return hierarchy


#    distance, question, answer, results, contents, call_chain = r

#physical_methods = extract_methods(extract_class_definition(class_def, "TheClass"))
#res = detailed_analysis(directory, chains, methods, physical_methods)


""" Start analysis functions """

def analyse_experiment(experiment, model):
    print(f"\nAnalysis of dir {experiment} for model {model}")
    all_results, batch_infos = detail_analysis_dir_v2(experiment, model)
    print(batch_infos)
    if len(all_results) == 0:
        print(f"No results for model {model}. Exiting...")
        return
    hier = sort_results(all_results)
    print()
    out_dir = "output" #experiment / "output" / model
    write_dir = "exp_out"
    # experiment_infos = {
    #     "model": model,
    #     "context_size": ctx_size
    # }
    write_files(experiment, out_dir, write_dir, batch_infos, model, hier)


def analyze_experiments(xps: list, models: list):
    for model in models:
        for xp in xps:
            xp_path = Path(xp)
            analyse_experiment(xp_path, model)

if __name__ == "__main__":
    experiments = sys.argv[1:]
    # model = sys.argv[-1]
    
    # models = [model]
    # models = ["coder-14b", "coder-32b", "Coder-7B", "Coder-1.5B", "Llama-3.1"]
    # models = ["Mistral-Small"]
    # models = ["Coder-7B", "Coder-3B"]
    models = ["output-Mistral-Small-3.1-24B-Instruct-2503-Q6_K"]
    models = ["Meta-Llama-3.1-8B.Q8_0",
              "output-Qwen2.5-Coder-1.5B.Q8_0",
              "output-Qwen2.5-Coder-7B-Instruct.Q8_0",
              "output-Qwen2.5-Coder-7B-Instruct.Q8_0",
              "output-qwen2.5-coder-32b-instruct-q8_0"]
        
    print("Analysing experiments for following directories:", experiments)
    print("And models:", models)
    analyze_experiments(experiments, models)
    

#right_files, back_files, wrong_files, hallucinated_files, interaction_files = count_files_with_categories(res)
#print(f"Files with only correct words: {right_files}")
#print(f"Files with words in 'back': {back_files}")
#print(f"Files with words in 'wrong': {wrong_files}")
#print(f"Files with words hallucinated: {hallucinated_files}")
#print(f"Files with subwords intersection: {interaction_files}")


""" More explanations and examples """


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