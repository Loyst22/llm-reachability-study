import os
import pandas as pd
import sys
import shutil
from pathlib import Path

def check_answer(filename, answer):
    # Extract the 'y' value from the filename (the second number in 'resultx_y.txt')
    y_value = int(filename.split('_')[1].split('.')[0])  # Extract the second number from filename
    
    # Return True or False based on the conditions
    if (y_value > 0 and answer == "FINAL ANSWER: YES") or \
       (y_value < 0 and answer == "FINAL ANSWER: NO"):
        return True
    else:
        return False
    
def copy_files_to_category_directories(directory, files_and_answers):
    """
    Copies files into subdirectories according to their categories.
    Skips copying if the target directory already exists.
    
    Args:
    - directory (str): The main directory to create subdirectories in.
    - files_by_category (dict): A dictionary with categories as keys and a list of filenames as values.
    """
    dir = Path(directory)
   
    for filename, answer in files_and_answers.items():
        # Define the path for the category directory
        if answer == "NOT FOUND":
                category_dir = "NOT_FOUND"
        elif check_answer(filename, answer):
                #answer was correct
            if answer == "FINAL ANSWER: YES":
              category_dir = "GUESSED YES AND RIGHT"
            else:
                category_dir = "GUESSED NO AND RIGHT"
        else:
            if answer == "FINAL ANSWER: YES":
                category_dir = "GUESSED YES AND WRONG"
            else:
                category_dir = "GUESSED NO AND WRONG"

        category_dir =  dir / category_dir.replace(" ", "_")


        # Check if the directory exists, and if it doesn't, create it
        if not category_dir.exists():
            category_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

        # copy file to the category directory
        origin = dir / filename
        target_file_path = category_dir / filename  # Set the target path for the copied file
            
        # Copy the file only if it doesn't already exist in the target location
        if not target_file_path.exists():
                shutil.copy(origin, target_file_path)
                print(f"Copied {filename} to {category_dir}")
        else:
                print(f"File {filename} already exists in {category_dir}, skipping copy.")

# Example usage:
# Assuming files_by_category is a dictionary like {"category1": ["file1.txt", "file2.txt"], ...}
# and `directory` is the main directory where category subdirectories should be created.


def search_answer_in_files(directory):
    # Initialize a dictionary to store results for each file
    results = {}
    
    # Define the categories (subdirectories)

    
    # Loop through each file in the given directory
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):  # Only consider .txt files
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, 'r') as file:
                    content = file.read()

                    last_20_chars = content[-20:] if len(content) > 20 else content


                    # Determine the file's status based on the content
                    if "FINAL ANSWER: YES" in content or "YES" in last_20_chars:
                        answer = "FINAL ANSWER: YES"
                    elif "FINAL ANSWER: NO" in content or "NO" in last_20_chars:
                        answer = "FINAL ANSWER: NO"
                    else:
                        answer = "NOT FOUND"
                    
                    # Store the results for each file
                    results[filename] = answer
            except Exception as e:
                results[filename] = f"Error reading file: {e}"
    
    copy_files_to_category_directories(directory, results)
    return results

""" commented out
    # Create subdirectories for each category and move files accordingly
    for category in categories:
        # Replace colons with underscores for compatibility and create subdirectory
        category_path = os.path.join(directory, category.replace(":", "_").replace(" ", "_"))
        os.makedirs(category_path, exist_ok=True)


    # Now categorize and move files to the corresponding subdirectories
    for filename, answer in results.items():
        if answer != "Error reading file:":
            if answer == "NOT FOUND":
                category_dir = os.path.join(directory, "NOT_FOUND".replace(" ", "_"))
            else:
                # Use the check_answer function to determine if the guess was right or wrong
                if check_answer(filename, answer):
                    if answer == "FINAL ANSWER: YES":
                        category_dir = os.path.join(directory, "GUESSED YES AND RIGHT".replace(" ", "_"))
                    else:
                        category_dir = os.path.join(directory, "GUESSED NO AND RIGHT".replace(" ", "_"))
                else:
                    if answer == "FINAL ANSWER: YES":
                        category_dir = os.path.join(directory, "GUESSED YES AND WRONG".replace(" ", "_"))
                    else:
                        category_dir = os.path.join(directory, "GUESSED NO AND WRONG".replace(" ", "_"))

            # Move the file to the corresponding subdirectory
            file_path = os.path.join(directory, filename)
            try:
                shutil.copy(file_path, category_dir)
                print(f"Copied {filename} to {category_dir}")
            except Exception as e:
                print(f"Error copying {filename}: {e}")

    return results
"""

def count_and_remove_not_found(results):
    # Count how many entries are "NOT FOUND"
    not_found_count = sum(1 for result in results.values() if result == "NOT FOUND")
    
    # Remove files with "NOT FOUND" from the results
    results_cleaned = {filename: result for filename, result in results.items() if result != "NOT FOUND"}

    return results_cleaned, not_found_count

def check_answer(filename, answer):
    # Extract the 'y' value from the filename (the second number in 'resultx_y.txt')
    y_value = int(filename.split('_')[1].split('.')[0])  # Extract the second number from filename
    
    # Return True or False based on the conditions
    if (y_value > 0 and answer == "FINAL ANSWER: YES") or \
       (y_value < 0 and answer == "FINAL ANSWER: NO"):
        return True
    else:
        return False

def count_trues_and_falses(filenames_and_answers):
    # Apply check_answer to each (filename, answer) pair using map, then sum the trues
    true_count = sum(map(lambda x: check_answer(x[0], x[1]), filenames_and_answers))
    false_count = len(filenames_and_answers) - true_count  # False count is the rest
    return true_count, false_count

def clean_results(results):
    """Clean the results by filtering out "NOT FOUND" entries."""
    return {filename: answer for filename, answer in results.items() if answer != "NOT FOUND"}

def count_right_and_wrong_answers(filenames_and_answers):
    """Count the number of right and wrong answers, and return the counts and percentages."""
    true_count, false_count = count_trues_and_falses(filenames_and_answers)  # Using map-based counting function
    return true_count, false_count

def compute_percentages(total_count, true_count, false_count, not_found_count):
    """Compute the percentages of right, wrong, and not found."""
    if total_count == 0:
        return 0.0, 0.0, 0.0  # Avoid division by zero if no files
    
    true_percentage = (true_count / total_count) * 100
    false_percentage = (false_count / total_count) * 100
    not_found_percentage = (not_found_count / total_count) * 100
    
    return true_percentage, false_percentage, not_found_percentage

def analyze_results(results):
    """Main function to clean results, count answers, and compute percentages."""
    # Clean results by removing "NOT FOUND" entries
    cleaned_results = clean_results(results)
    
    # Count the right and wrong answers
    true_count, false_count = count_right_and_wrong_answers(list(cleaned_results.items()))
    
    # Count the "NOT FOUND" entries
    not_found_count = len(results) - len(cleaned_results)
    
    # Calculate the total number of files
    total_count = len(results)
    
    # Compute percentages for each category
    true_percentage, false_percentage, not_found_percentage = compute_percentages(
        total_count, true_count, false_count, not_found_count
    )
    
    # Get lists of filenames for each category
    true_files = [filename for filename, answer in cleaned_results.items() if check_answer(filename, answer)]
    false_files = [filename for filename, answer in cleaned_results.items() if not check_answer(filename, answer)]
    not_found_files = [filename for filename, answer in results.items() if answer == "NOT FOUND"]
    
    # Print the filenames in each category
    print("Files with correct answers (True):", true_files)
    print("Files with incorrect answers (False):", false_files)
    print("Files with NOT FOUND:", not_found_files)
    
    # Return the counts and percentages along with the filenames
    return {
        "not_found_count": not_found_count,
        "true_count": true_count,
        "false_count": false_count,
        "true_percentage": true_percentage,
        "false_percentage": false_percentage,
        "not_found_percentage": not_found_percentage
    }

from collections import defaultdict


def print_last_100_chars_directory(dir, sub):
    """
    Checks if the subdirectory exists and prints the last 100 characters of each file in that directory.
    """
    directory = Path(dir) / sub
    
    # Ensure that the "not found" directory exists by calling the copy function with an empty dictionary
    if not directory.exists():
        print(f"No 'not found' directory exists in '{directory}'. Please run the file copy process first.")
        return

    # Loop through each file in the "not found" directory and print last 100 characters
    for file_path in directory.glob("*.txt"):
        try:
            with file_path.open('r') as file:
                content = file.read()
                last_100_chars = content[-100:]
                print(f"{file_path.name}: {last_100_chars}")
        except Exception as e:
            print(f"Error reading file {file_path.name}: {e}")

def analyze_results_by_distance(results):
    """Analyze results grouped by distance, calculating counts and percentages for each distance."""
    # Group results by distance
    grouped_results = defaultdict(dict)
    for filename, answer in results.items():
        try:
            # Extract 'y' value as the distance from the filename format 'resultx_y.txt'
            distance = int(filename.split('_')[1].split('.')[0])
            grouped_results[distance][filename] = answer
        except (IndexError, ValueError):
            print(f"Skipping invalid filename format: {filename}")
            continue

    # Analyze each group by distance
    analysis_by_distance = []
    for distance, distance_results in sorted(grouped_results.items()):
        # Clean the results for this distance
        cleaned_results = clean_results(distance_results)
        
        # Count true and false answers for this distance
        true_count, false_count = count_right_and_wrong_answers(list(cleaned_results.items()))
        
        # Count NOT FOUND entries for this distance
        not_found_count = len(distance_results) - len(cleaned_results)
        
        # Total count for this distance
        total_count = len(distance_results)
        
        # Calculate percentages
        true_percentage, false_percentage, not_found_percentage = compute_percentages(
            total_count, true_count, false_count, not_found_count
        )
        
        # Get lists of filenames for each category in this distance group
        true_files = [filename for filename, answer in cleaned_results.items() if check_answer(filename, answer)]
        false_files = [filename for filename, answer in cleaned_results.items() if not check_answer(filename, answer)]
        not_found_files = [filename for filename, answer in distance_results.items() if answer == "NOT FOUND"]
        print("!!!!!!!!!!!! not found files")
        print(not_found_files)
        
        # Append the analysis for this distance
        analysis_by_distance.append({
            "distance": distance,
            "not_found_count": not_found_count,
            "true_count": true_count,
            "false_count": false_count,
            "true_percentage": true_percentage,
            "false_percentage": false_percentage,
            "not_found_percentage": not_found_percentage,
            "true_files": true_files,
            "false_files": false_files,
            "not_found_files": not_found_files
        })

    return analysis_by_distance



# print(analyze_results_by_distance(search_answer_in_files(directory)))


import matplotlib.pyplot as plt

def plot_analysis_by_distance(analysis_by_distance, window_size = 3):
    """Plots the True, False, and NOT FOUND counts across different distances."""
    # Extract distances and counts for each category
    distances = [entry['distance'] for entry in analysis_by_distance]
    true_counts = [entry['true_count'] for entry in analysis_by_distance]
    false_counts = [entry['false_count'] for entry in analysis_by_distance]
    not_found_counts = [entry['not_found_count'] for entry in analysis_by_distance]


    # Calculate rolling means using pandas
    true_counts_smoothed = pd.Series(true_counts).rolling(window=window_size, min_periods=1).mean()
    false_counts_smoothed = pd.Series(false_counts).rolling(window=window_size, min_periods=1).mean()
    not_found_counts_smoothed = pd.Series(not_found_counts).rolling(window=window_size, min_periods=1).mean()
    
    
    # Set up the plot
    plt.figure(figsize=(10, 6))

    # Plot each line with a different color and label (original counts)
    #plt.plot(distances, true_counts, label="True", color="green", marker='o', linestyle="--")
    #plt.plot(distances, false_counts, label="False", color="red", marker='o', linestyle="--")
    #plt.plot(distances, not_found_counts, label="NOT FOUND", color="blue", marker='o', linestyle="--")
    
    # Plot the rolling mean with a solid line for each category
    plt.plot(distances, true_counts_smoothed, label=f"True (Rolling Mean)", color="darkgreen", linestyle="-")
    plt.plot(distances, false_counts_smoothed, label=f"False (Rolling Mean)", color="darkred", linestyle="-")
    plt.plot(distances, not_found_counts_smoothed, label=f"NOT FOUND (Rolling Mean)", color="darkblue", linestyle="-")
   
    
    # Label the axes and title
    plt.xlabel("Distance")
    plt.ylabel("Count")
    plt.title("True, False, and NOT FOUND Counts by Distance")
    
    # Show legend
    plt.legend()
    
    # Show grid for better readability
    plt.grid(True)
    
    # Display the plot
    plt.show()



def plot_analysis_by_distance_with_signs(analysis_by_distance, title, window_size=3):
    """Plots True, False, and NOT FOUND counts by positive and negative distances with rolling mean."""
    # Separate entries by positive and negative distances
    positive_entries = [entry for entry in analysis_by_distance if entry['distance'] >= 0]
    negative_entries = [entry for entry in analysis_by_distance if entry['distance'] < 0]
    
    # Extract distances and counts for each group
    pos_distances = [entry['distance'] for entry in positive_entries]
    pos_true_counts = [entry['true_percentage'] for entry in positive_entries]
    pos_false_counts = [entry['false_percentage'] for entry in positive_entries]
    pos_not_found_counts = [entry['not_found_percentage'] for entry in positive_entries]
    
    neg_distances = [entry['distance'] for entry in negative_entries]
    neg_true_counts = [entry['true_percentage'] for entry in negative_entries]
    neg_false_counts = [entry['false_percentage'] for entry in negative_entries]
    neg_not_found_counts = [entry['not_found_percentage'] for entry in negative_entries]
    
    # Calculate rolling means using pandas
    pos_true_counts_smoothed = pd.Series(pos_true_counts).rolling(window=window_size, min_periods=1).mean()
    pos_false_counts_smoothed = pd.Series(pos_false_counts).rolling(window=window_size, min_periods=1).mean()
    pos_not_found_counts_smoothed = pd.Series(pos_not_found_counts).rolling(window=window_size, min_periods=1).mean()
    
    neg_true_counts_smoothed = pd.Series(neg_true_counts).rolling(window=window_size, min_periods=1).mean()
    neg_false_counts_smoothed = pd.Series(neg_false_counts).rolling(window=window_size, min_periods=1).mean()
    neg_not_found_counts_smoothed = pd.Series(neg_not_found_counts).rolling(window=window_size, min_periods=1).mean()
    
    # Set up the plot
    fig = plt.figure(figsize=(12, 7))
    
    # Plot positive distance data and rolling mean
    plt.plot(pos_distances, pos_true_counts, label="True (Positive)", color="green", marker='o', linestyle="--")
    plt.plot(pos_distances, pos_false_counts, label="False (Positive)", color="red", marker='o', linestyle="--")
    plt.plot(pos_distances, pos_not_found_counts, label="NOT FOUND (Positive)", color="blue", marker='o', linestyle="--")
    
    #plt.plot(pos_distances, pos_true_counts_smoothed, label="True (Positive, Rolling Mean)", color="darkgreen", linestyle="-")
    #plt.plot(pos_distances, pos_false_counts_smoothed, label="False (Positive, Rolling Mean)", color="darkred", linestyle="-")
    #plt.plot(pos_distances, pos_not_found_counts_smoothed, label="NOT FOUND (Positive, Rolling Mean)", color="darkblue", linestyle="-")
    
    # Plot negative distance data and rolling mean
    #plt.plot(neg_distances, neg_true_counts, label="True (Negative)", color="lightgreen", marker='o', linestyle="--")
    #plt.plot(neg_distances, neg_false_counts, label="False (Negative)", color="lightcoral", marker='o', linestyle="--")
    #plt.plot(neg_distances, neg_not_found_counts, label="NOT FOUND (Negative)", color="lightblue", marker='o', linestyle="--")
    
    plt.plot(neg_distances, neg_true_counts_smoothed, label="True (Negative, Rolling Mean)", color="darkgreen", linestyle="-.")
    plt.plot(neg_distances, neg_false_counts_smoothed, label="False (Negative, Rolling Mean)", color="darkred", linestyle="-.")
    plt.plot(neg_distances, neg_not_found_counts_smoothed, label="NOT FOUND (Negative, Rolling Mean)", color="darkblue", linestyle="-.")
    
    # Label the axes and title
    plt.xlabel("Distance")
    plt.ylabel("Count")
    plt.ylim(0,100) 
    plt.title("True, False, and NOT FOUND Counts by Distance Sign (with Rolling Mean)")
    
    # Show legend
    plt.legend()
    
    # Show grid for better readability
    plt.grid(True)
    
    # Set the window title
    fig.canvas.manager.set_window_title(title)
    # Save the plot as a PDF
    plt.savefig(title + ".pdf")



#directory = "/Users/rrobbes/Projects/reachability/llama.cpp-master/output/reachability_questions.txt_2024-11-07_14-59-24"
#
#if len(sys.argv) > 1:
#    directories = sys.argv[1:]
#else:
#    print("default dir")

import itertools

def generate_letter_sequences(n):
    # Determine the length of sequences needed based on the required count n
    sequence_length = 1
    while (26 ** sequence_length) < n:
        sequence_length += 1
    
    # Use itertools.product to create combinations
    combinations = itertools.product("abcdefghijklmnopqrstuvwxyz", repeat=sequence_length)
    # Join each tuple of letters into a string and limit to n sequences
    return ["".join(combo) for combo in itertools.islice(combinations, n)]

def count_subdirs(directory):
    return sum(1 for entry in os.listdir(directory) if os.path.isdir(os.path.join(directory, entry)))

def merge_dicts_unique(dict1, dict2, prefix):
    for key, value in dict2.items():
        unique_key = prefix + key
        dict1[unique_key] = value
    return dict1

def analyze_experiment(dir):
    # Convert the base directory to a Path object
    base_path = Path(dir)
    all_results = {}
    # Iterate over each subdirectory in the base directory
    n_dirs = count_subdirs(dir)
    seq = generate_letter_sequences(n_dirs)
    for subdir in base_path.iterdir():
        if subdir.is_dir():
            # Check if there's an 'output' subdirectory in this subdir
            output_dir = subdir / 'output'
            if output_dir.is_dir():
                results = search_answer_in_files(output_dir)  # Apply the function to the output directory
                merge_dicts_unique(all_results, results, seq.pop())
    print("dict size", len(all_results))
    plot_analysis_by_distance_with_signs(analyze_results_by_distance(all_results), dir)




#all_results = []
#for i in directories:
#    results = search_answer_in_files(i)
#    all_results.extend(results)
#    print_last_100_chars_directory(i, "NOT_FOUND")

#results_cleaned = {filename: result for filename, result in results.items() if result != "NOT FOUND"}

#plot_analysis_by_distance_with_signs(analyze_results_by_distance(search_answer_in_files(directory)))

#plot_analysis_by_distance_with_signs(analyze_results_by_distance(results))

for dir in sys.argv[1:]:
    analyze_experiment(dir)
plt.show()


### get fine-grained stats of distance without rolling mean
### do the file moving just once, not over and over ...
### somehow performance at distance 1 is very very bad! check ....
### we need more precise distances than just "all distances"