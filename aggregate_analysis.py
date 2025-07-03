import os
import re
import csv
import sys

def parse_directory_structure(base_path):
    data = []
    max_depth = calculate_max_depth(base_path)  # Calculate max depth upfront

    for root, dirs, files in os.walk(base_path):
        # Extract properties from the directory structure
        path_hierarchy = extract_hierarchy_from_path(root, base_path)

        for file_name in files:
            if file_name.startswith("result") and file_name.endswith(".txt"):
                file_properties = extract_properties_from_file(file_name)
                # Combine hierarchy properties with file properties
                combined_properties = {**path_hierarchy, **file_properties}
                result_file_path = os.path.join(root, file_name)
                # Add the new file path column
                combined_properties['context_file_path'] = generate_new_file_path(
                    result_file_path, file_properties.get('Batch number'), base_path, max_depth - 4
                )
                combined_properties['result_file_path'] = os.path.abspath(result_file_path)
                data.append(combined_properties)
    
    return data

def calculate_max_depth(base_path):
    """Calculate the maximum depth of the directory structure."""
    max_depth = 0
    for root, dirs, files in os.walk(base_path):
        relative_path = os.path.relpath(root, base_path)
        depth = len(relative_path.split(os.sep))
        max_depth = max(max_depth, depth)
    return max_depth

def extract_hierarchy_from_path(path, base_path):
    properties = {}
    relative_path = os.path.relpath(path, base_path)
    components = relative_path.split(os.sep)

    levels = ["_Experiment", "_Context size", "_Model", "_answer type", "answer category", "answer subcategory"]
    for i, component in enumerate(components):
        level_name = levels[i] # f"level_{i + 1}"
        # Match directory name in the format "DIRECTORY_NAME-N (P)": N amount, P percent
        match = re.match(r"(?P<name>.+?)-(?P<N>\d+) \((?P<P>\d+)\)", component)
        if match:
            properties[level_name] = match.group("name")
            properties[f"{level_name} Amount"] = int(match.group("N"))
            properties[f"{level_name} Percent"] = int(match.group("P"))
        else:
            if level_name == "_Context size":
                # TODO: use *rest to analyse data
                context, comments, *rest = re.findall(r'\d+', component)
                # context = re.findall(r'\d+', component)
                properties["_Context"] = context
                properties["_Comments"] = comments
            else:
                # Generic property for non-matching directories
                properties[level_name] = component
    
    return properties

def extract_properties_from_file(file_name):
    properties = {}
    # Match file name in the format "resultX-Y-(Z).txt", allowing Z to be negative
    match = re.match(r"result(?P<X>\d+)-(?P<Y>\d+)-\((?P<Z>-?\d+)\)\.txt", file_name)
    if match:
        properties['Answer number'] = int(match.group("X"))
        properties['Batch number'] = int(match.group("Y"))
        properties['Depth'] = int(match.group("Z"))
    return properties

def generate_new_file_path(file_path, X, base_path, max_depth):
    """Generate the new file path, truncated to the maximum depth."""
    if X is None:
        return None  # Skip if X is not defined

    # Determine relative path and truncate to max depth
    relative_path = os.path.relpath(file_path, base_path)
    path_parts = relative_path.split(os.sep)
    truncated_path = path_parts[:max_depth]

    # Add the new file name
    new_file_name = f"TheClass-{X}.java"
    new_file_path = os.path.abspath(os.path.join(base_path, *truncated_path, new_file_name))

    return new_file_path

def write_to_csv(data, output_file):
    # Gather all keys dynamically
    all_keys = set()
    for entry in data:
        all_keys.update(entry.keys())

    fieldnames = sorted(all_keys)

    with open(output_file, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for entry in data:
            writer.writerow(entry)



if __name__ == "__main__":
 
    base_path = "./exp_out/expanded"  # Change to your directory path
    #base_path = "./exp_out/tempxps"  # Change to your directory path
    #output_file = "./exp_out/output-aggregated.csv"

    base_path = sys.argv[1]
    print(base_path)
    output_file = f"{base_path}/aggregated.csv"
    #print(experiments)
    #analyze_experiments(experiments)

    data = parse_directory_structure(base_path)
    write_to_csv(data, output_file)

    print(f"Data has been written to {output_file}.")
