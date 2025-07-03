import subprocess
import os
import sys
import threading

def stream_to_file(stream, filepath):
    """Reads from a stream and writes to a file in real time."""
    with open(filepath, "w", encoding="latin1", errors="replace") as f:
        for line in iter(stream.readline, ''):
            if line:
                f.write(line)
                f.flush()
    stream.close()
    
def extract_assistant_answer(full_text):
    """
    Extract only the assistant's answer from the output text.
    Assumes that the assistant's answer starts after a line starting with 'assistant' or 'A:' or after the prompt.
    """
    import re

    # Try to find the assistant block starting point
    # For example: find "assistant\n" or "A:" or "assistant\nLet's think step"
    pattern = re.compile(r'(assistant\s*\n|A:\s*\n)', re.IGNORECASE)
    match = pattern.search(full_text)

    if match:
        start_idx = match.end()
        answer = full_text[start_idx:].strip()
        return answer
    else:
        # fallback: try to find 'Let's think step' as start
        idx = full_text.find("Let's think step")
        if idx != -1:
            return full_text[idx:].strip()
        else:
            # If nothing found, return the whole text
            return full_text.strip()

def run_one_prompt(dir, question, prompt, idx, model_with_name, distance):
    print(f"Running prompt {idx} in: {dir}")

    llama_executable = os.path.abspath("../llama-cpp-win/llama-cli.exe")
    
    model = model_with_name[0]    
    model_path = os.path.abspath(os.path.join("..", "models", model))

    env = os.environ.copy()
    env["PATH"] = os.path.dirname(llama_executable) + ";" + env["PATH"]
    env["LLAMA_WORK_DIR"] = os.path.abspath(dir)

    # Prépare la commande sans prompt en argument (on passe par stdin)
    command = [
        llama_executable,
        # "--threads", "8",
        # "-context-size", "100000",
        "-c", "32768",
        "-m", model_path,
        "-ngl", "32",
        "--n-predict", "512",
        "--prompt", prompt,
        "-no-cnv"
    ]

    # On lance le processus en pipe pour stdin, stdout, stderr
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='latin1',
        errors="replace",
        env=env
    )

    # Ecrit le prompt dans stdin et ferme stdin pour signaler la fin
    # process.stdin.write(prompt)
    # process.stdin.close()
    
    # Fichiers de sortie
    output_file = os.path.join(dir, f"result{idx}_{distance}.txt")
    stdout_file = os.path.join(dir, f"stdout_{idx}.txt")
    
    stderr_dir = os.path.join(dir, "stderr")
    stderr_file = os.path.join(stderr_dir, f"stderr_{idx}.txt")
    
    # If the output files already exists, remove them
    if os.path.exists(output_file):
        os.remove(output_file)
    if os.path.exists(stdout_file):
        os.remove(stdout_file)
    if os.path.exists(stderr_file):
        os.remove(stderr_file)
    

    # Threads pour récupérer stdout et stderr en live dans des fichiers
    t_out = threading.Thread(target=stream_to_file, args=(process.stdout, stdout_file))
    t_err = threading.Thread(target=stream_to_file, args=(process.stderr, stderr_file))
    t_out.start()
    t_err.start()

    # Attendre la fin du processus et des threads
    process.wait()
    t_out.join()
    t_err.join()
    
    # Copier stdout dans output_file (ou inversement)
    # Ici, on met dans output_file le contenu de stdout pour simplifier
    os.rename(stdout_file, output_file)
    # On peut aussi garder stdout_file, à toi de choisir si besoin

    # Now trim output file after line "assistant"
    with open(output_file, 'r', encoding='latin1', errors='replace') as f:
        lines = f.readlines()

    if "Coder" in model_with_name[1]:
        # Find line with exactly "assistant" (case insensitive, stripped)
        for i, line in enumerate(lines):
            if line.strip().lower() == "assistant":
                # Keep everything AFTER this line
                trimmed_lines = lines[i+1:]
                break
        else:
            # If no "assistant" line found, keep full content
            trimmed_lines = lines
    
    elif model_with_name[1] == "Mistral-7B":
        # Find line with exactly "assistant" (case insensitive, stripped)
        for i, line in enumerate(lines):
            if line.strip().lower() == "<|im_start|>assistant":
                # Keep everything AFTER this line
                trimmed_lines = lines[i+1:]
                break
        else:
            # If no "assistant" line found, keep full content
            trimmed_lines = lines

    with open(output_file, 'w', encoding='latin1', errors='replace') as f:
        f.write(question.strip() + '\n\n')
        f.writelines(trimmed_lines)

    print(f"Finished prompt {idx} in: {dir}")

def run_one_dir(dir, model_with_name):
    print("Running in:", dir)
    print("With model:", model_with_name[1])

    system_txt_path = os.path.join(dir, "system.txt")
    questions_path = os.path.join(dir, "reachability_questions.txt")
    if not os.path.exists(system_txt_path) or not os.path.exists(questions_path):
        print(f"Missing system.txt or reachability_questions.txt in {dir}, skipping")
        return
    
    model_dir = os.path.join(dir, model_with_name[1])
    stderr_dir = os.path.join(model_dir, "stderr")

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    if not os.path.exists(stderr_dir):
        os.mkdir(stderr_dir)

    system_txt = open(system_txt_path, encoding='latin1', errors='replace').read()
    questions = open(questions_path, encoding='latin1', errors='replace').readlines()
    

    for idx, line in enumerate(questions):
        # Enlever le nombre et tabulation au début
        question = line.strip()
        question = question.split('\t', 1)
        if len(question) == 2:
            distance = question[0]
            question = question[1]
        else:
            question = question[0]

        prompt = system_txt + "\n" + question
        
        # Format the chat prompt correctly
        chat_prompt = (
            f"<|im_start|>system\n"
            f"{system_txt}\n"
            f"<|im_end|>\n"
            f"<|im_start|>user\n"
            f"{question}\n"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        run_one_prompt(model_dir, question, chat_prompt, idx, model_with_name, distance)
    
    # After all prompts, move stdout and stderr files if they remain here
    for filename in os.listdir(dir):
        if filename.startswith("result") and filename.endswith(".txt"):
            os.rename(os.path.join(dir, filename), os.path.join(model_dir, filename))
        elif filename.startswith("stderr") and filename.endswith(".txt"):
            os.rename(os.path.join(dir, filename), os.path.join(stderr_dir, filename))


def get_subdirectories(directory):
    return [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]


def run_experiment(directory, model_with_name):
    # If this directory has system.txt and reachability_questions.txt, run and stop recursion
    if run_one_dir(directory, model_with_name):
        return

    # Else, recurse into subdirectories
    for entry in os.listdir(directory):
        full_path = os.path.join(directory, entry)
        if os.path.isdir(full_path):
            run_experiment(full_path, model_with_name)

def run_experiments(directory, models_with_names):
    for model_with_name in models_with_names:
        run_experiment(directory, model_with_name)

if __name__ == "__main__":
    
    # model_with_name = ["Mistral-7B-Instruct-v0.3.IQ1_S.gguf", "Mistral-7B"]
    coder7B = ["qwen2.5-coder-7b-instruct-q4_k_m.gguf", "Coder-7B"]
    coder3B = ["qwen2.5-coder-3b-instruct-q4_k_m.gguf", "Coder-3B"]
    # model_with_name = ["qwen2.5-coder-1.5b-instruct-q4_k_m.gguf", "Coder-1.5B"]
    # model_with_name = ["qwen2.5-Coder-0.5B-Instruct-Q4_K_L", "Coder-0.5B"]
    
    if len(sys.argv) > 1:
        for directory in sys.argv[1:]:
            run_experiments(directory, [coder3B, coder7B])
            # run_experiment(directory, coder3B)
    else:
        print("Usage: python run_experiment.py <directory1> [<directory2> ...]")
