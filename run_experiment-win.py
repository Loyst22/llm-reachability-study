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

def run_one_prompt(dir, prompt, idx):
    print(f"Running prompt {idx} in: {dir}")

    llama_executable = os.path.abspath("../llama-cpp-win/llama-run.exe")
    model = "qwen2.5-coder-7b-instruct-q4_k_m.gguf"
    # model = "qwen2.5-coder-3b-instruct-q4_k_m.gguf"
    # model = "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
    # model = "qwen2.5-Coder-0.5B-Instruct-Q4_K_L" 
    
    model_path = os.path.abspath(os.path.join("..", "models", model))

    env = os.environ.copy()
    env["PATH"] = os.path.dirname(llama_executable) + ";" + env["PATH"]
    env["LLAMA_WORK_DIR"] = os.path.abspath(dir)

    # Prépare la commande sans prompt en argument (on passe par stdin)
    command = [
        llama_executable,
        "--ngl", "32",
        # "--threads", "8",
        "--context-size", "100000",
        model_path
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
    process.stdin.write(prompt)
    process.stdin.close()

    # Fichiers de sortie
    output_file = os.path.join(dir, f"output_{idx}.txt")
    stdout_file = os.path.join(dir, f"stdout_{idx}.txt")
    stderr_file = os.path.join(dir, f"stderr_{idx}.txt")

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

    print(f"Finished prompt {idx} in: {dir}")

def run_one_dir(dir):
    print("Running in:", dir)

    system_txt_path = os.path.join(dir, "system.txt")
    questions_path = os.path.join(dir, "reachability_questions.txt")

    if not os.path.exists(system_txt_path) or not os.path.exists(questions_path):
        print(f"Missing system.txt or reachability_questions.txt in {dir}, skipping")
        return

    system_txt = open(system_txt_path, encoding='latin1', errors='replace').read()
    questions = open(questions_path, encoding='latin1', errors='replace').readlines()

    for idx, line in enumerate(questions):
        # Enlever le nombre et tabulation au début
        question = line.strip()
        question = question.split('\t', 1)
        if len(question) == 2:
            question = question[1]
        else:
            question = question[0]

        prompt = system_txt + "\n" + question

        run_one_prompt(dir, prompt, idx)


def get_subdirectories(directory):
    return [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]


def run_experiment(directory):
    # If this directory has system.txt and reachability_questions.txt, run and stop recursion
    if run_one_dir(directory):
        return

    # Else, recurse into subdirectories
    for entry in os.listdir(directory):
        full_path = os.path.join(directory, entry)
        if os.path.isdir(full_path):
            run_experiment(full_path)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        for directory in sys.argv[1:]:
            run_experiment(directory)
    else:
        print("Usage: python run_experiment.py <directory1> [<directory2> ...]")
