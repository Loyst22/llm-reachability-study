import os
import time
import pathlib
import random
import sys
from datetime import datetime
from llama_cpp import Llama

# ============================================================== 
# Utility functions 
# ==============================================================

def trim(s: str) -> str:
    return s.strip()

def read_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error: Could not open the file {path}: {e}")
        return ""

def read_file_from_env_directory(filename: str, work_dir_var: str = "LLAMA_WORK_DIR") -> str:
    dir_path = os.getenv(work_dir_var)
    if not dir_path:
        print(f"Error: Environment variable {work_dir_var} is not set.")
        return ""
    file_path = pathlib.Path(dir_path) / filename
    
    for encoding in ["utf-8", "cp1252"]:  # Handle Windows/Linux encodings
        try:
            with open(file_path, "r", encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    print(f"Error: Could not decode {file_path} with utf-8 or cp1252.")
    return ""

def append_result_to_big_file(sub_dir: str, file_name: str, seq_id: int, distance: int, question: str, response: str, work_dir_var: str = "LLAMA_WORK_DIR"):
    base_dir = os.getenv(work_dir_var)
    if not base_dir:
        print(f"Error: Environment variable {work_dir_var} is not set.")
        return

    full_dir = pathlib.Path(base_dir) / sub_dir
    full_dir.mkdir(parents=True, exist_ok=True)
    file_path = full_dir / file_name

    try:
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(f"=== seq={seq_id}, distance={distance} ===\n")
            f.write(f"Q: {question}\n")
            f.write(f"A:\n{response}\n\n")
        print(f"Appended result to {file_path}")
    except Exception as e:
        print(f"Error: Unable to write to {file_path}: {e}")

def split_string(s: str, delimiter: str) -> list[str]:
    return s.split(delimiter)

# ============================================================== 
# Client structure 
# ==============================================================

class Client:
    def __init__(self, cid: int):
        self.id = cid
        self.seq_id = -1
        self.input = ""
        self.prompt = ""
        self.response = ""
        self.distance = 0
        self.start_time = 0.0

# ============================================================== 
# Main simulation logic 
# ==============================================================

def main():
    random.seed(1234)

    # --- Load system prompt and questions ---
    system_prompt = read_file_from_env_directory("system.txt")
    questions = read_file_from_env_directory("reachability_questions.txt")
    q_prompts = split_string(questions.strip(), "\n")

    # --- Init llama model ---
    model_path = os.getenv("LLAMA_MODEL", "../models/Mistral-7B-Instruct-v0.3.IQ1_S.gguf")
    model_name = "Mistral-7B"
    llm = Llama(model_path=model_path, n_ctx=2048, n_threads=8)

    # --- Parallel clients setup ---
    n_clients = 2
    clients = [Client(i) for i in range(n_clients)]

    # Output directory + single file
    output_dir = model_name
    big_file = "all_results.txt"

    seq_counter = 0
    while seq_counter < len(q_prompts):
        for client in clients:
            if client.seq_id == -1 and seq_counter < len(q_prompts):
                parts = split_string(q_prompts[seq_counter], "\t")
                if len(parts) >= 2:
                    client.distance = int(parts[0])
                    client.input = parts[1]
                else:
                    client.distance = 0
                    client.input = q_prompts[seq_counter]

                client.prompt = f"{system_prompt}\nUser: {client.input}\nAssistant:"
                client.seq_id = seq_counter
                client.start_time = time.time()

                print(f"Client {client.id} started seq {client.seq_id} with input: {client.input}")

                # Generate response
                output = llm(client.prompt, max_tokens=256, stop=["User:"])
                client.response = output["choices"][0]["text"]

                elapsed = time.time() - client.start_time
                print(f"Client {client.id}, seq {client.seq_id}, time {elapsed:.2f}s")

                # Append result to one big file
                append_result_to_big_file(output_dir, big_file, client.seq_id, client.distance, trim(client.input), trim(client.response))

                client.seq_id = -1
                seq_counter += 1

    print("All sequences processed.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python reachability_bench.py /path/to/dir_containing_txt_files")
        sys.exit(1)

    provided_dir = pathlib.Path(sys.argv[1]).expanduser().resolve()
    if not provided_dir.exists() or not provided_dir.is_dir():
        print(f"Provided path is not a directory: {provided_dir}", file=sys.stderr)
        sys.exit(1)

    os.environ["LLAMA_WORK_DIR"] = str(provided_dir)
    main()
