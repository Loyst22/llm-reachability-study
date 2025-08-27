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
    """Trim whitespace from beginning and end of string."""
    return s.strip()


def read_file_from_env_directory(filename: str, work_dir_var: str = "LLAMA_WORK_DIR") -> str:
    """Read file from directory specified in environment variable."""
    dir_path = os.getenv(work_dir_var)
    if not dir_path:
        print(f"Error: Environment variable {work_dir_var} is not set.")
        return ""
    file_path = pathlib.Path(dir_path) / filename
    
    for encoding in ["utf-8", "cp1252"]: # Small issue with windows encoding....
        try:
            with open(file_path, "r", encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    print(f"Error: Could not decode {file_path} with utf-8 or cp1252.")
    return ""


def write_result_to_env_directory(sub_dir: str, file_name: str, question: str, response: str, work_dir_var: str = "LLAMA_WORK_DIR"):
    """Write results (Q + A) into environment-defined directory."""
    base_dir = os.getenv(work_dir_var)
    if not base_dir:
        print(f"Error: Environment variable {work_dir_var} is not set.")
        return

    full_dir = pathlib.Path(base_dir) / sub_dir
    full_dir.mkdir(parents=True, exist_ok=True)
    file_path = full_dir / file_name

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"Q: {question}\nA:\n{response}\n")
        print(f"Content written to {file_path}")
    except Exception as e:
        print(f"Error: Unable to write to {file_path}: {e}")


def time_stamped_name(prefix: str) -> str:
    """Generate a timestamped name with prefix."""
    now = datetime.now()
    return f"{prefix}_{now.strftime('%Y-%m-%d_%H-%M-%S')}"


def split_string(s: str, delimiter: str) -> list[str]:
    """Split string into tokens by delimiter."""
    return s.split(delimiter)


# ==============================================================
# Client structure
# ==============================================================
class Client:
    """Simulated client holding state of request/response."""

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
    # NOTE: model can be adjusted here
    model_path = os.getenv("LLAMA_MODEL", "../models/Mistral-7B-Instruct-v0.3.IQ1_S.gguf")
    # model_path = os.getenv("LLAMA_MODEL", "../models/qwen2.5-coder-7b-instruct-q4_k_m.gguf")
    model_name = "Mistral-7B"
    # model_name = "Qwen-Coder-7B"
    llm = Llama(model_path=model_path, n_ctx=2048, n_threads=8)

    # --- Parallel clients setup ---
    n_clients = 2  # could be param
    clients = [Client(i) for i in range(n_clients)]

    # Output directory
    # output_dir = "output-" + pathlib.Path(model_path).stem # Older version, could be better
    output_dir = model_name

    # --- Processing loop ---
    seq_counter = 0
    while seq_counter < len(q_prompts):
        for client in clients:
            if client.seq_id == -1 and seq_counter < len(q_prompts):
                # Assign new sequence to this client
                parts = split_string(q_prompts[seq_counter], "\t")
                if len(parts) >= 2:
                    client.distance = int(parts[0])
                    client.input = parts[1]
                else:
                    client.distance = 0
                    client.input = q_prompts[seq_counter]

                client.prompt = f"{system_prompt}\nUser: {client.input}\nAssistant:"
                
                # print("Client prompt is :", client.prompt)
                client.seq_id = seq_counter
                client.start_time = time.time()

                print(f"Client {client.id} started seq {client.seq_id} with input: {client.input}")

                # Generate response
                output = llm(client.prompt, max_tokens=256, stop=["User:"])
                client.response = output["choices"][0]["text"]

                elapsed = time.time() - client.start_time

                print(f"Client {client.id}, seq {client.seq_id}, time {elapsed:.2f}s")
                print(f"Q: {client.input}\nA: {client.response}\n")

                # Write result
                fname = f"result{client.seq_id}_{client.distance}.txt"
                write_result_to_env_directory(output_dir, fname, trim(client.input), trim(client.response))

                # Reset client for next question
                client.seq_id = -1

                seq_counter += 1

    print("All sequences processed.")


if __name__ == "__main__":
    # Minimal change: accept 1 argument as the directory that contains the .txt files,
    # set the LLAMA_WORK_DIR environment variable accordingly, then run main().
    if len(sys.argv) != 2:
        print("Usage: python reachability_bench.py /path/to/dir_containing_txt_files")
        sys.exit(1)

    provided_dir = pathlib.Path(sys.argv[1]).expanduser().resolve()
    if not provided_dir.exists() or not provided_dir.is_dir():
        print(f"Provided path is not a directory: {provided_dir}", file=sys.stderr)
        sys.exit(1)

    # Set the environment variable used by the helper functions (no other changes).
    os.environ["LLAMA_WORK_DIR"] = str(provided_dir)
    
    main()
