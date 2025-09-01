import os
import pathlib
import random
import re
import subprocess
import sys
import tempfile
import time
import requests

# ==========================
# Utilities
# ==========================
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

def write_file_to_env_model_dir(filename: str, content: str, model_name: str, work_dir_var: str = "LLAMA_WORK_DIR"):
    """Write a file under the env/model_name directory."""
    work_dir = os.getenv(work_dir_var)
    if not work_dir:
        print(f"Error: Environment variable {work_dir_var} is not set.")
        return
    
    output_dir = pathlib.Path(work_dir) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)  # create dir if missing

    file_path = output_dir / filename
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

def count_tokens_in_file(model_path, tokenizer_path, file_path):
    print(f"Tokenizing file: {file_path}")
    start_time = time.time()

    with open(file_path, "r", encoding="cp1252") as file:
        text = file.read()

    with tempfile.NamedTemporaryFile("w+", encoding="cp1252", delete=False, suffix=".txt") as tmp_file:
        tmp_file.write(text)
        tmp_file_path = tmp_file.name

    try:
        result = subprocess.run(
            [tokenizer_path, "-m", model_path, "-f", tmp_file_path],
            capture_output=True,
            text=True
        )
    finally:
        os.remove(tmp_file_path)

    if result.returncode != 0:
        print(f"❌ Error tokenizing {file_path}:\n{result.stderr}")
        return None, None

    output = result.stdout
    tokens = []
    for line in output.strip().split("\n"):
        match = re.match(r"\s*(\d+)\s*->", line)
        if match:
            tokens.append(int(match.group(1)))

    token_count = len(tokens)
    elapsed = time.time() - start_time
    print(f"✅ Token count: {token_count} (processed in {elapsed:.2f} seconds)\n")
    return token_count, output
    
# ==========================
# Main Function
# ==========================
def main():
    random.seed(1234)

    # Read prompts
    system_prompt = read_file_from_env_directory("system.txt")
    questions_raw = read_file_from_env_directory("reachability_questions.txt").splitlines()
    
    print("System prompt:", system_prompt)
    print("Questions:", questions_raw)
    
    count_tokens_in_file(r"..\models\Mistral-7B-Instruct-v0.3.IQ1_S.gguf",
                         r"..\llama-cpp-win\llama-tokenize.exe",
                         r".\experiments\adv_lin\ctx_10_depths_1--8_com_0_var_0_loop_0_if_0_qs_0--16_java\system.txt")
    
    # Start llama-server in a subprocess
    server_cmd = [
        r"..\llama-cpp-win\llama-server.exe",
        "--model", r"..\models\Mistral-7B-Instruct-v0.3.IQ1_S.gguf",
        "--ctx-size", "4096", # Total ctx so divide by parallel to get ctx_slot
        "--gpu-layers", "24",
        "--parallel", "1",
        "--cache-reuse", "128",
        "--no-warmup"
    ]

    print("Starting LLaMA server...")
    server_proc = subprocess.Popen(server_cmd)

    # Give the server a few seconds to start
    time.sleep(5)
    
    model_name = "Mistral-7B-Server-4"

    # Ask all the questions
    for seq_id, question_line in enumerate(questions_raw):
        # Expect "X\tquestion"
        try:
            distance, actual_question = question_line.split("\t", 1)
        except ValueError:
            print(f"Skipping malformed question line: {question_line}")
            continue
        
        # Prepare the system prompt + current question
        prompt = f"{system_prompt}\n{actual_question}"

        payload = {
            "prompt": [prompt],
            "n_predict": 4096,
            "cache_prompt": True,
            "id_slot": 0,
            "stop": ["User:", "YES", "NO"]
        }

        try:
            response = requests.post("http://localhost:8080/completions", json=payload)
            response.raise_for_status()
            result = response.json()
            processed_prompt = result.get("prompt", "").strip()
            answer = result.get("content", "").strip()
            # Append stopping word if it's YES or NO
            stopped_word = result.get("stopping_word", "")
            if stopped_word in ["YES", "NO"]:
                if not answer.endswith(" "):
                    answer += " "
                answer += stopped_word

            # Print in console
            print(f"\n[Q{seq_id}] Distance={distance}")
            # print("Question:", processed_prompt)
            # print("Answer:", answer)

            file_content = f"Question:\n{actual_question}\n\nAnswer:\n{answer}"
            write_file_to_env_model_dir(f"result{seq_id}_{distance}.txt", file_content, model_name)

        except requests.RequestException as e:
            print("Error while calling server:", e)

    # Stop the server
    server_proc.terminate()
    server_proc.wait()

# ==========================
# Entry Point
# ==========================
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python reachability_bench_server.py /path/to/data")
        sys.exit(1)
    os.environ["LLAMA_WORK_DIR"] = sys.argv[1]
    main()