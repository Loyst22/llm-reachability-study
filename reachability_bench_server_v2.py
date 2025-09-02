from concurrent.futures import ThreadPoolExecutor, as_completed
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
#       Utilities
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

def append_file_to_env_model_dir(filename: str, content: str, model_name: str, work_dir_var: str = "LLAMA_WORK_DIR"):
    """Append text to a file under the env/model_name directory."""
    work_dir = os.getenv(work_dir_var)
    if not work_dir:
        print(f"Error: Environment variable {work_dir_var} is not set.")
        return
    
    output_dir = pathlib.Path(work_dir) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)  # create dir if missing

    file_path = output_dir / filename
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(content + "\n")  # add newline separator

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

def wait_for_server(url="http://localhost:8080/health", timeout=60, interval=1):
    """
    Wait until the LLaMA server is ready.
    Default health endpoint is /health.
    """
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url)
            if r.status_code == 200:
                print("✅ Server is ready")
                return True
        except requests.RequestException:
            pass
        print("⌛ Waiting for server...")
        time.sleep(interval)
    raise TimeoutError(f"Server not ready after {timeout} seconds")

# ==========================
#       Question worker
# ==========================
def ask_question(seq_id, distances, actual_question, system_prompt, model_name, timings_list):
    prompt = f"{system_prompt}\n{actual_question}"

    payload = {
        "prompt": [prompt],
        "n_predict": 4096,
        "cache_prompt": True,
        # "id_slot": 0,   # ! IMPORTANT: must be -1 (auto) if multiple slots
        "stop": ["User:", "YES", "NO"],
        "n_keep": ctx_size
    }

    try:
        print(f"[Q{seq_id}] ⌛ Starting")
        start_total = time.perf_counter()
        response = requests.post("http://localhost:8080/completions", json=payload)
        response.raise_for_status()
        end_total = time.perf_counter()
        
        result = response.json()
        answer = result.get("content", "").strip()
        stopped_word = result.get("stopping_word", "")
        if stopped_word in ["YES", "NO"]:
            if not answer.endswith(" "):
                answer += " "
            answer += stopped_word

        distance_str = ", ".join(distances)
        global_content = f"[Q{seq_id}] Distance={distance_str}\nQuestion: {actual_question}\nAnswer: {answer}\n"
        append_file_to_env_model_dir("results.txt", global_content, model_name)
        
        # Record timing
        generation_time = result.get("timing", {}).get("generation_time", end_total - start_total)
        total_time = end_total - start_total
        timings_list.append({
            "id": seq_id,
            "generation_time": generation_time,
            "total_time": total_time
        })

        print(f"[Q{seq_id}] ✅ done")
    except requests.RequestException as e:
        print(f"[Q{seq_id}] ❌ Error:", e)


# ==========================
#       Main Function
# ==========================


sys_token_count, _ = count_tokens_in_file(r"..\models\Mistral-7B-Instruct-v0.3.IQ1_S.gguf",
                                          r"..\llama-cpp-win\llama-tokenize.exe",
                                          r".\experiments\adv_lin\ctx_10_depths_1--8_com_0_var_0_loop_0_if_0_qs_0--16_java\system.txt")
    
# Add padding (for question)
token_count = sys_token_count + 100
# Add padding (for response)  
token_count += 500
# Take closest power of 2 above
next_pow2 = 1 << (token_count - 1).bit_length()

# Take into account the number of slots
n_parallel = 2
ctx_size = n_parallel*next_pow2

def main():
    random.seed(1234)

    # Read prompts
    system_prompt = read_file_from_env_directory("system.txt")
    questions_raw = read_file_from_env_directory("reachability_questions.txt").splitlines()
    
    # print("System prompt:", system_prompt)
    # print("Questions:", questions_raw)
    
    # Start llama-server in a subprocess
    server_cmd = [
        r"..\llama-cpp-win-newer\llama-server.exe",
        "--model", r"..\models\Mistral-7B-Instruct-v0.3.IQ1_S.gguf",
        "--ctx-size", str(ctx_size), # Total ctx so divide by parallel to get ctx_slot, ex: 32768=4096*8
        "--keep", str(sys_token_count+10),
        "--gpu-layers", "24",
        "--parallel", str(n_parallel),
        "--cache-reuse", "128",
        "--port", "8080", # Default is 8080 but just to make sure
        "--kv-unified",
        "--no-warmup",
    ]

    print("Starting LLaMA server...")
    server_proc = subprocess.Popen(server_cmd)
    wait_for_server()
    
    model_name = "Mistral-7B-Server-Parallel-4"
    timings_list = []
    start_all = time.time()
    
    # For effective KV caching, we process the first question before starting the parallel computations
    """
    question_line = questions_raw[0]
    parts = question_line.split("\t")
    actual_question = parts[1]
    distances = [parts[0]] + (parts[2:] if len(parts) > 2 else [])
    
    ask_question(0, distances, actual_question, system_prompt, model_name, timings_list)
    """
    

    with ThreadPoolExecutor(max_workers=n_parallel) as executor: # Check if it should be a lower max_workers value (8/None)
        futures = []
        for seq_id, question_line in enumerate(questions_raw): # questions_raw[1:]):
            # seq_id += 1 # To take into account the first seq_id
            try:
                parts = question_line.split("\t")
                actual_question = parts[1]
                distances = [parts[0]] + (parts[2:] if len(parts) > 2 else [])
            except ValueError:
                print(f"Skipping malformed question line: {question_line}")
                continue
            futures.append(executor.submit(ask_question, seq_id, distances, actual_question, system_prompt, model_name, timings_list))
            
        # Wait for all to finish
        for future in as_completed(futures):
            pass
    
    end_all = time.time()
    
    # Print all timings
    print("\n=== Per-request timings ===")
    for t in sorted(timings_list, key=lambda x: x["id"]):
        print(f"ID {t['id']:>2}: generation_time={t['generation_time']:.2f}s, total_time={t['total_time']:.2f}s")

    total_gen_time = sum(t["generation_time"] for t in timings_list)
    total_elapsed_time = end_all - start_all
    print(f"\nTotal generation time (sum of requests): {total_gen_time:.2f}s")
    print(f"Total elapsed wall-clock time: {total_elapsed_time:.2f}s")
    
    # Stop the server
    server_proc.terminate()
    server_proc.wait()

# ==========================
#       Entry Point
# ==========================
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python reachability_bench_server.py /path/to/data")
        sys.exit(1)
    os.environ["LLAMA_WORK_DIR"] = sys.argv[1]
    main()