import os
import pathlib
import random
import sys
import time
from threading import Thread
from queue import Queue
from llama_cpp import Llama

# ==========================
# Utilities
# ==========================
def read_file_from_env_directory(filename: str):
    dir_path = os.getenv("LLAMA_WORK_DIR")
    file_path = pathlib.Path(dir_path) / filename
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""

def append_result(output_dir, file_name, seq_id, distance, question, response):
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = pathlib.Path(output_dir) / file_name
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"=== seq={seq_id}, distance={distance} ===\n")
        f.write(f"Q: {question}\nA:\n{response}\n\n")

# ==========================
# Worker Thread
# ==========================
class Worker(Thread):
    def __init__(self, cid, task_queue, system_prompt, model_path, output_dir, big_file):
        super().__init__()
        self.cid = cid
        self.task_queue = task_queue
        self.system_prompt = system_prompt
        self.model_path = model_path
        self.output_dir = output_dir
        self.big_file = big_file
        # Each thread must have its own Llama instance
        self.llm = Llama(model_path=self.model_path, n_ctx=2048, n_threads=4)

    def run(self):
        while not self.task_queue.empty():
            try:
                seq_id, distance, question = self.task_queue.get_nowait()
            except:
                break

            prompt = f"{self.system_prompt}\nUser: {question}\nAssistant:"
            start = time.time()
            output = self.llm(prompt, max_tokens=256, stop=["User:"])
            response = output["choices"][0]["text"]
            elapsed = time.time() - start
            print(f"[Worker {self.cid}] seq={seq_id}, time={elapsed:.2f}s")
            append_result(self.output_dir, self.big_file, seq_id, distance, question, response)
            self.task_queue.task_done()

# ==========================
# Main Function
# ==========================
def main():
    random.seed(1234)

    # Read prompts
    system_prompt = read_file_from_env_directory("system.txt")
    questions_raw = read_file_from_env_directory("reachability_questions.txt").splitlines()

    # Fill queue with tasks
    tasks = Queue()
    for seq_id, q in enumerate(questions_raw):
        parts = q.split("\t")
        if len(parts) >= 2:
            distance = int(parts[0])
            question = parts[1]
        else:
            distance = 0
            question = q
        tasks.put((seq_id, distance, question))

    # Model & output settings
    model_path = os.getenv("LLAMA_MODEL", "../models/Mistral-7B-Instruct-v0.3.IQ1_S.gguf")
    output_dir = "results"
    big_file = "all_results.txt"

    # Start worker threads
    n_workers = 2
    workers = [Worker(i, tasks, system_prompt, model_path, output_dir, big_file) for i in range(n_workers)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()

    print("All sequences processed.")

# ==========================
# Entry Point
# ==========================
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python reachability_bench.py /path/to/data")
        sys.exit(1)
    os.environ["LLAMA_WORK_DIR"] = sys.argv[1]
    main()
