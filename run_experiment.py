import subprocess
import os
import sys
import select
import time

def run_one_dir(dir):
    print("dir is", dir)
    where_is_llama =  "../llama.cpp-master"
    where_is_llama =  "../newllama/llama.cpp/build/bin"
    # Define the environment variable just for the subprocess
    env = os.environ.copy()
    env["PATH"] = where_is_llama + ":" + os.environ["PATH"]
    env["LLAMA_WORK_DIR"] = "./" + dir

    model = "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"
    model = "Qwen2.5-Coder-1.5B.Q8_0.gguf"
    model = "Qwen2.5-Coder-7B-Instruct.Q8_0.gguf"
    model = "qwen2.5-coder-32b-instruct-q8_0.gguf"
    model ="qwen2.5-coder-14b-instruct-q8_0.gguf"
    model = "Mistral-Small-3.1-24B-Instruct-2503-Q6_K.gguf"
    #model = "Qwen2.5-Coder-1.5B.Q8_0.gguf"


    # Call the program with the specific environment
    command =  f"reachability-bench -m ../models/{model} -ns 60 -np 42 -b 50000 -c 100000"
    command =  f"reachability_bench -m ../models/{model} -ns 60 -np 42 -b 100000 -c 100000 --n-predict 1000"
    print(command)


    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        encoding='latin1',
        errors="replace"
    )

    # Use select to monitor both stdout and stderr
    while True:
        # Check if both streams are still open
        rlist = []
        if process.stdout and not process.stdout.closed:
            rlist.append(process.stdout)
        if process.stderr and not process.stderr.closed:
            rlist.append(process.stderr)

        if rlist:
            # Wait for either stdout or stderr to have data ready
            ready_to_read, _, _ = select.select(rlist, [], [])
            
            for stream in ready_to_read:
                output = stream.readline()
                if output:  # Check if there's any output
                    print(output, end='')  # Print each line as it appears
                else:
                    # If output is empty, it means the stream is done (EOF)
                    if stream == process.stdout:
                        process.stdout.close()
                    if stream == process.stderr:
                        process.stderr.close()

        # Check if the process has finished
        if process.poll() is not None and process.stdout.closed and process.stderr.closed:
            break


def get_subdirectories(directory):
    # List all entries in the directory
    return [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

def run_experiment(directory):
    dirs = get_subdirectories(directory)
    for dir in dirs:
        run_one_dir(directory + "/" + dir)

if len(sys.argv) > 1:
    for directory in sys.argv[1:]:
        run_experiment(directory)
        # cool down a bit
        # not needed on JZ
        # time.sleep(120)
    
else:
    print("no work to do!")
# Wait for the process to finish and also check for errors
#stderr = process.stderr.read()

# Check if there was any error
#if stderr:
#    print("Error:", stderr)