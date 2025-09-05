// A basic application simulating a server with multiple clients.
// The clients submit requests to the server and they are processed in parallel.

#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "log.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include <ctime>
#include <algorithm>

// 17-23: added for reachability
#include <fstream>
#include <iostream>
#include <filesystem>
#include <cstdlib> 

static std::string WORK_DIR = "LLAMA_WORK_DIR";

// trim whitespace from the beginning and end of a string
static std::string trim(const std::string & str) {
    size_t start = 0;
    size_t end = str.size();

    while (start < end && isspace(str[start])) {
        start += 1;
    }

    while (end > start && isspace(str[end - 1])) {
        end -= 1;
    }

    return str.substr(start, end - start);
}

// 41-142: added for reachability
std::string read_file(const std::string& path) {
    std::ifstream file(path);  // Open file
    if (!file.is_open()) {
        std::cerr << "Error: Could not open the file!" << path << std::endl;
        return "";
    }

    std::ostringstream ss;
    ss << file.rdbuf();  // Read the file's buffer into the stringstream
    return ss.str();      // Return the stringstream's string content
}

std::string read_file_from_env_directory(const std::string& filename) {
    // Get the directory path from the environment variable
    const char* dir = std::getenv(WORK_DIR.c_str());
    if (!dir) {
        std::cerr << "Error: Environment variable " << WORK_DIR << " is not set." << std::endl;
        return "";
    }

    // Construct the full file path
    //std::__fs::filesystem::path full_path = std::__fs::filesystem::path(dir) / filename;
    // jean zay fix?
    std::filesystem::path full_path = std::filesystem::path(dir) / filename;
    
    // Open the file
    std::ifstream file(full_path);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open the file " << full_path << std::endl;
        return "";
    }

    // Read file content into a string
    std::ostringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

void write_result_to_env_directory(const std::string& subDir, const std::string& fileName, const std::string& question, const std::string& response) {
    // Retrieve the base directory path from the environment variable
    const char* baseDir = std::getenv(WORK_DIR.c_str());
    if (!baseDir) {
        std::cerr << "Error: Environment variable " << WORK_DIR << " is not set." << std::endl;
        return;
    }

    // Construct the full directory path by combining baseDir and subDir
    //std::__fs::filesystem::path fullDirPath = std::__fs::filesystem::path(baseDir) / subDir;
    // jean zay fix?
    std::filesystem::path fullDirPath = std::filesystem::path(baseDir) / subDir;

    // Ensure the full directory path exists
    //std::__fs::filesystem::create_directories(fullDirPath);
    std::filesystem::create_directories(fullDirPath);

    // Create the full file path
    //std::__fs::filesystem::path fullFilePath = fullDirPath / fileName;
    std::filesystem::path fullFilePath = fullDirPath / fileName;

    // Open the file and write the question and response
    std::ofstream outFile(fullFilePath);
    if (outFile) {
        outFile << "Q: " << question << "\n" << "A:\n" << response << "\n";
        std::cout << "Content written to " << fullFilePath << std::endl;
    } else {
        std::cerr << "Error: Unable to open file " << fullFilePath << std::endl;
    }
}

void append_result_to_env_directory(const std::string& subDir, int seq_id, const std::string& distance_str, const std::string& question, const std::string& response) {
    const char* baseDir = std::getenv(WORK_DIR.c_str());
    if (!baseDir) {
        std::cerr << "Error: Environment variable " << WORK_DIR << " is not set." << std::endl;
        return;
    }
    std::filesystem::path fullDirPath = std::filesystem::path(baseDir) / subDir;
    std::filesystem::create_directories(fullDirPath);
    std::filesystem::path fullFilePath = fullDirPath / "results";

    std::ofstream outFile(fullFilePath, std::ios::app);
    if (outFile) {
        outFile << "[Q" << seq_id << "] Distance=" << distance_str << "\n";
        outFile << "Question: " << question << "\n";
        outFile << "Answer: " << response << "\n\n";
    } else {
        std::cerr << "Error: Unable to open file " << fullFilePath << std::endl;
    }
}

std::string getFileNameWithoutExtension(const std::string& path) {
    //std::__fs::filesystem::path p(path);
    std::filesystem::path p(path);
    
    // Get the filename without the extension
    return p.stem().string();
}

void write_result(const std::string& subDir, const std::string& fileName, const std::string& question, const std::string& response) {
    // Ensure the subdirectory exists
    //std::__fs::filesystem::create_directories(subDir);
    std::filesystem::create_directories(subDir);

    // Create the full path by combining directory and file name
    std::string fullPath = subDir + "/" + fileName;

    // Open the file and write both strings, separated by a newline
    std::ofstream outFile(fullPath);
    if (outFile) {
        outFile << "Q:" << question << "\n" << "A:\n" << response << "\n";
        std::cout << "Content written to " << fullPath << std::endl;
    } else {
        std::cerr << "Error: Unable to open file " << fullPath << std::endl;
    }
}

std::string time_stamped_name(const std::string& prefix) {
    // Get current time as time_t (seconds since epoch)
    std::time_t t = std::time(nullptr);
    
    // Convert to struct tm for local time
    std::tm* localTime = std::localtime(&t);
    
    // Format the time into a string (e.g., "2024-11-07_15-30-45")
    char buffer[100];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%d_%H-%M-%S", localTime);
    
    // Combine the prefix with the formatted timestamp
    return prefix + "_" + buffer;
}

/* 144-165: commented out for reachability
static std::string k_system =
R"(Transcript of a never ending dialog, where the User interacts with an Assistant.
The Assistant is helpful, kind, honest, good at writing, and never fails to answer the User's requests immediately and with precision.

User: Recommend a nice restaurant in the area.
Assistant: I recommend the restaurant "The Golden Duck". It is a 5 star restaurant with a great view of the city. The food is delicious and the service is excellent. The prices are reasonable and the portions are generous. The restaurant is located at 123 Main Street, New York, NY 10001. The phone number is (212) 555-1234. The hours are Monday through Friday from 11:00 am to 10:00 pm. The restaurant is closed on Saturdays and Sundays.
User: Who is Richard Feynman?
Assistant: Richard Feynman was an American physicist who is best known for his work in quantum mechanics and particle physics. He was awarded the Nobel Prize in Physics in 1965 for his contributions to the development of quantum electrodynamics. He was a popular lecturer and author, and he wrote several books, including "Surely You're Joking, Mr. Feynman!" and "What Do You Care What Other People Think?".
User:)";

static std::vector<std::string> k_prompts = {
    "What is the meaning of life?",
    "Tell me an interesting fact about llamas.",
    "What is the best way to cook a steak?",
    "Are you familiar with the Special Theory of Relativity and can you explain it to me?",
    "Recommend some interesting books to read.",
    "What is the best way to learn a new language?",
    "How to get a job at Google?",
    "If you could have any superpower, what would it be?",
    "I want to learn how to play the piano.",
};*/

// 167-169 added for reachability
static std::string k_system = read_file_from_env_directory("system.txt");
static std::string questions = read_file_from_env_directory("reachability_questions.txt");


struct client {
    ~client() {
        if (smpl) {
            common_sampler_free(smpl);
        }
    }

    int32_t id = 0;
    llama_seq_id seq_id = -1;

    llama_token sampled;

    int64_t t_start_prompt;
    int64_t t_start_gen;

    int32_t n_prompt  = 0;
    int32_t n_decoded = 0;
    int32_t i_batch   = -1;

    std::string input;
    std::string prompt;
    std::string response;
    std::string distance; //195: added for reachability
    
    struct common_sampler * smpl = nullptr;
};

static void print_date_time() {
    std::time_t current_time = std::time(nullptr);
    std::tm* local_time = std::localtime(&current_time);
    char buffer[80];
    strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", local_time);

    LOG_INF("\n");
    LOG_INF("\033[35mrun parameters as of %s\033[0m\n", buffer);
    LOG_INF("\n");
}

// Define a split string function to ...
static std::vector<std::string> split_string(const std::string& input, char delimiter) {
    std::vector<std::string> tokens;
    std::istringstream stream(input);
    std::string token;
    while (std::getline(stream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

int main(int argc, char ** argv) {
    srand(1234);

    common_params params;

    // default value, should be toggled by command line
    params.n_predict = 128;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_PARALLEL)) {
        return 1;
    }

    common_init();

    // number of simultaneous "clients" to simulate
    const int32_t n_clients = params.n_parallel;

    // dedicate one sequence to the system prompt
    params.n_parallel += 1;

    // 242: removed for reachability
    // requests to simulate
    // const int32_t n_seq = params.n_sequences;

    // insert new requests as soon as the previous one is done
    const bool cont_batching = params.cont_batching;
    
    // Old:
    // const bool dump_kv_cache = params.dump_kv_cache;
    // New:
    const bool dump_kv_cache = true;

    // init llama.cpp
    llama_backend_init();
    llama_numa_init(params.numa);

    // load the target model
    common_init_result llama_init = common_init_from_params(params);
    llama_model * model = llama_init.model.get();
    llama_context * ctx = llama_init.context.get();
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // 261-278 commented out for reachability
    // (although it's almost what we want)
    // load the prompts from an external file if there are any
    /*if (params.prompt.empty()) {
        LOG_INF("\033[32mNo new questions so proceed with build-in defaults.\033[0m\n");
    } else {
        // Output each line of the input params.prompts vector and copy to k_prompts
        int index = 0;
        LOG_INF("\033[32mNow printing the external prompt file %s\033[0m\n\n", params.prompt_file.c_str());

        std::vector<std::string> prompts = split_string(params.prompt, '\n');
        for (const auto& prompt : prompts) {
            k_prompts.resize(index + 1);
            k_prompts[index] = prompt;
            index++;
            LOG_INF("%3d prompt: %s\n", index, prompt.c_str());
        }
    }*/

    // 280-282: added & changed for reachability
    std::vector<std::string> q_prompts = split_string(questions, '\n');
    const int32_t n_seq = q_prompts.size();
    const int n_ctx = llama_n_ctx(ctx);

    LOG_INF("\n\n");

    // prepare clients
    std::vector<client> clients(n_clients);
    for (size_t i = 0; i < clients.size(); ++i) {
        auto & client = clients[i];
        client.id = i;
        client.smpl = common_sampler_init(model, params.sampling);
    }

    // tokenize system prompt
    std::vector<llama_token> tokens_system = common_tokenize(ctx, k_system, true);
    const int32_t n_tokens_system = tokens_system.size();

    // the max batch size is as large as the context to handle cases where we get very long input prompt from multiple
    // users. regardless of the size, the main loop will chunk the batch into a maximum of params.n_batch tokens at a time
    llama_batch batch = llama_batch_init(n_ctx, 0, 1);

    int32_t n_total_prompt = 0;
    int32_t n_total_gen    = 0;
    int32_t n_cache_miss   = 0;
    
    const auto t_main_start = ggml_time_us();

    // --------- 1. Evaluate system prompt and save its state ----------
    
    LOG_INF("%s: Evaluating the system prompt ...\n", __func__);
    
    for (int32_t i = 0; i < n_tokens_system; ++i) {
        common_batch_add(batch, tokens_system[i], i, { 0 }, false);
    }
    
    if (llama_decode(ctx, batch) != 0) {
        LOG_ERR("%s: llama_decode() failed\n", __func__);
        return 1;
    }
    
    // New:
    // Save the system-prompt state snapshot (client #0 contains the sys prompt)
    size_t sys_state_size = llama_state_get_size(ctx);
    std::vector<uint8_t> sys_state(sys_state_size);
    llama_state_get_data(ctx, sys_state.data(), sys_state_size);
    LOG_INF("%s: system prompt snapshot saved (%zu bytes)\n", __func__, sys_state_size);
    
    // Old:
    // struct llama_kv_cache_view kvc_view = llama_kv_cache_view_init(ctx, n_clients);
    // New: 
    // Allocate per-client state buffers
    std::vector<std::vector<uint8_t>> client_states(n_clients);
    std::vector<size_t> state_sizes(n_clients, sys_state_size);
    for (int i = 0; i < n_clients; i++) {
        client_states[i].resize(sys_state_size);
        std::copy(sys_state.begin(), sys_state.end(), client_states[i].begin());
    }

    /* Old method, useless now with one way snapshot of cache
    // Added to allow KV cache to be copied
    size_t state_size = llama_state_get_size(ctx);
    
    // assign the system KV cache to all parallel sequences
    for (int32_t i = 1; i <= n_clients; ++i) {
        // Old:
        // llama_kv_self_seq_cp(ctx, 0, i, -1, -1);
        // New:
        state_sizes[i] = state_size;
        client_states[i].resize(state_size);
        llama_state_get_data(ctx, client_states[i].data(), state_sizes[i]);
    }
    */
   
   
   LOG_INF("%s: Simulating parallel requests from clients:\n", __func__);
   LOG_INF("%s: n_parallel = %d, n_sequences = %d, cont_batching = %d, system tokens = %d\n", __func__, n_clients, n_seq, cont_batching, n_tokens_system);
   LOG_INF("\n");
   
   LOG_INF("Processing requests ...\n\n");

   llama_seq_id g_seq_id = 0;

    // 339-340: added for reachability
    std::string output_dir = "output-" + getFileNameWithoutExtension(params.model.path);

    while (true) {
        if (dump_kv_cache) {
            // Old:
            // llama_kv_cache_view_update(ctx, &kvc_view);
            // common_kv_cache_dump_view_seqs(kvc_view, 40);
            // New:
            for (int i = 0; i <= n_clients; ++i) {
                printf("Client %d state size: %zu bytes\n", i, client_state_sizes[i]);
            }
        }

        common_batch_clear(batch);

        // ---- 2. Resume active clients ---
        // decode any currently ongoing sequences
        for (auto & client : clients) {
            if (client.seq_id == -1) {
                continue;
            }

            // restore client state before adding tokens
            llama_state_set_data(ctx, client_states[client.id].data(), state_sizes[client.id]);

            client.i_batch = batch.n_tokens;

            common_batch_add(batch, client.sampled, 
                             n_tokens_system + client.n_prompt + client.n_decoded, 
                             { client.id + 1 }, true);

            client.n_decoded += 1;

            // save state after update
            state_sizes[client.id] = llama_state_get_size(ctx);
            client_states[client.id].resize(state_sizes[client.id]);
            llama_state_get_data(ctx, client_states[client.id].data(), state_sizes[client.id]);
        }

        if (batch.n_tokens == 0) {
            // all sequences have ended - clear the entire KV cache
            for (int i = 1; i <= n_clients; ++i) {
                // Old:
                // llama_kv_self_seq_rm(ctx, i, -1, -1);
                // but keep the system prompt
                // llama_kv_self_seq_cp(ctx, 0, i, -1, -1);
            }
            // New:
            // reset all clients back to system prompt
            for (int i = 0; i < n_clients; ++i) {
                std::copy(sys_state.begin(), sys_state.end(), client_states[i].begin());
                state_sizes[i] = sys_state_size;
            }

            LOG_INF("%s: clearing the KV cache (restored to system prompt)\n", __func__);
        }

        // ---- 3. Add new clients ----
        // insert new sequences for decoding
        if (cont_batching || batch.n_tokens == 0) {
            for (auto & client : clients) {
                if (client.seq_id == -1 && g_seq_id < n_seq) {
                    client.seq_id = g_seq_id;
                    client.t_start_prompt = ggml_time_us();
                    client.t_start_gen    = 0;

                    //383 - 390: changed for reachability
                    //client.input    = k_prompts[rand() % k_prompts.size()];
                    // reading in the prompt and the distance
                    auto input = split_string(q_prompts[g_seq_id], '\t');
                    client.input = input[1];
                    
                    // New:
                    // distances are at positions 0 (+ 2 and 3 if tree shapes calls)
                    std::string distances;
                    for (size_t j = 0; j < input.size(); ++j) {
                        if (j == 1) continue;
                        if (!distances.empty()) distances += ", ";
                        distances += input[j];
                    }
                    client.distance = distances;
                    client.prompt   = client.input + "\nAssistant:";
                    client.response = "";

                    common_sampler_reset(client.smpl);

                    // do not prepend BOS because we have a system prompt!
                    std::vector<llama_token> tokens_prompt = common_tokenize(ctx, client.prompt, false);

                    // restore sys_state for this client
                    llama_state_set_data(ctx, client_states[client.id].data(), state_sizes[client.id]);

                    for (size_t i = 0; i < tokens_prompt.size(); ++i) {
                        common_batch_add(batch, 
                                         tokens_prompt[i], 
                                         i + n_tokens_system, 
                                         { client.id + 1 }, 
                                         false);
                    }

                    // extract the logits only for the last token
                    if (batch.n_tokens > 0) {
                        batch.logits[batch.n_tokens - 1] = true;
                    }

                    client.n_prompt  = tokens_prompt.size();
                    client.n_decoded = 0;
                    client.i_batch   = batch.n_tokens - 1;

                    LOG_INF("\033[31mClient %3d, seq %4d, started decoding ...\033[0m\n", client.id, client.seq_id);

                    g_seq_id += 1;
                }
            }
        }

        if (batch.n_tokens == 0) {
            break;
        }

        // ---- 4. Process batches ----
        // process in chunks of params.n_batch
        int32_t n_batch = params.n_batch;

        for (int32_t i = 0; i < (int32_t) batch.n_tokens; i += n_batch) {
            // experiment: process in powers of 2
            //if (i + n_batch > (int32_t) batch.n_tokens && n_batch > 32) {
            //    n_batch /= 2;
            //    i -= n_batch;
            //    continue;
            //}

            const int32_t n_tokens = std::min(n_batch, (int32_t) (batch.n_tokens - i));

            llama_batch batch_view = {
                n_tokens,
                batch.token    + i,
                nullptr,
                batch.pos      + i,
                batch.n_seq_id + i,
                batch.seq_id   + i,
                batch.logits   + i,
            };

            const int ret = llama_decode(ctx, batch_view);
            if (ret != 0) {
                if (n_batch == 1 || ret < 0) {
                    // if you get here, it means the KV cache is full - try increasing it via the context size
                    LOG_ERR("%s : failed to decode the batch, n_batch = %d, ret = %d\n", __func__, n_batch, ret);
                    return 1;
                }

                LOG_ERR("%s : failed to decode the batch, retrying with n_batch = %d\n", __func__, n_batch / 2);

                n_cache_miss += 1;

                // retry with half the batch size to try to find a free slot in the KV cache
                n_batch /= 2;
                i -= n_batch;

                continue;
            }

            LOG_DBG("%s : decoded batch of %d tokens\n", __func__, n_tokens);

            for (auto & client : clients) {
                if (client.i_batch < (int) i || client.i_batch >= (int) (i + n_tokens)) {
                    continue;
                }

                //printf("client %d, seq %d, token %d, pos %d, batch %d\n",
                //        client.id, client.seq_id, client.sampled, client.n_decoded, client.i_batch);

                const llama_token id = common_sampler_sample(client.smpl, ctx, client.i_batch - i);
                common_sampler_accept(client.smpl, id, true);

                if (client.n_decoded == 1) {
                    // start measuring generation time after the first token to make sure all concurrent clients
                    // have their prompt already processed
                    client.t_start_gen = ggml_time_us();
                }

                const std::string token_str = common_token_to_piece(ctx, id);

                client.response += token_str;
                client.sampled = id;

                //printf("client %d, seq %d, token %d, pos %d, batch %d: %s\n",
                //        client.id, client.seq_id, id, client.n_decoded, client.i_batch, token_str.c_str());

                // 498-505: added and changed for reachability, but should be debugged (threshold too small)
		// commenting out max_tokens: let's figure it out in the generator, given the task, etc
                //int max_tokens = std::max(500,std::abs(client.distance));
                if (client.n_decoded > 2 &&
                        (llama_vocab_is_eog(vocab, id) ||
                         (params.n_predict > 0 && client.n_decoded + client.n_prompt >= params.n_predict) ||
                         //(client.n_decoded + client.n_prompt >= max_tokens) || // added for reachability
                         client.response.find("User:") != std::string::npos ||
                         client.response.find("YES") != std::string::npos ||  // added for reachability
                         client.response.find("NO") != std::string::npos  // added for reachability
                        )) {
                        // commented out to allow multi-line responses
                        //client.response.find('\n') != std::string::npos
                    // basic reverse prompt
                    const size_t pos = client.response.find("User:");
                    if (pos != std::string::npos) {
                        client.response = client.response.substr(0, pos);
                    }

                    // delete only the generated part of the sequence, i.e. keep the system prompt in the cache
                    // Old:
                    // llama_kv_self_seq_rm(ctx,    client.id + 1, -1, -1);
                    // llama_kv_self_seq_cp(ctx, 0, client.id + 1, -1, -1);
                    // New:
                    // reset to sys_state
                    std::copy(sys_state.begin(), sys_state.end(), client_states[client.id].begin());
                    state_sizes[client.id] = sys_state_size;

                    const auto t_main_end = ggml_time_us();

                    LOG_INF("\033[31mClient %3d, seq %3d/%3d, prompt %4d t, response %4d t, time %5.2f s, speed %5.2f t/s, cache miss %d \033[0m \n\nInput:    %s\n\033[35mResponse: %s\033[0m\n\n",
                            client.id, client.seq_id, n_seq, client.n_prompt, client.n_decoded,
                            (t_main_end - client.t_start_prompt) / 1e6,
                            (double) (client.n_prompt + client.n_decoded) / (t_main_end - client.t_start_prompt) * 1e6,
                            n_cache_miss,
                            ::trim(client.input).c_str(),
                            ::trim(client.response).c_str());

                    // 529-530: added for reachability
                    // write_result_to_env_directory(output_dir, "result" + std::to_string(client.seq_id) + "_" + std::to_string(client.distance) + ".txt", ::trim(client.input), ::trim(client.response).c_str());
                    
                    // modified to avoid generating too many files
                    append_result_to_env_directory(output_dir, client.seq_id, client.distance, ::trim(client.input), ::trim(client.response));

                    n_total_prompt += client.n_prompt;
                    n_total_gen    += client.n_decoded;

                    client.seq_id = -1;
                }

                client.i_batch = -1;
            }
        }
    }

    const auto t_main_end = ggml_time_us();

    print_date_time();

    LOG_INF("%s: n_parallel = %d, n_sequences = %d, cont_batching = %d, system tokens = %d\n", __func__, n_clients, n_seq, cont_batching, n_tokens_system);
    if (params.prompt_file.empty()) {
        params.prompt_file = "used built-in defaults";
    }
    LOG_INF("External prompt file: \033[32m%s\033[0m\n", params.prompt_file.c_str());
    LOG_INF("Model and path used:  \033[32m%s\033[0m\n\n", params.model.path.c_str());

    LOG_INF("Total prompt tokens: %6d, speed: %5.2f t/s\n", n_total_prompt, (double) (n_total_prompt              ) / (t_main_end - t_main_start) * 1e6);
    LOG_INF("Total gen tokens:    %6d, speed: %5.2f t/s\n", n_total_gen,    (double) (n_total_gen                 ) / (t_main_end - t_main_start) * 1e6);
    LOG_INF("Total speed (AVG):   %6s  speed: %5.2f t/s\n", "",             (double) (n_total_prompt + n_total_gen) / (t_main_end - t_main_start) * 1e6);
    LOG_INF("Cache misses:        %6d\n", n_cache_miss);

    LOG_INF("\n");

    // TODO: print sampling/grammar timings for all clients
    llama_perf_context_print(ctx);

    llama_batch_free(batch);

    llama_backend_free();

    LOG("\n\n");
    LOG("python analysis.py ../llama.cpp-master/%s", output_dir.c_str());

    return 0;
}
