

from generator_8lang import ExperimentRunner


if __name__ == "__main__":
    runner = ExperimentRunner()
    
    # For this experience we only generate context without comments and with level 1 control flow4
    # Without params so that we can compare the results for linear experiments
    experiment_configs = [
        # ([context], comments, vars, loops, if, params)
        # ? should we add 50 (not included for linear experiments)
        ([50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000], 0, 1, 1, 1, 0)
    ]
    
    type = "tree"
    langage = "java"
    print(f"\n=== Generating {type} experiments ===")
    print(f"\n=== Generating experiments for {langage} ===")
    for context_ranges, n_comments, n_vars, n_loops, n_if, n_params in experiment_configs:
        runner.generate_batch_experiments(context_ranges=context_ranges, 
                                          n_comments=n_comments,
                                          n_vars=n_vars,
                                          n_loops=n_loops,
                                          n_if=n_if, 
                                          n_params=n_params, 
                                          language=langage,
                                          experiment_type=type)