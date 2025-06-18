from experiment import Experiment
import generate_tree_chains as gen

class TreeCallExperiment(Experiment):
    def __init__(
        self,
        name,
        depth=3,
        n_tree=3,
        branching_factor=2,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self.depth = depth
        self.n_tree = n_tree
        self.branching_factor = branching_factor
        self.n_methods = self._calculate_n_methods()
        self.calls_per_function = branching_factor
        
        self._setup_experiment_dir()

    def _calculate_n_methods(self):
        # TODO
        pass

    def _metadata_dict(self):
        return {
            "type": "tree",
            "depth": self.depth,
            "n_tree": self.n_tree,
            "branching_factor": self.branching_factor,
            "n_methods": self.n_methods,
            "calls_per_function": self.calls_per_function,
            "n_questions_per_distance": self.n_questions_per_distance,
            "n_comments": self.n_comments,
            "n_loops": self.n_loops,
            "n_if": self.n_if,
        }

    def generate(self):
        print(f"Generating experiment in directory {self.experiment_path}")
        print(f"=> Tree experiment with depth {self.depth}, {self.n_tree} tree and branching factor {self.branching_factor}...")
        
        # Ton code de génération ici
        gen.generate_exp(exp_name=self.name, n_trees=self.n_tree, tree_depth=self.depth, n_questions=self.n_questions_per_distance)
        
