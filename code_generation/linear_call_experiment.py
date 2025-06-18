from experiment import Experiment
import generate_chain as gen

class LinearCallExperiment(Experiment):
    def __init__(
        self,
        name,
        n_methods=50,
        n_pad=0,
        range_of_depth=range(1, 8),
        **kwargs  # autres params passés à Experiment
    ):
        super().__init__(name, **kwargs)
        self.n_methods = n_methods
        self.range_of_depth = range_of_depth
        self.n_pad = n_pad
        self.calls_per_function = 1  # par définition
        
        self._setup_experiment_dir()

    def _metadata_dict(self):
        return {
            "type": "linear",
            "n_methods": self.n_methods,
            "n_pad": self.n_pad,
            "range_of_depth": list(self.range_of_depth),
            "calls_per_function": self.calls_per_function,
            "n_questions_per_distance": self.n_questions_per_distance,
            "n_comments": self.n_comments,
            "n_loops": self.n_loops,
            "n_if": self.n_if,
        }

    def generate(self):
        print(f"Generating experiment in directory {self.experiment_path}")
        print(f"=> Linear experiment with {self.n_methods} methods...")
        
        gen.generate_all(self.name, self.n_methods, self.range_of_depth, self.n_questions_per_distance, self.n_pad, self.n_comments)
