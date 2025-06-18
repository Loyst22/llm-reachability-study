import os
import json
from datetime import datetime
from abc import ABC, abstractmethod
import generate_chain as gen

class Experiment(ABC):
    """ Modelise a generic (abstract) experiment """
    def __init__(
        self,
        name,
        n_questions_per_distance=10,
        n_comments=0,
        n_loops=0,
        n_if=0
    ):
        self.name = name
        self.experiment_path = os.path.join("experiments", self.name)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.n_questions_per_distance = n_questions_per_distance
        self.n_comments = n_comments
        self.n_loops = n_loops
        self.n_if = n_if


    def _setup_experiment_dir(self):
        """ Create the directory in which the files are generated and save metadata about the experiment """
        os.makedirs(self.experiment_path, exist_ok=True)
        self._save_metadata()

    def _save_metadata(self):
        metadata = self._metadata_dict()
        metadata["name"] = self.name
        metadata["timestamp"] = self.timestamp
        metadata_path = os.path.join(self.experiment_path, "metadata.json")
        with open(file=metadata_path, mode="w") as f:
            json.dump(metadata, f, indent=4)

    @abstractmethod
    def _metadata_dict(self):
        """À surcharger pour ajouter des champs spécifiques à chaque type d’expérience"""
        return {}

    @abstractmethod
    def generate(self):
        """À surcharger pour générer les programmes java"""
        pass


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
        
