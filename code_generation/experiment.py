import os
import json
from datetime import datetime
from abc import ABC, abstractmethod

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
