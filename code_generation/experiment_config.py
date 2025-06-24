from dataclasses import dataclass
from typing import List


@dataclass
class ExperimentConfig:
    """Configuration for generating experiments"""
    name: str
    context_size: int
    depths: List[int]
    n_questions: int
    n_padding: int = 0
    n_comment_lines: int = 0
    n_vars: int = 0
    n_loops: int = 0
    n_if: int = 0
    time_limit: str = "1:00:00"
    language: str = "java"  # Added language parameter
    type: str = "linear"
    
    # Default values for experiments
    DEFAULT_DIR_NAME = "default_test"
    DEFAULT_EXP_TYPE = "linear"
    DEFAULT_CTX_SIZE = 50
    DEFAULT_DEPTHS = [1, 2, 3, 4, 5, 6, 7, 8]
    DEFAULT_N_QUESTIONS = 3
    DEFAULT_N_PADDING = 2
    DEFAULT_N_COMMENTS = 2
    DEFAULT_N_LOOPS = 0
    DEFAULT_N_IF = 0
    DEFAULT_N_VARS = 0
    DEFAULT_LANGUAGE = "java"
    
    def __str__(self) -> str:
        return (
            f"\n{'-'*46}\n"
            f"Name:                 {self.name}\n"
            f"Context Size:         {self.context_size}\n"
            f"Depths:               {self.depths}\n"
            f"Questions per depth:  {self.n_questions}\n"
            f"Padding:              {self.n_padding}\n"
            f"Comment Lines:        {self.n_comment_lines}\n"
            f"Variables:            {self.n_vars}\n"
            f"Loops:                {self.n_loops}\n"
            f"If Statements:        {self.n_if}\n"
            f"Time Limit:           {self.time_limit}\n"
            f"Language:             {self.language}\n"
            f"Type:                 {self.type}\n"
            f"{'-'*46}"
        )

@dataclass
class LinearCallExperimentConfig(ExperimentConfig):
    pass
@dataclass
class TreeCallExperimentConfig(ExperimentConfig):
    tree_depth: int = 3
    n_tree: int = 3
    calls_per_function: int = 2
    type: str = "tree"
    
    @property
    def n_method(self) -> int:
        """Calcul automatique du nombre de méthodes d'un arbre"""
        return self.n_tree * (self.calls_per_function**(self.tree_depth+1) - 1) // (self.calls_per_function - 1)
