import random
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Callable
from abc import ABC, abstractmethod
from math import ceil

import prompts
import control_flow
import comments_generation
import generate_tree_chains as gen_tree
from experiment_config import ExperimentConfig, LinearCallExperimentConfig, TreeCallExperimentConfig

class MethodNameGenerator:
    """Base class for generating method/function names"""
    
    PREFIXES = [
        "get", "set", "is", "calculate", "process", "fetch", "update", "create", "delete", 
        "find", "check", "load", "save", "reset", "clear", "validate", "initialize", 
        "convert", "apply", "enable", "disable", "sort", "merge", "copy", "generate", 
        "retrieve", "parse", "extract", "compare", "build", "register", "unregister",
        "sync", "execute", "dispatch", "resolve", "filter", "log"
    ]
    
    VERBS = [
        "Data", "Item", "Value", "State", "Config", "Status", "Object", "Parameter", "Setting", 
        "Resource", "Detail", "Info", "Message", "Handler", "Element", "Connection", "Index", 
        "Entry", "Key", "Session", "Metric", "Field", "Action", "Notification", "Instance", 
        "Node", "Task", "Job", "Event", "Request", "Response", "Flag", "File", "Directory", 
        "Path", "Buffer", "User", "Account", "Transaction", "Cache", "Result", "List", 
        "Map", "Queue", "Stack", "Collection", "Component", "Service", "Manager"
    ]
    
    NOUNS = [
        "ById", "ForUser", "WithFilter", "InCache", "FromDatabase", "FromFile", "ToJson", 
        "FromXml", "IfAvailable", "OrDefault", "AsString", "FromUrl", "OnClick", "InMemory", 
        "FromApi", "ForSession", "WithTimeout", "ForRequest", "FromResponse", "AtIndex", 
        "WithKey", "WithIndex", "ForTransaction", "IfValid", "OnInit", "AsList", "ForRole", 
        "ToBuffer", "ForMapping", "OnComplete", "AtPosition", "ToSet", "AsMap", "AsQueue", 
        "WithLimit", "ToCollection", "ForEach", "IfEnabled", "WithPolicy", "InThread", 
        "ForExecution", "InParallel", "AsObservable", "IfExists", "WithRetries"
    ]

    @classmethod
    def generate_method_name(cls, style: str = "camelCase") -> str:
        """Generate a single random method name in specified style"""
        prefix = random.choice(cls.PREFIXES)
        verb = random.choice(cls.VERBS)
        noun = random.choice(cls.NOUNS)
        
        if style == "camelCase":
            return f"{prefix}{verb}{noun}"
        elif style == "snake_case":
            return f"{prefix}_{verb}_{noun}".lower()
        elif style == "PascalCase":
            return f"{prefix.capitalize()}{verb}{noun}"
        else:
            return f"{prefix}{verb}{noun}"

    @classmethod
    def generate_unique_method_names(cls, n: int, style: str = "camelCase") -> List[str]:
        """Generate n unique method names"""
        unique_names = set()
        while len(unique_names) < n:
            unique_names.add(cls.generate_method_name(style))
        return list(unique_names)


class LanguageGenerator(ABC):
    """Abstract base class for language-specific code generators"""
    
    @abstractmethod
    def generate_chained_method_calls(self, method_names: List[str]) -> List[str]:
        """Generate chained method calls for the specific language"""
        pass
    
    @abstractmethod
    def generate_class_with_multiple_chains(self, class_name: str, chains: List[List[str]], 
                                          chain_generator: Callable) -> str:
        """Generate a class/module with multiple method chains"""
        pass
    
    @abstractmethod
    def generate_class_from_multiple_trees(self, trees: list, method_names: list, selection: list,
                                           config: TreeCallExperimentConfig, class_name: str ="TheClass") -> str:
        """Generate a class/module based on the trees generated"""
    
    @abstractmethod
    def get_file_extension(self) -> str:
        """Get the file extension for this language"""
        pass
    
    @abstractmethod
    def get_method_name_style(self) -> str:
        """Get the method naming style for this language"""
        pass
    
    @staticmethod
    def chain_generator(method_names: list, config: ExperimentConfig) -> list:
        """Generates a list of method bodies with respect to the config"""
        # Just an alias for the moment since control works only for java
        if config.language.lower() != "java":
            return comments_generation.generate_chained_method_calls(method_names, config.n_comment_lines)
        
        method_bodies = []
        
        # Loop through the list of method names
        for i, method in enumerate(method_names):
            # Generate comments for this method if necessary
            comment = comments_generation.generate_lorem_ipsum_comments(config.n_comment_lines, config.language)

            if i < len(method_names) - 1:
                # Generate content of the method body with control flow if necessary
                method_body = control_flow.generate_method(caller_method=method,
                                                            called_methods=[method_names[i+1]],
                                                            n_vars=config.n_vars,
                                                            n_loops=config.n_loops,
                                                            n_if=config.n_if)
            else:
                method_body = control_flow.generate_method(caller_method=method,
                                                           called_methods=None,
                                                           n_vars=config.n_vars,
                                                           n_loops=config.n_loops,
                                                           n_if=config.n_if)
            comment = "\t" + comment.replace("\n", "\n\t")
            method_body = "\t" + method_body.replace("\n", "\n\t")
            
            method_bodies.append(f"{comment}\n{method_body}")
            
        return method_bodies
            
class JavaGenerator(LanguageGenerator):
    """Java-specific code generator"""
    
    def generate_chained_method_calls(self, method_names: List[str]) -> List[str]:
        """Generate chained method calls for Java"""
        method_bodies = []
        for i, method in enumerate(method_names):
            if i < len(method_names) - 1:
                next_method = method_names[i + 1]
                method_body = f"    public void {method}() {{\n        {next_method}();\n    }}"
            else:
                method_body = f"    public void {method}() {{\n        // End of chain\n    }}"
            method_bodies.append(method_body)
        return method_bodies

    def generate_class_with_multiple_chains(self, class_name: str, chains: List[List[str]], chain_generator: Callable) -> str:
        """Generate a Java class with multiple method chains"""
        method_bodies = []
        for chain in chains:
            method_bodies.extend(chain_generator(chain))
        
        random.shuffle(method_bodies)
        
        class_body = f"public class {class_name} {{\n"
        class_body += "\n\n".join(method_bodies)
        class_body += "\n}"
        
        return class_body
    
    def generate_class_from_multiple_trees(self, trees: list, config: TreeCallExperimentConfig, class_name: str ="TheClass") -> str:
        """Generate a class with methods that call each other in a tree-like structure.

        Args:
            directory (str): The directory where the generated files will be saved.
            tree_depth (int): The depth of the tree to be generated.
        """
        
        # method_bodies = gen_tree.generate_tree_method_calls(trees)
        method_bodies = gen_tree.generate_tree_method_calls(trees=trees, config=config)
    
        print(f"Generated {len(method_bodies)} method bodies")

        # Shuffle the method bodies to create random order in the class
        random.shuffle(method_bodies)

        # Construct the class with shuffled method bodies
        class_body = f"public class {class_name} {{\n"
        class_body += "\n\n".join(method_bodies)
        class_body += "\n}"
        
        return class_body
    
    def get_file_extension(self) -> str:
        return ".java"
    
    def get_method_name_style(self) -> str:
        return "camelCase"


class CppGenerator(LanguageGenerator):
    """C++ specific code generator"""
    
    def generate_chained_method_calls(self, method_names: List[str]) -> List[str]:
        """Generate chained method calls for C++"""
        method_bodies = []
        for i, method in enumerate(method_names):
            if i < len(method_names) - 1:
                next_method = method_names[i + 1]
                method_body = f"\tvoid {method}() {{\n\t\t{next_method}();\n    }}"
            else:
                method_body = f"\tvoid {method}() {{\n\t// End of chain\n    }}"
            method_bodies.append(method_body)
        return method_bodies

    def generate_class_with_multiple_chains(self, class_name: str, chains: List[List[str]], chain_generator: Callable) -> str:
        """Generate a C++ class with multiple method chains"""
        method_bodies = []
        for chain in chains:
            method_bodies.extend(chain_generator(chain))
        
        random.shuffle(method_bodies)
        
        # Generate header section
        class_body = f"#include <iostream>\n\nclass {class_name} {{\npublic:\n"
        class_body += "\n\n".join(method_bodies)
        class_body += "\n};"
        
        return class_body
    
    def get_file_extension(self) -> str:
        return ".cpp"
    
    def get_method_name_style(self) -> str:
        return "camelCase"


class FortranGenerator(LanguageGenerator):
    """Fortran specific code generator"""
    
    def generate_chained_method_calls(self, method_names: List[str]) -> List[str]:
        """Generate chained subroutine calls for Fortran"""
        method_bodies = []
        for i, method in enumerate(method_names):
            if i < len(method_names) - 1:
                next_method = method_names[i + 1]
                method_body = f"    subroutine {method}()\n        call {next_method}()\n    end subroutine {method}"
            else:
                method_body = f"    subroutine {method}()\n        ! End of chain\n    end subroutine {method}"
            method_bodies.append(method_body)
        return method_bodies

    def generate_class_with_multiple_chains(self, class_name: str, chains: List[List[str]], 
                                          chain_generator: Callable) -> str:
        """Generate a Fortran module with multiple subroutine chains"""
        method_bodies = []
        for chain in chains:
            method_bodies.extend(chain_generator(chain))
        
        random.shuffle(method_bodies)
        
        # Generate Fortran module
        class_body = f"module {class_name.lower()}\n    implicit none\n\ncontains\n\n"
        class_body += "\n\n".join(method_bodies)
        class_body += f"\n\nend module {class_name.lower()}"
        
        return class_body
    
    def get_file_extension(self) -> str:
        return ".f90"
    
    def get_method_name_style(self) -> str:
        return "snake_case"


class PascalGenerator(LanguageGenerator):
    """Pascal specific code generator"""
    
    def generate_chained_method_calls(self, method_names: List[str]) -> List[str]:
        """Generate chained procedure calls for Pascal"""
        method_bodies = []
        for i, method in enumerate(method_names):
            if i < len(method_names) - 1:
                next_method = method_names[i + 1]
                method_body = f"    procedure {method};\n    begin\n        {next_method};\n    end;"
            else:
                method_body = f"    procedure {method};\n    begin\n        // End of chain\n    end;"
            method_bodies.append(method_body)
        return method_bodies

    def generate_class_with_multiple_chains(self, class_name: str, chains: List[List[str]], 
                                          chain_generator: Callable) -> str:
        """Generate a Pascal unit with multiple procedure chains"""
        method_bodies = []
        for chain in chains:
            method_bodies.extend(chain_generator(chain))
        
        random.shuffle(method_bodies)
        
        # Generate Pascal unit
        class_body = f"unit {class_name};\n\ninterface\n\ntype\n    T{class_name} = class\n    public\n"
        
        # Add procedure declarations
        for body in method_bodies:
            # this is wrong when we have some comments!
            # // comments are ok in pascal, so are { } and (* *) comments
            proc_name = body.split()[1].rstrip(';')
            class_body += f"        procedure {proc_name};\n"
        
        class_body += "    end;\n\nimplementation\n\n"
        class_body += "\n\n".join(method_bodies)
        class_body += "\n\nend."
        
        return class_body
    
    def get_file_extension(self) -> str:
        return ".pas"
    
    def get_method_name_style(self) -> str:
        return "PascalCase"


class RubyGenerator(LanguageGenerator):
    """Ruby specific code generator"""
    
    def generate_chained_method_calls(self, method_names: List[str]) -> List[str]:
        """Generate chained method calls for Ruby"""
        method_bodies = []
        for i, method in enumerate(method_names):
            if i < len(method_names) - 1:
                next_method = method_names[i + 1]
                method_body = f"  def {method}\n    {next_method}\n  end"
            else:
                method_body = f"  def {method}\n    # End of chain\n  end"
            method_bodies.append(method_body)
        return method_bodies

    def generate_class_with_multiple_chains(self, class_name: str, chains: List[List[str]], 
                                          chain_generator: Callable) -> str:
        """Generate a Ruby class with multiple method chains"""
        method_bodies = []
        for chain in chains:
            method_bodies.extend(chain_generator(chain))
        
        random.shuffle(method_bodies)
        
        # Generate Ruby class
        class_body = f"class {class_name}\n"
        class_body += "\n\n".join(method_bodies)
        class_body += "\nend"
        
        return class_body
    
    def get_file_extension(self) -> str:
        return ".rb"
    
    def get_method_name_style(self) -> str:
        return "snake_case"


class PhpGenerator(LanguageGenerator):
    """PHP specific code generator"""
    
    def generate_chained_method_calls(self, method_names: List[str]) -> List[str]:
        """Generate chained method calls for PHP"""
        method_bodies = []
        for i, method in enumerate(method_names):
            if i < len(method_names) - 1:
                next_method = method_names[i + 1]
                method_body = f"    public function {method}() {{\n        $this->{next_method}();\n    }}"
            else:
                method_body = f"    public function {method}() {{\n        // End of chain\n    }}"
            method_bodies.append(method_body)
        return method_bodies

    def generate_class_with_multiple_chains(self, class_name: str, chains: List[List[str]], 
                                          chain_generator: Callable) -> str:
        """Generate a PHP class with multiple method chains"""
        method_bodies = []
        for chain in chains:
            method_bodies.extend(chain_generator(chain))
        
        random.shuffle(method_bodies)
        
        # Generate PHP class
        class_body = f"<?php\n\nclass {class_name} {{\n"
        class_body += "\n\n".join(method_bodies)
        class_body += "\n}\n?>"
        
        return class_body
    
    def get_file_extension(self) -> str:
        return ".php"
    
    def get_method_name_style(self) -> str:
        return "camelCase"


class QuestionGenerator:
    """Generates reachability questions for method chains"""
    
    @staticmethod
    def generate_call_questions_with_distances_and_chains(method_names: List[str], 
                                                        language: str = "java") -> List[Tuple]:
        """Generate questions, distances, and chains for all pairs of methods"""
        questions_with_distances_and_chains = []
        num_methods = len(method_names)
        
        # Language-specific terminology
        if language in ["fortran", "f90"]:
            call_term = "call"
            method_term = "subroutine"
        elif language in ["pascal", "pas"]:
            call_term = "call"
            method_term = "procedure"
        elif language in ["cpp", "c++"]:
            call_term = "call"
            method_term = "function"
        elif language in ["ruby", "rb", "php", "java"]:
            call_term = "call"
            method_term = "method"
        else:  # others
            call_term = "call"
            method_term = "method"

        for i in range(num_methods):
            for j in range(num_methods):
                if i != j:
                    question = (
                        f"Does `{method_names[i]}` {call_term} `{method_names[j]}`, either directly or indirectly? "
                        f"Think step-by-step by following the {method_term} calls from `{method_names[i]}.`"
                    )

                    if i < j:
                        chain = method_names[i:j + 1]
                        distance = len(chain) - 1
                    else:
                        chain = method_names[i:]
                        distance = -(len(chain) - 1)
                    
                    start_back_chain = max(0, i - len(chain))
                    back_chain = method_names[start_back_chain:i]
                    
                    questions_with_distances_and_chains.append((question, distance, chain, back_chain))
        
        return questions_with_distances_and_chains

    @staticmethod
    def select_questions_by_distance(questions_with_distances: List[Tuple], distance: int, n: int) -> List[Tuple]:
        """Select up to n questions with a specified distance"""
        filtered = [q for q in questions_with_distances if q[1] == distance]
        return random.sample(filtered, min(n, len(filtered)))


class FileWriter:
    """Handles all file writing operations"""
    
    @staticmethod
    def write_class_to_file(body: str, filename: Path) -> None:
        """Write class body to file"""
        with open(filename, 'w') as f:
            f.write(body)

    @staticmethod
    def write_prompt_to_file(prompt: dict, body: str, filename: Path) -> None:
        """Write prompt with body to file"""
        with open(filename, 'w') as f:
            f.write(prompt["start"])
            f.write(body)
            f.write(prompt["end"])

    @staticmethod
    def write_questions_to_file(questions_with_distances: List[Tuple], filename: Path) -> None:
        """Write questions to file, one per line"""
        with open(filename, 'w') as f:
            for question, dist, *rest in questions_with_distances:
                f.write(f"{dist}\t{question}\n")

    @staticmethod
    def write_chains_to_file(questions: List[Tuple], filename: Path, config: ExperimentConfig) -> None:
        """Write all chains from the questions list to a file"""
        if config.type == "linear":
            with open(filename, 'w') as f:
                for question, dist, chain, back_chain in questions:
                    f.write(" ".join(chain) + '\t' + " ".join(back_chain) + '\n')
        elif config.type == "tree":
            with open(filename, 'w') as file:
                for question, dist, chain in questions:
                    file.write(" ".join(chain) + '\n')
        else :
            raise ValueError(f"Unknow experiment type: {config.type}")

    @staticmethod
    def write_methods_to_file(methods: List[str], filename: Path) -> None:
        """Write methods to file"""
        with open(filename, 'w') as f:
            f.write(" ".join(methods))

    @staticmethod
    def write_slurm_script(name: str, path: str, time: str) -> None:
        """Write SLURM job script"""
        contents = f"""#! /usr/bin/bash
#SBATCH --job-name=reachability-{name}
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --constraint=h100
#SBATCH --ntasks-per-node=1
#SBATCH --account=spk@h100
#SBATCH --time={time}
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=romain.robbes@labri.fr

module load arch/h100
module load cuda/12.8.0
cd $WORK
python run_experiment.py {path}"""
        
        with open(f'reachability-{name}.slurm', 'w') as f:
            f.write(contents)


class LanguageFactory:
    """Factory class for creating language-specific generators"""
    
    _generators = {
        "java": JavaGenerator,
        "cpp": CppGenerator,
        "c++": CppGenerator,
        "fortran": FortranGenerator,
        "f90": FortranGenerator,
        "pascal": PascalGenerator,
        "pas": PascalGenerator,
        "ruby": RubyGenerator,
        "rb": RubyGenerator,
        "php": PhpGenerator
    }
    
    @classmethod
    def get_generator(cls, language: str) -> LanguageGenerator:
        """Get a language generator instance"""
        language = language.lower()
        if language not in cls._generators:
            raise ValueError(f"Unsupported language: {language}. Supported languages: {list(cls._generators.keys())}")
        return cls._generators[language]()
    
    @classmethod
    def get_supported_languages(cls) -> List[str]:
        """Get list of supported languages"""
        return list(cls._generators.keys())


class ExperimentRunner:
    """Main class for running experiments"""
    
    def __init__(self):
        self.method_generator = MethodNameGenerator()
        self.question_generator = QuestionGenerator()
        self.file_writer = FileWriter()

    @staticmethod
    def divide_list_into_chunks(lst: List, chunk_size: int) -> List[List]:
        """Divide a list into chunks of specified size"""
        return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

    @staticmethod
    def flatten_list(lst: List[List]) -> List:
        """Flatten a list of lists"""
        return [item for sublist in lst for item in sublist]

    @staticmethod
    def count_distances(tuples_list: List[Tuple]) -> dict:
        """Count occurrences of each distance"""
        count_dict = defaultdict(int)
        for item in tuples_list:
            count_dict[item[1]] += 1
        return count_dict

    def generate_single_linear_context(self, directory: Path, n_chains: int, chain_size: int, n_questions: int, 
                               chain_generator: Callable, config: LinearCallExperimentConfig) -> None:
        """Generate a single context with specified parameters"""
        # print(f"Generating {config.language} class with {config.context_size} methods")
        # Get language-specific generator
        lang_generator = LanguageFactory.get_generator(config.language)
        
        # Generate method names with appropriate naming style
        method_names = self.method_generator.generate_unique_method_names(
            config.context_size, 
            lang_generator.get_method_name_style()
        )
        
        # Subdivide the list of method names into chains of methods
        all_chains = self.divide_list_into_chunks(method_names, chain_size)
        
        
        # Generate the class/module
        the_class = lang_generator.generate_class_with_multiple_chains(
            "TheClass",
            all_chains,
            chain_generator
        )
                
        # Generate questions for all chains
        all_questions = []
        for chain_names in all_chains:
            questions = self.question_generator.generate_call_questions_with_distances_and_chains(
                chain_names,
                config.language
            )
            all_questions.extend(questions)
                    
        # Select questions for each depth
        selection = []
        for depth in config.depths:
            selection.extend(self.question_generator.select_questions_by_distance(all_questions, depth, n_questions))
            selection.extend(self.question_generator.select_questions_by_distance(all_questions, -depth, n_questions))
        
        print(f"Chains:\n\tExpected total: {n_chains}\n\tGround truth: {len(all_chains)}")
        print(f"Questions:\n\tExpected total: {2 * n_questions * len(config.depths)}\n\tGround truth: {len(selection)}")
        print(f"Distance distribution: {self.count_distances(selection)}")

        # Write all files
        directory.mkdir(parents=True, exist_ok=True)
        
        # Use language-specific file extension
        class_filename = f"TheClass{lang_generator.get_file_extension()}"
        self.file_writer.write_class_to_file(the_class, directory / class_filename)
        self.file_writer.write_prompt_to_file(prompts.in_context, the_class, directory / "system.txt")
        self.file_writer.write_questions_to_file(selection, directory / "reachability_questions.txt")
        self.file_writer.write_chains_to_file(selection, directory / "chains.txt", config)
        self.file_writer.write_methods_to_file(method_names, directory / "methods.txt")
        
    def generate_single_tree_context(self, directory:str, n_trees:int, tree_depth:int, config:TreeCallExperimentConfig, max_chain_length:int = None, n_questions:int = 100) -> int:
        """Generate an experiment with multiple trees and save the class to a file.

        Args:
            directory (str): The name of the experiment.
            n_trees (int): The number of trees to generate.
            tree_depth (int): The depth of each tree.
        """        
        lang_generator = LanguageFactory.get_generator(config.language)
        
        print(f"Generating {n_trees} trees with depth {tree_depth} for experiment {directory}")
        trees, method_names = gen_tree.generate_many_call_trees(directory, tree_depth, n_trees)
        print(f"Generated {len(trees)} trees")
        _, valid_questions = gen_tree.find_all_valid_chains(trees=trees)
        _, invalid_questions = gen_tree.find_all_invalid_chains(trees=trees)
        
        selection = []
       
        # The maximum length of a chain is deduced from the tree depth
        if max_chain_length is None:
            max_chain_length = (2**(tree_depth + 1) - 1)
        
        for depth in range(max_chain_length + 1):
            # TODO : choose a better number of questions to select (e.g. 100 is kind of arbitrary) 
            selection.extend(QuestionGenerator.select_questions_by_distance(valid_questions, depth, n_questions))
            # TODO : see if we can manage to get negative questions for all distances
            selection.extend(QuestionGenerator.select_questions_by_distance(invalid_questions, -depth, n_questions))
        
        distance_dict = self.count_distances(selection)
    
        min_amount_of_questions = n_questions

        for value in distance_dict.values():
            if value < min_amount_of_questions:
                min_amount_of_questions = value

        selection = []
        
        for depth in range(max_chain_length + 1):
            selection.extend(QuestionGenerator.select_questions_by_distance(valid_questions, depth, min_amount_of_questions))
            selection.extend(QuestionGenerator.select_questions_by_distance(invalid_questions, -depth, min_amount_of_questions))
    
        print(f"Selected {len(selection)} questions")
        print(f"Questions per distance after selection: {self.count_distances(selection)}")

        the_class = lang_generator.generate_class_from_multiple_trees(trees=trees, config=config)

        # Use language-specific file extension
        class_filename = f"TheClass{lang_generator.get_file_extension()}"
        self.file_writer.write_class_to_file(the_class, directory / class_filename)
        self.file_writer.write_prompt_to_file(prompts.in_context_tree_calls, the_class, directory / "system.txt")
        self.file_writer.write_questions_to_file(selection, directory / "reachability_questions.txt")
        self.file_writer.write_chains_to_file(selection, directory / "chains.txt", config)
        self.file_writer.write_methods_to_file(method_names, directory / "methods.txt")
        
        return min_amount_of_questions

    def generate_experiment(self, config: ExperimentConfig) -> List[Path]:
        """Generate an experiment based on configuration (and its type)"""
        print(f"Starting experiment with config {config}")
        if config.type == "linear":
            return self.generate_linear_experiment(config)
        elif config.type == "tree":
            return self.generate_tree_experiment(config)
        else: 
            raise ValueError(f"Unknow experiment type: {config.type}")
        
    
    def generate_linear_experiment(self, config: LinearCallExperimentConfig):
        """Generate a complete linear-chain experiment based on configuration
        This method generates a series of directories, each containing:
        - A Java class with chained method calls
        - A system prompt for the llm
        - A file with reachability questions
        - A file with the names of the methods used in the experiment
        - A file with the chains of methods used for the reachability questions

        Args:
            config (LinearCallExperimentConfig): Configuration for the linear call experiment

        Returns:
            list: List of directories where the experiments were generated
        """
        base_dir = Path(config.name)
        directories = []
        
        # It is here necessary to add at least 2 to the size of the chain because:
        # - For a valid chain of n methods, the largest depth/distance is n-1
        # - For an invalid chain of n methods, the largest depth/distance is -(n-2)
        # If the padding is 0 or 1, we need at least 2 additional methods to take that into account
        chain_size = max(config.depths) + min(config.n_padding, 2)
        
        # In case the context size chosen is too small, we take an appropriate size wrt the chain size
        # This is to ensure that the number of chains that fit into the context is at least 1
        if chain_size > config.context_size:
            config.context_size = chain_size + 2
        
        # We need n questions per depth, each chain has a size of chain_size
        n_methods_needed = chain_size * config.n_questions
        
        print()
        print(f"Methods needed to generate enough questions: {n_methods_needed}")
        
        # We also define the number of chains that fit wrt the context size
        n_chains_in_context = config.context_size // chain_size
        
        # And we create new contexts until we have enough questions
        n_questions_left = config.n_questions
        
        while n_questions_left > 0:
            # Name the experiment sub-directory:
            depth_str = self._format_depths(config.depths)
            q_start = (config.n_questions - n_questions_left) * len(config.depths) * 2
            q_end = (config.n_questions - n_questions_left + n_chains_in_context) * len(config.depths) * 2
            
            exp_dir = base_dir / f"ctx_{config.context_size}_depths_{depth_str}_com_{config.n_comment_lines}_var_{config.n_vars}_loop_{config.n_loops}_if_{config.n_if}_qs_{q_start}--{q_end}_{config.language}"
            
            # One chain account for one set of questions at most 
            # as the larger depth question often require a full chain
            n_qs = min(n_chains_in_context, n_questions_left)
            
            print(f"\nGenerating {config.language} context of size {config.context_size} for {2*n_qs*len(config.depths)} questions")
            
            # Use language-specific chain generator
            # lang_generator = LanguageFactory.get_generator(config.language)
            # chain_generator = lambda c: comments_generation.generate_chained_method_calls(c, config.n_comment_lines)
            
            # first claude fix, disregarded and overlooking all the architectural stuff :-)
            # chain_generator = lang_generator.generate_chained_method_calls
            
            # chain_generator = lambda c: comments_generation.generate_chained_method_calls_with_comments(c, config.n_comment_lines, config.language)
            chain_generator = lambda c: LanguageGenerator.chain_generator(method_names=c, config=config)
            
            self.generate_single_linear_context(exp_dir, n_chains_in_context, chain_size, n_qs, chain_generator, config)
            
            directories.append(exp_dir)
            n_questions_left -= n_chains_in_context
            
            print(f"Output directory: {exp_dir}")
        
        return directories
    
    def generate_tree_experiment(self, config: TreeCallExperimentConfig):
        """Generate a complete tree-chain experiment based on configuration
        This method generates a series of directories, each containing:
        - A Java class with chained method calls
        - A system prompt for the llm
        - A file with reachability questions
        - A file with the names if the methods used in the experiment
        - A file with the chains of methods used for the reachability questions
        - A directory with files describing the trees used for the experiment

        Args:
            config (TreeCallExperimentConfig): Configuration for the tree call experiment
        
        Returns:
            list: List of directories where the experiments were generated
        """
        base_dir = Path(config.name)
        directories = []
        
        chain_size = max(config.depths) + min(config.n_padding, 2)
        
        # In case the context size chosen is too small, we take an appropriate size wrt the chain size
        # This is to ensure that the number of chains that fit into the context is at least 1
        if chain_size > config.context_size:
            config.context_size = chain_size + 2
        
        # We need to define the number of trees and their depths
        # To do so we use the chain size and find the appropriate tree depth
        tree_depth = 0
        methods_per_tree = 2**(tree_depth + 1) - 1
        while methods_per_tree - 1 < chain_size:
            tree_depth += 1
            methods_per_tree = 2**(tree_depth + 1) - 1
            
        # Now that we know the depth of the trees we must determine the number of trees to generate
        # The number of trees must correspond to the context size of the config
        methods_per_tree = 2**(tree_depth + 1) - 1
        n_trees = ceil(config.context_size/methods_per_tree)
        
        # We create new contexts until we have enough questions
        n_questions_left = config.n_questions
        
        context_counter = 1
        
        while n_questions_left > 0:
            depth_str = self._format_depths(config.depths)
            exp_dir = base_dir / f"ctx_{config.context_size}_depths_{depth_str}_com_{config.n_comment_lines}_var_{config.n_vars}_loop_{config.n_loops}_if_{config.n_if}_qs_{context_counter}_{config.language}"
            
            
            # n_questions_generated = gen_tree.generate_exp(exp_dir, n_trees, tree_depth, max(config.depths), n_questions_left)
            n_questions_generated = self.generate_single_tree_context(exp_dir, n_trees, tree_depth, config, max(config.depths), n_questions_left)
            n_questions_left -= n_questions_generated
            
            print(f"\nGenerating {config.language} context of size {n_trees*methods_per_tree} for {2*n_questions_generated*len(config.depths)} questions")

            context_counter += 1
            
            print(f"Output directory: {exp_dir}")
        
        return directories

    @staticmethod
    def _format_depths(depths: List[int]) -> str:
        """Format depth list for directory naming"""
        if len(depths) > 4:
            return f"{depths[0]}--{depths[-1]}"
        return "_".join(str(d) for d in depths)

    def generate_batch_experiments(self, context_ranges: List[int], n_comments: int,
                                   n_vars: int, n_loops:int, n_if: int, language: str = "java",
                                   experiment_type: str = "linear") -> None:
        """Generate multiple experiments for different context sizes"""
        for context_size in context_ranges:
            config = ExperimentConfig(
                name=f'experiments/{language}/{experiment_type}/context_{context_size}_comments_{n_comments}_vars_{n_vars}_loops_{n_loops}_if_{n_if}',
                context_size=context_size,
                depths=list(range(1, 11)),
                n_questions= 5, # was 200
                n_padding=0,
                n_comment_lines=n_comments,
                n_vars=n_vars,
                n_loops=n_loops,
                n_if=n_if,
                language=language,
                type=experiment_type
            )
            
            self.generate_experiment(config)
            
            # Commented for debugging purposes
            """
            self.file_writer.write_slurm_script(
                f'{language}_context_{context_size}_comments_{n_comments}',
                config.name,
                config.time_limit
            )
            """

    def generate_all_experiments(self, languages: List[str] = ["java"]) -> None:
        """Generate all predefined experiments for specified languages"""
        experiment_configs = [
            # ([50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000], 0),
            # ([50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000], 2),
            # ([50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500], 4),
            ([50, 75, 100, 150, 200, 250, 300, 350], 7),
            # ([50, 75, 100, 150, 200], 12),
            # ([100], 24)
        ]
        
        experiment_configs = [
            # ([depths], comments, vars, loops, if)
            ([50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000], 0, 0, 0, 0),
            ([50, 75, 100, 150, 200, 250, 300, 350, 400], 0, 1, 1, 1),
            ([50, 75, 100, 150, 200, 250], 0, 2, 2, 2),
            
            ([50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000], 2, 0, 0, 0),
            ([50, 75, 100, 150, 200, 250, 300, 350, 400], 2, 1, 1, 1),
            ([50, 75, 100, 150, 200, 250], 2, 2, 2, 2),
            
            ([50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500], 4, 0, 0, 0),
            ([50, 75, 100, 150, 200, 250, 300, 350], 4, 1, 1, 1),
            ([50, 75, 100, 150, 200], 4, 2, 2, 2),
            
            ([50, 75, 100, 150, 200, 250, 300, 350, 400], 7, 0, 0, 0),
            ([50, 75, 100, 150, 200, 250, 300], 7, 1, 1, 1),
            ([50, 75, 100, 150, 200], 7, 2, 2, 2),
            
            ([50, 75, 100, 150, 200], 12, 0, 0, 0),
            ([50, 75, 100, 150, 200], 12, 1, 1, 1),
            ([50, 75, 100, 150, 200], 12, 2, 2, 2),
            
            ([50, 75, 100], 24, 0, 0, 0),
            ([50, 75, 100], 24, 1, 1, 1),
            ([50, 75, 100], 24, 2, 2, 2),
        ]
        
        for type in ["linear", "tree"]:
            print(f"\n=== Generating {type} experiments ===")
            for language in languages:
                print(f"\n=== Generating experiments for {language.upper()} ===")
                for context_ranges, n_comments, n_vars, n_loops, n_if in experiment_configs:
                    self.generate_batch_experiments(context_ranges=context_ranges,
                                                    n_comments=n_comments,
                                                    n_vars=n_vars,
                                                    n_loops=n_loops,
                                                    n_if=n_if,
                                                    language=language,
                                                    experiment_type=type)


# Backward compatibility - keep the original JavaMethodGenerator for existing code
JavaMethodGenerator = MethodNameGenerator
JavaClassGenerator = LanguageFactory.get_generator("java")


# Usage examples
if __name__ == "__main__":
    runner = ExperimentRunner()
    
    # Generate for all supported languages
    supported_languages = LanguageFactory.get_supported_languages()
    print(f"Supported languages: {supported_languages}")
    
    # Generate experiments for Java, C++, Fortran, Pascal, Ruby, and PHP
    # runner.generate_all_experiments(["java", "cpp", "fortran", "pascal", "ruby", "php"])
    # runner.generate_all_experiments(["cpp", "fortran"])

    # Or generate for a single language
    runner.generate_all_experiments(["java"])
    
    # Example of generating a single experiment with custom config
    
    # custom_config = ExperimentConfig(
    #     name="experiment/custom_java_experiment",
    #     context_size=100,
    #     depths=[1, 2, 3],
    #     n_questions=50,
    #     n_padding=2,
    #     n_comment_lines=2,
    #     n_vars=2,
    #     n_loops=2,
    #     n_if=2,
    #     language="java",
    #     type="linear"
    # )
    
    # runner.generate_experiment(custom_config)
    
    custom_config = ExperimentConfig(
        name="experiments/test",
        context_size=5,
        depths=[1, 2, 3, 4, 5, 6, 7, 8],
        n_questions=1,
        n_padding=0,
        n_comment_lines=1,
        n_vars=1,
        n_loops=1,
        n_if=1,
        language="Java",
        type="linear"
    )
    
    # runner.generate_experiment(custom_config)

