import random
import string
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Callable, Optional, Dict
from abc import ABC, abstractmethod
import prompts as p
# import control_flow_3
import comments_generation as comments_generation


@dataclass
class ExperimentConfig:
    """Configuration for generating experiments"""
    name: str
    context_size: int
    depths: List[int]
    n_questions: int
    n_padding: int = 0
    n_comment_lines: int = 0
    time_limit: str = "1:00:00"
    language: str = "java"  # Added language parameter


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
    def get_file_extension(self) -> str:
        """Get the file extension for this language"""
        pass
    
    @abstractmethod
    def get_method_name_style(self) -> str:
        """Get the method naming style for this language"""
        pass


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

    def generate_class_with_multiple_chains(self, class_name: str, chains: List[List[str]], 
                                          chain_generator: Callable) -> str:
        """Generate a Java class with multiple method chains"""
        method_bodies = []
        for chain in chains:
            method_bodies.extend(chain_generator(chain))
        
        random.shuffle(method_bodies)
        
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
                method_body = f"    void {method}() {{\n        {next_method}();\n    }}"
            else:
                method_body = f"    void {method}() {{\n        // End of chain\n    }}"
            method_bodies.append(method_body)
        return method_bodies

    def generate_class_with_multiple_chains(self, class_name: str, chains: List[List[str]], 
                                          chain_generator: Callable) -> str:
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
        elif language in ["ruby", "rb"]:
            call_term = "call"
            method_term = "method"
        elif language == "php":
            call_term = "call"
            method_term = "method"
        else:  # java and others
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
    def write_chains_to_file(questions: List[Tuple], filename: Path) -> None:
        """Write all chains from the questions list to a file"""
        with open(filename, 'w') as f:
            for question, dist, chain, back_chain in questions:
                f.write(" ".join(chain) + '\t' + " ".join(back_chain) + '\n')

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

    def generate_single_context(self, directory: Path, context_size: int, n_chains: int, 
                               chain_size: int, depths: List[int], n_questions: int, 
                               chain_generator: Callable, language: str = "java") -> None:
        """Generate a single context with specified parameters"""
        print(f"Generating {language} class with {context_size} methods, {n_chains} chains")
        print(f"Questions per depth: {n_questions}, expected total: {n_questions * len(depths)}")

        # Get language-specific generator
        lang_generator = LanguageFactory.get_generator(language)
        
        # Generate method names with appropriate naming style
        method_names = self.method_generator.generate_unique_method_names(
            context_size, lang_generator.get_method_name_style()
        )
        all_chains = self.divide_list_into_chunks(method_names, chain_size)
        
        print(f"Actual number of chains: {len(all_chains)}")
        
        # Generate the class/module
        the_class = lang_generator.generate_class_with_multiple_chains(
            "TheClass", all_chains, chain_generator
        )
        
        # Generate questions for all chains
        all_questions = []
        for chain_names in all_chains:
            questions = self.question_generator.generate_call_questions_with_distances_and_chains(
                chain_names, language
            )
            all_questions.extend(questions)
        
        # Select questions for each depth
        selection = []
        for depth in depths:
            selection.extend(self.question_generator.select_questions_by_distance(all_questions, depth, n_questions))
            selection.extend(self.question_generator.select_questions_by_distance(all_questions, -depth, n_questions))
        
        print(f"Actual number of questions: {len(selection)}")
        print(f"Distance distribution: {self.count_distances(selection)}")

        # Write all files
        directory.mkdir(parents=True, exist_ok=True)
        
        # Use language-specific file extension
        class_filename = f"TheClass{lang_generator.get_file_extension()}"
        self.file_writer.write_class_to_file(the_class, directory / class_filename)
        self.file_writer.write_prompt_to_file(p.in_context, the_class, directory / "system.txt")
        self.file_writer.write_questions_to_file(selection, directory / "reachability_questions.txt")
        self.file_writer.write_chains_to_file(selection, directory / "chains.txt")
        self.file_writer.write_methods_to_file(method_names, directory / "methods.txt")

    def generate_experiment(self, config: ExperimentConfig) -> List[Path]:
        """Generate a complete experiment based on configuration"""
        chain_size = max(config.depths) + config.n_padding
        n_methods_needed = chain_size * config.n_questions
        
        print(f"Methods needed: {n_methods_needed}")
        print(f"Language: {config.language}")
        
        n_chains_in_context = config.context_size // chain_size
        n_questions_left = config.n_questions
        
        base_dir = Path(config.name)
        directories = []
        
        while n_questions_left > 0:
            depth_str = self._format_depths(config.depths)
            q_start = (config.n_questions - n_questions_left) * len(config.depths) * 2
            q_end = (config.n_questions - n_questions_left + n_chains_in_context) * len(config.depths) * 2
            
            exp_dir = base_dir / f"ctx_{config.context_size}_depths_{depth_str}_com_{config.n_comment_lines}_qs_{q_start}--{q_end}_{config.language}"
            
            n_qs = min(n_chains_in_context, n_questions_left)
            print(f"Generating context size {config.context_size} for {n_qs * len(config.depths)} questions")
            print(f"Output directory: {exp_dir}")
            
            # Use language-specific chain generator
            lang_generator = LanguageFactory.get_generator(config.language)
            #chain_generator = lambda c: comments_generation.generate_chained_method_calls(c, config.n_comment_lines)
            
            # first claude fix, disregarded and overlooking all the architectural stuff :-)
            # chain_generator = lang_generator.generate_chained_method_calls
            chain_generator = lambda c: comments_generation.generate_chained_method_calls_with_comments(c, config.n_comment_lines, config.language)

            
            self.generate_single_context(
                exp_dir, config.context_size, n_chains_in_context, 
                chain_size, config.depths, n_qs, chain_generator, config.language
            )
            
            directories.append(exp_dir)
            n_questions_left -= n_chains_in_context
        
        return directories

    @staticmethod
    def _format_depths(depths: List[int]) -> str:
        """Format depth list for directory naming"""
        if len(depths) > 8:
            return f"{depths[0]}--{depths[-1]}"
        return "_".join(str(d) for d in depths)

    def generate_batch_experiments(self, context_ranges: List[int], n_comments: int, 
                                 language: str = "java") -> None:
        """Generate multiple experiments for different context sizes"""
        for context_size in context_ranges:
            config = ExperimentConfig(
                name=f'xps/{language}/context_{context_size}_comments_{n_comments}',
                context_size=context_size,
                depths=list(range(1, 11)),
                n_questions=200,
                n_padding=0,
                n_comment_lines=n_comments,
                language=language
            )
            
            self.generate_experiment(config)
            self.file_writer.write_slurm_script(
                f'{language}_context_{context_size}_comments_{n_comments}',
                config.name,
                config.time_limit
            )

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
        
        for language in languages:
            print(f"\n=== Generating experiments for {language.upper()} ===")
            for context_ranges, n_comments in experiment_configs:
                self.generate_batch_experiments(context_ranges, n_comments, language)


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
    runner.generate_all_experiments(["java", "cpp", "fortran", "pascal", "ruby", "php"])
    #runner.generate_all_experiments(["cpp", "fortran"])

    # Or generate for a single language
    # runner.generate_all_experiments(["java"])
    
    # Example of generating a single experiment with custom config
    custom_config = ExperimentConfig(
        name="custom_cpp_experiment",
        context_size=100,
        depths=[1, 2, 3],
        n_questions=50,
        n_comment_lines=2,
        language="cpp"
    )
    # runner.generate_experiment(custom_config)

