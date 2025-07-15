import os
import tkinter as tk
from tkinter import ttk, messagebox

from generator_8lang import ExperimentRunner, ExperimentConfig, LinearCallExperimentConfig, TreeCallExperimentConfig


class ExperimentGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Experiment Launcher")

        self.runner = ExperimentRunner()

        self.type_var = tk.StringVar(value="linear")
        self.entries = {}
        self.field_widgets = {}

        self.fields = {
            "common": [
                ("Name", "name"),
                ("Context size", "context_size"),
                ("Depths (comma-separated)", "depths"),
                ("Questions/distance", "n_questions"),
                ("Padding", "n_padding"),
                ("Comments", "n_comment_lines"),
                ("Loops", "n_loops"),
                ("Conditions", "n_if"),
                ("Variables", "n_vars"),
                ("Parameters", "n_params"),
                ("Language", "language"),
            ],
            "linear": [],
            "tree": []
            #     ("Tree Depth", "tree_depth"),
            #     ("Number of Trees", "n_tree"),
            #     ("Calls per Function", "calls_per_function"),
            # ]
        }

        self.build_interface()

    def build_interface(self):
        frame = ttk.Frame(self.root, padding=10)
        frame.grid(row=0, column=0, sticky="nsew")

        ttk.Label(frame, text="Type of experiment:").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(frame, text="Linear", variable=self.type_var, value="linear", command=self.update_fields).grid(row=1, column=0, sticky="w")
        ttk.Radiobutton(frame, text="Tree", variable=self.type_var, value="tree", command=self.update_fields).grid(row=2, column=0, sticky="w")

        self.form_frame = ttk.Frame(frame, padding=5)
        self.form_frame.grid(row=3, column=0, columnspan=2, sticky="nsew")
        self.build_fields()

        ttk.Button(frame, text="Create Experiment", command=self.create_experiment).grid(row=99, column=0, columnspan=2, pady=10)

    def build_fields(self):
        row = 0
        for section in ["common", "linear", "tree"]:
            for label_text, key in self.fields[section]:
                label = ttk.Label(self.form_frame, text=label_text + ":")
                entry = ttk.Entry(self.form_frame)
                self.field_widgets[key] = (label, entry)
                self.entries[key] = entry
                label.grid(row=row, column=0, sticky="w", pady=2)
                entry.grid(row=row, column=1, pady=2)
                row += 1
        self.update_fields()

    def update_fields(self):
        kind = self.type_var.get()
        for section in ["common", "linear", "tree"]:
            for _, key in self.fields[section]:
                label, entry = self.field_widgets[key]
                if section == "common" or section == kind:
                    label.grid()
                    entry.grid()
                else:
                    label.grid_remove()
                    entry.grid_remove()

    def create_experiment(self):
        try:
            kind = self.type_var.get() or ExperimentConfig.DEFAULT_EXP_TYPE
            # Get shared fields
            entry_name = self.entries["name"].get().strip() or ExperimentConfig.DEFAULT_DIR_NAME
            name = os.path.join("experiments", entry_name)
            context_size = int(self.entries["context_size"].get() or ExperimentConfig.DEFAULT_CTX_SIZE)
            depths_raw = self.entries["depths"].get()
            depths = list(map(int, depths_raw.split(","))) if depths_raw else ExperimentConfig.DEFAULT_DEPTHS
            n_questions = int(self.entries["n_questions"].get() or ExperimentConfig.DEFAULT_N_QUESTIONS)
            n_padding = int(self.entries["n_padding"].get() or ExperimentConfig.DEFAULT_N_PADDING)
            n_comment_lines = int(self.entries["n_comment_lines"].get() or ExperimentConfig.DEFAULT_N_COMMENTS)
            n_loops = int(self.entries["n_loops"].get() or ExperimentConfig.DEFAULT_N_LOOPS)
            n_if = int(self.entries["n_if"].get() or ExperimentConfig.DEFAULT_N_IF)
            n_vars = int(self.entries["n_vars"].get() or ExperimentConfig.DEFAULT_N_VARS)
            n_params = int(self.entries["n_params"].get() or ExperimentConfig.DEFAULT_N_PARAMS)
            language = self.entries["language"].get() or ExperimentConfig.DEFAULT_LANGUAGE
            
            if n_if != 0 and n_loops != 0 and n_vars == 0:
                n_vars = 1

            # Add type-specific config
            if kind == "linear":
                config = LinearCallExperimentConfig(
                    name=name,
                    context_size=context_size,
                    depths=depths,
                    n_questions=n_questions,
                    n_padding=n_padding,
                    n_comment_lines=n_comment_lines,
                    n_loops=n_loops,
                    n_if=n_if,
                    n_vars=n_vars,
                    n_params=n_params,
                    language=language,
                    type="linear"
                )
            elif kind == "tree":
                # tree_depth = int(self.entries["tree_depth"].get() or 3)
                # n_tree = int(self.entries["n_tree"].get() or 3)
                # calls_per_function = int(self.entries["calls_per_function"].get() or 2)

                config = TreeCallExperimentConfig(
                    name=name,
                    context_size=context_size,
                    depths=depths,
                    n_questions=n_questions,
                    n_padding=n_padding,
                    n_comment_lines=n_comment_lines,
                    n_loops=n_loops,
                    n_if=n_if,
                    n_vars=n_vars,
                    n_params=n_params,
                    language=language,
                    type="tree"
                    # tree_depth=tree_depth,
                    # n_tree=n_tree,
                    # calls_per_function=calls_per_function
                )
            else:
                raise ValueError("Unsupported experiment type")

            self.runner.generate_experiment(config)

            if self.entries["name"].get():
                messagebox.showinfo("Success", f"Experiment created in:\n{config.name}")
            else:
                messagebox.showinfo("Success", f"Directory name not specified.\nExperiment created in:\n{config.name}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"An error occurred:\n{e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ExperimentGUI(root)
    
    # Centering the window
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (int(height / 1.5))
    root.geometry(f'+{x}+{y}')

    # Starting main loop
    root.mainloop()
