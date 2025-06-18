import tkinter as tk
from tkinter import ttk, messagebox
from linear_call_experiment import LinearCallExperiment
from tree_call_experiment import TreeCallExperiment

class ExperimentGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Experiment Launcher")

        self.type_var = tk.StringVar(value="linear")
        self.entries = {}
        self.field_widgets = {}

        self.fields = {
            "common": [
                ("Name", "name"),
                ("Experiment folder", "exp_dir"),
                ("Questions/distance", "n_questions_per_distance"),
                ("Comments", "n_comments"),
                ("Loops", "n_loops"),
                ("Conditions", "n_if"),
            ],
            "linear": [
                ("Methods", "n_methods"),
            ],
            "tree": [
                ("Depth", "depth"),
                ("Trees", "n_tree"),
                ("Branching", "branching_factor"),
            ]
        }

        self.build_interface()

    def build_interface(self):
        frame = ttk.Frame(self.root, padding=10)
        frame.grid(row=0, column=0, sticky="nsew")

        # Radio buttons for experiment type
        ttk.Label(frame, text="Type of experiment:").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(frame, text="Linear", variable=self.type_var, value="linear", command=self.update_fields).grid(row=1, column=0, sticky="w")
        ttk.Radiobutton(frame, text="Tree", variable=self.type_var, value="tree", command=self.update_fields).grid(row=2, column=0, sticky="w")

        self.form_frame = ttk.Frame(frame, padding=5)
        self.form_frame.grid(row=3, column=0, columnspan=2, sticky="nsew")
        self.build_fields()

        ttk.Button(frame, text="Create Experiment", command=self.create_experiment).grid(row=99, column=0, columnspan=2, pady=10)

    def build_fields(self):
        """Create all fields but hide the ones not needed"""
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
        # Show common fields
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
            kind = self.type_var.get()
            kwargs = {
                "name": self.entries["name"].get(),
                "exp_dir": self.entries["exp_dir"].get(),
                "n_questions_per_distance": int(self.entries["n_questions_per_distance"].get() or 10),
                "n_comments": int(self.entries["n_comments"].get() or 0),
                "n_loops": int(self.entries["n_loops"].get() or 0),
                "n_if": int(self.entries["n_if"].get() or 0)
            }

            if kind == "linear":
                n_methods = int(self.entries["n_methods"].get() or 50)
                exp = LinearCallExperiment(**kwargs, n_methods=n_methods)
            else:
                depth = int(self.entries["depth"].get() or 3)
                n_tree = int(self.entries["n_tree"].get() or 3)
                branching = int(self.entries["branching_factor"].get() or 2)
                exp = TreeCallExperiment(**kwargs, depth=depth, n_tree=n_tree, branching_factor=branching)

            exp.generate()
            messagebox.showinfo("Success", f"Experiment created at:\n{exp.experiment_path}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred:\n{e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ExperimentGUI(root)
    root.mainloop()
