import random

# Basic Latin-inspired words and syllables
lorem_words = [
    "lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing",
    "elit", "sed", "do", "eiusmod", "tempor", "incididunt", "ut", "labore",
    "et", "dolore", "magna", "aliqua", "ut", "enim", "ad", "minim", "veniam",
    "quis", "nostrud", "exercitation", "ullamco", "laboris", "nisi", "aliquip",
    "consequat", "duis", "aute", "irure", "voluptate", "velit", "esse", "cillum",
    "eu", "fugiat", "nulla", "pariatur", "excepteur", "sint", "occaecat", "cupidatat",
    "non", "proident", "culpa", "qui", "officia", "deserunt", "mollit", "anim",
    "id", "est", "laborum", "perspiciatis", "unde", "omnis", "iste", "natus", "error",
    "santium", "doloremque", "laudantium", "totam", "rem", "aperiam", "eaque", "ipsa",
    "quae", "ab", "illo", "inventore", "veritatis", "et", "quasi", "architecto", "beatae",
    "vitae", "dicta", "sunt", "explicabo", "aspernatur", "aut", "odit", "aut", "fugit",
    "sed", "quia", "consequuntur", "magnam", "dolores", "eos", "qui", "ratione", "voluptatem",
    "nesciunt", "neque", "porro", "quisquam", "est", "qui", "dolorem", "ipsum", "quia",
    "dolor", "sit", "amet", "consectetur", "adipisci", "velit", "sed", "quia", "non",
    "numquam", "eius", "modi", "tempora", "incidunt", "ut", "labore", "et", "dolore",
    "magnam", "aliquam", "quaerat", "voluptatem", "ut", "enim", "minima", "veniam"
]

def generate_lorem_ipsum_comments(num_lines, language="java"):
    """Generate Lorem Ipsum comments in the specified language's comment style"""
    comments = []
    
    # Define comment styles for different languages
    comment_styles = {
        "java": "//",
        "cpp": "//",
        "c++": "//",
        "fortran": "!",
        "f90": "!",
        "pascal": "//",
        "pas": "//",
        "ruby": "#",
        "rb": "#",
        "php": "//"
    }
    
    comment_prefix = comment_styles.get(language.lower(), "//")
    
    for _ in range(num_lines):
        # Randomly decide on the sentence length for each line
        sentence_length = random.randint(6, 12)
        sentence = []
        
        # Build sentence with natural-looking word structure
        while len(sentence) < sentence_length:
            word_count = random.randint(1, 3)  # Small clusters of 1-3 words
            phrase = " ".join(random.choice(lorem_words) for _ in range(word_count))
            sentence.append(phrase)
        
        # Join words/phrases and add it as a comment line
        lorem_comment = f"{comment_prefix} {' '.join(sentence).capitalize()}."
        comments.append(lorem_comment)
    
    return "\n".join(comments)

def generate_chained_method_calls_with_comments(method_names, lines=20, language="java"):
    """Generate chained method calls with comments for the specified language"""
    method_bodies = []
    
    # Loop through the list of method names
    for i, method in enumerate(method_names):
        # Generate comments for this method
        comment = generate_lorem_ipsum_comments(lines, language)
        
        # Generate the method body based on language
        if language.lower() in ["java"]:
            if i < len(method_names) - 1:
                next_method = method_names[i + 1]
                method_body = f"    public void {method}() {{\n        {next_method}();\n    }}"
            else:
                method_body = f"    public void {method}() {{\n        // End of chain\n    }}"
        
        elif language.lower() in ["cpp", "c++"]:
            if i < len(method_names) - 1:
                next_method = method_names[i + 1]
                method_body = f"    void {method}() {{\n        {next_method}();\n    }}"
            else:
                method_body = f"    void {method}() {{\n        // End of chain\n    }}"
        
        elif language.lower() in ["fortran", "f90"]:
            if i < len(method_names) - 1:
                next_method = method_names[i + 1]
                method_body = f"    subroutine {method}()\n        call {next_method}()\n    end subroutine {method}"
            else:
                method_body = f"    subroutine {method}()\n        ! End of chain\n    end subroutine {method}"
        
        elif language.lower() in ["pascal", "pas"]:
            if i < len(method_names) - 1:
                next_method = method_names[i + 1]
                method_body = f"    procedure {method};\n    begin\n        {next_method};\n    end;"
            else:
                method_body = f"    procedure {method};\n    begin\n        // End of chain\n    end;"
        
        elif language.lower() in ["ruby", "rb"]:
            if i < len(method_names) - 1:
                next_method = method_names[i + 1]
                method_body = f"  def {method}\n    {next_method}\n  end"
            else:
                method_body = f"  def {method}\n    # End of chain\n  end"
        
        elif language.lower() == "php":
            if i < len(method_names) - 1:
                next_method = method_names[i + 1]
                method_body = f"    public function {method}() {{\n        $this->{next_method}();\n    }}"
            else:
                method_body = f"    public function {method}() {{\n        // End of chain\n    }}"
        
        else:
            # Default to Java-style for unknown languages
            if i < len(method_names) - 1:
                next_method = method_names[i + 1]
                method_body = f"    public void {method}() {{\n        {next_method}();\n    }}"
            else:
                method_body = f"    public void {method}() {{\n        // End of chain\n    }}"
        
        # Combine comment and method body
        method_bodies.append(f"{comment}\n{method_body}")
    
    return method_bodies

# Backward compatibility - keep the original function name
def generate_chained_method_calls(method_names, lines=20):
    """Original function for backward compatibility - defaults to Java"""
    return generate_chained_method_calls_with_comments(method_names, lines, "java")

# Example usage
if __name__ == "__main__":
    methods = ["processData", "validateInput", "formatOutput"]
    
    print("=== Java Example ===")
    java_methods = generate_chained_method_calls_with_comments(methods, 3, "java")
    for method in java_methods:
        print(method)
        print()
    
    print("=== Fortran Example ===")
    fortran_methods = generate_chained_method_calls_with_comments(methods, 3, "fortran")
    for method in fortran_methods:
        print(method)
        print()
    
    print("=== Ruby Example ===")
    ruby_methods = generate_chained_method_calls_with_comments(methods, 3, "ruby")
    for method in ruby_methods:
        print(method)
        print()