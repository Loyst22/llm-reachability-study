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

# Function to generate Java-style Lorem Ipsum comments
def generate_lorem_ipsum_comments(num_lines):
    comments = []
    
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
        lorem_comment = f"// {' '.join(sentence).capitalize()}."
        comments.append(lorem_comment)
    
    return "\n".join(comments)

# Generate and print 10 lines of Lorem Ipsum Java comments
#print(generate_lorem_ipsum_comments(10))

def generate_chained_method_calls(method_names, lines=20):
    method_bodies = []

    # Loop through the list of method names
    for i, method in enumerate(method_names):
        # Check if this is the last method in the list
        comment = generate_lorem_ipsum_comments(lines)
        if i < len(method_names) - 1:
            # Call the next method in the list
            next_method = method_names[i + 1]
            method_body = f"public void {method}() {{\n    {next_method}();\n}}"
        else:
            # Last method, no call to the next method
            method_body = f"public void {method}() {{\n    // End of chain\n}}"
        
        # Append to the list of method bodies
        method_bodies.append(f"{comment}\n{method_body}")
    
    return method_bodies