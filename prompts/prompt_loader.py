def load_prompt_from_file(filename="prompt.txt"):
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()
