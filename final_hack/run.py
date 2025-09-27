import os

def print_tree(start_path, prefix=""):
    files = os.listdir(start_path)
    for i, name in enumerate(files):
        path = os.path.join(start_path, name)
        connector = "└── " if i == len(files)-1 else "├── "
        print(prefix + connector + name)
        if os.path.isdir(path):
            extension = "    " if i == len(files)-1 else "│   "
            print_tree(path, prefix + extension)

print_tree(".")  # current folder
