import os
import argparse

def generate_tree(directory, prefix="", ignore_dirs=None, ignore_files=None, max_depth=None, current_depth=0):
    if ignore_dirs is None:
        ignore_dirs = {'.git', '__pycache__', 'node_modules', '.vscode', '.idea', '.ipynb_checkpoints'}
    if ignore_files is None:
        ignore_files = {'.DS_Store', '.gitignore', '*.pyc', '*.pyo'}
    
    if max_depth is not None and current_depth >= max_depth:
        return ""
    
    tree_str = ""
    try:
        items = sorted(os.listdir(directory))
    except PermissionError:
        return ""
    
    # Filter items
    dirs = [item for item in items if os.path.isdir(os.path.join(directory, item)) and item not in ignore_dirs]
    files = [item for item in items if os.path.isfile(os.path.join(directory, item)) and 
             not any(item.endswith(ext.replace('*', '')) for ext in ignore_files if not ext.startswith('*')) and
             item not in ignore_files]
    
    all_items = dirs + files
    
    for i, item in enumerate(all_items):
        is_last = i == len(all_items) - 1
        is_dir = item in dirs
        
        tree_str += f"{prefix}{'└── ' if is_last else '├── '}{item}{'/' if is_dir else ''}\n"
        
        if is_dir:
            extension = "    " if is_last else "│   "
            tree_str += generate_tree(
                os.path.join(directory, item),
                prefix + extension,
                ignore_dirs,
                ignore_files,
                max_depth,
                current_depth + 1
            )
    
    return tree_str

def main():
    parser = argparse.ArgumentParser(description='Generate file structure markdown')
    parser.add_argument('--path', default='.', help='Path to analyze (default: current directory)')
    parser.add_argument('--output', default='FILE_STRUCTURE.md', help='Output markdown file')
    parser.add_argument('--max-depth', type=int, default=4, help='Maximum depth to traverse')
    
    args = parser.parse_args()
    
    project_name = os.path.basename(os.path.abspath(args.path))
    
    with open(args.output, 'w') as f:
        f.write(f"# {project_name} - File Structure\n\n")
        f.write("```\n")
        f.write(f"{project_name}/\n")
        f.write(generate_tree(args.path, max_depth=args.max_depth))
        f.write("```\n")
        f.write("\n## Description\n\n")
        f.write("This file structure was automatically generated.\n")
    
    print(f"File structure generated in {args.output}")

if __name__ == "__main__":
    main()