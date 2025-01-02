import nbformat
import os
import uuid
import subprocess

def ensure_ids(notebook):
    """Ensure every cell in the notebook has a unique id."""
    for cell in notebook.cells:
        if "id" not in cell:
            cell["id"] = str(uuid.uuid4())
    return notebook

def create_notebook():
    """Creates a new Jupyter Notebook file by asking the user for a filename and opens it in VS Code."""
    filename = input("Enter the name for the new notebook (without .ipynb extension): ").strip()
    if not filename:
        print("Filename cannot be empty, file not created")
        return
    
    if not filename.endswith(".ipynb"):
        filename += ".ipynb"

    current_dir = os.path.dirname(__file__)
    directory = os.path.join(current_dir, "..", "code_sharing")

    if not os.path.exists(directory):
        os.makedirs(directory)

    filepath = os.path.join(directory, filename)

    notebook = nbformat.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "name": "python3",
            "display_name": "Python 3",
            "language": "python"
        },
        "language_info": {
            "name": "python",
            "version": "3.10"
        }
    }

    notebook = ensure_ids(notebook)

    try:
        with open(filepath, "w") as f:
            nbformat.write(notebook, f)
        print(f"Created '{filepath}'")
        
        subprocess.run(["code", filepath], check=True)
    except Exception as e:
        print(f"Error creating or opening notebook: {e}")

    return filepath

def set_title(filepath):
    """Adds a markdown cell with a h1 title to the specified notebook."""
    if not os.path.exists(filepath):
        print("The specified notebook file does not exist")
        return

    try:
        with open(filepath, "r") as f:
            notebook = nbformat.read(f, as_version=4)
    except Exception as e:
        print(f"Error reading notebook: {e}")
        return

    title = input("Enter the title for the notebook: ").strip()
    if not title:
        print("No title inputted, skipped insertion")
        return

    markdown_cell = nbformat.v4.new_markdown_cell(f"# {title}")
    notebook.cells.insert(0, markdown_cell)

    notebook = ensure_ids(notebook)

    try:
        with open(filepath, "w") as f:
            nbformat.write(notebook, f)
        print(f"Set title for '{filepath}': '{title}'")
    except Exception as e:
        print(f"Error updating notebook: {e}")

def add_about(filepath):
    """Appends the contents of 'about-validmind.ipynb' to the specified notebook if the user agrees."""
    source_notebook_path = os.path.join(os.path.dirname(__file__), "about-validmind.ipynb")

    if not os.path.exists(source_notebook_path):
        print(f"Source notebook '{source_notebook_path}' does not exist")
        return

    user_input = input("Do you want to include information about ValidMind? (yes/no): ").strip().lower()
    if user_input not in ("yes", "y"):
        print("Skipping appending 'about-validmind.ipynb'")
        return

    try:
        with open(filepath, "r") as target_file:
            target_notebook = nbformat.read(target_file, as_version=4)

        with open(source_notebook_path, "r") as source_file:
            source_notebook = nbformat.read(source_file, as_version=4)
    except Exception as e:
        print(f"Error reading notebooks: {e}")
        return
    
    for cell in source_notebook.cells:
        cell["id"] = f"about-{uuid.uuid4()}"

    target_notebook.cells.extend(source_notebook.cells)

    target_notebook = ensure_ids(target_notebook)

    try:
        with open(filepath, "w") as target_file:
            nbformat.write(target_notebook, target_file)
        print(f"'about-validmind.ipynb' appended to '{filepath}'")
    except Exception as e:
        print(f"Error appending notebooks: {e}")

def add_install(filepath):
    """Appends the contents of 'install-initialize-validmind.ipynb' to the specified notebook if the user agrees."""
    source_notebook_path = os.path.join(os.path.dirname(__file__), "install-initialize-validmind.ipynb")

    if not os.path.exists(source_notebook_path):
        print(f"Source notebook '{source_notebook_path}' does not exist")
        return

    user_input = input("Do you want to include installation and initialization instructions? (yes/no): ").strip().lower()
    if user_input not in ("yes", "y"):
        print("Skipping appending 'install-initialize-validmind.ipynb'")
        return

    try:
        with open(filepath, "r") as target_file:
            target_notebook = nbformat.read(target_file, as_version=4)

        with open(source_notebook_path, "r") as source_file:
            source_notebook = nbformat.read(source_file, as_version=4)
    except Exception as e:
        print(f"Error reading notebooks: {e}")
        return

    target_notebook.cells.extend(source_notebook.cells)

    target_notebook = ensure_ids(target_notebook)

    try:
        with open(filepath, "w") as target_file:
            nbformat.write(target_notebook, target_file)
        print(f"'install-initialize-validmind.ipynb' appended to '{filepath}'")
    except Exception as e:
        print(f"Error appending notebooks: {e}")

def next_steps(filepath):
    """Appends the contents of 'next-steps.ipynb' to the specified notebook if the user agrees."""
    source_notebook_path = os.path.join(os.path.dirname(__file__), "next-steps.ipynb")

    if not os.path.exists(source_notebook_path):
        print(f"Source notebook '{source_notebook_path}' does not exist")
        return

    user_input = input("Do you want to include next steps? (yes/no): ").strip().lower()
    if user_input not in ("yes", "y"):
        print("Skipping appending 'next-steps.ipynb'")
        return

    try:
        with open(filepath, "r") as target_file:
            target_notebook = nbformat.read(target_file, as_version=4)

        with open(source_notebook_path, "r") as source_file:
            source_notebook = nbformat.read(source_file, as_version=4)
    except Exception as e:
        print(f"Error reading notebooks: {e}")
        return

    target_notebook.cells.extend(source_notebook.cells)

    target_notebook = ensure_ids(target_notebook)

    try:
        with open(filepath, "w") as target_file:
            nbformat.write(target_notebook, target_file)
        print(f"'next-steps.ipynb' appended to '{filepath}'")
    except Exception as e:
        print(f"Error appending notebooks: {e}")

def add_upgrade(filepath):
    """Appends the contents of 'upgrade-validmind.ipynb' to the specified notebook if the user agrees."""
    source_notebook_path = os.path.join(os.path.dirname(__file__), "upgrade-validmind.ipynb")

    if not os.path.exists(source_notebook_path):
        print(f"Source notebook '{source_notebook_path}' does not exist")
        return

    user_input = input("Do you want to include information about upgrading ValidMind? (yes/no): ").strip().lower()
    if user_input not in ("yes", "y"):
        print("Skipping appending 'upgrade-validmind.ipynb'")
        return

    try:
        with open(filepath, "r") as target_file:
            target_notebook = nbformat.read(target_file, as_version=4)

        with open(source_notebook_path, "r") as source_file:
            source_notebook = nbformat.read(source_file, as_version=4)
    except Exception as e:
        print(f"Error reading notebooks: {e}")
        return

    target_notebook.cells.extend(source_notebook.cells)
    target_notebook = ensure_ids(target_notebook)

    try:
        with open(filepath, "w") as target_file:
            nbformat.write(target_notebook, target_file)
        print(f"'upgrade-validmind.ipynb' appended to '{filepath}'")
    except Exception as e:
        print(f"Error appending notebooks: {e}")

if __name__ == "__main__":

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    filepath = create_notebook()
    if filepath:
        set_title(filepath)
        add_about(filepath)
        add_install(filepath)
        next_steps(filepath)
        add_upgrade(filepath)