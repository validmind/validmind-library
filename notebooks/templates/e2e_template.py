import nbformat
import os
import uuid

def ensure_cell_ids(notebook):
    """Ensure every cell in the notebook has a unique id."""
    for cell in notebook.cells:
        if "id" not in cell:
            cell["id"] = str(uuid.uuid4())
    return notebook

def create_notebook():
    """Creates a new Jupyter Notebook file by asking the user for a filename."""
    filename = input("Enter the name for the new notebook (without .ipynb extension): ").strip()
    if not filename:
        print("Filename cannot be empty")
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

    # Ensure cells have IDs
    notebook = ensure_cell_ids(notebook)

    try:
        with open(filepath, "w") as f:
            nbformat.write(notebook, f)
        print(f"'{filepath}' created successfully")
    except Exception as e:
        print(f"Error creating notebook: {e}")

    return filepath

def add_title(filepath):
    """Adds a markdown cell with a title to the specified notebook."""
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
        print("Title cannot be empty")
        return

    markdown_cell = nbformat.v4.new_markdown_cell(f"# {title}")
    notebook.cells.insert(0, markdown_cell)

    # Ensure all cells have IDs
    notebook = ensure_cell_ids(notebook)

    try:
        with open(filepath, "w") as f:
            nbformat.write(notebook, f)
        print(f"'{title}' added to '{filepath}' as title")
    except Exception as e:
        print(f"Error updating notebook: {e}")

def add_about(filepath):
    """Appends the contents of 'about-validmind.ipynb' to the specified notebook if the user agrees."""
    source_notebook_path = os.path.join(os.path.dirname(__file__), "about-validmind.ipynb")

    if not os.path.exists(source_notebook_path):
        print(f"Source notebook '{source_notebook_path}' does not exist")
        return

    user_input = input("Do you want to include 'about-validmind.ipynb'? (yes/no): ").strip().lower()
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

    target_notebook.cells.extend(source_notebook.cells)

    # Ensure all cells have IDs
    target_notebook = ensure_cell_ids(target_notebook)

    try:
        with open(filepath, "w") as target_file:
            nbformat.write(target_notebook, target_file)
        print(f"Contents of 'about-validmind.ipynb' appended to '{filepath}' successfully")
    except Exception as e:
        print(f"Error appending notebooks: {e}")

# Example usage
if __name__ == "__main__":
    filepath = create_notebook()
    if filepath:
        add_title(filepath)
        add_about(filepath)