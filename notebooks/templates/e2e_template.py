import nbformat
import os

def create_notebook():
    """Creates a new Jupyter Notebook file by asking the user for a filename."""
    # Prompt the user for a filename
    filename = input("Enter the name for the new notebook (without .ipynb extension): ").strip()
    
    # Ensure the filename is valid
    if not filename:
        print("Filename cannot be empty")
        return
    
    # Add the .ipynb extension if not provided
    if not filename.endswith(".ipynb"):
        filename += ".ipynb"

    # Define the directory to save the file (relative to the current script's location)
    current_dir = os.path.dirname(__file__)
    directory = os.path.join(current_dir, "..", "code_sharing")

    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Full file path
    filepath = os.path.join(directory, filename)

    # Create a new notebook object
    notebook = nbformat.v4.new_notebook()

    # Add some default metadata (optional)
    notebook["metadata"] = {
        "kernelspec": {
            "name": "python3",
            "display_name": "Python 3",
            "language": "python"
        },
        "language_info": {
            "name": "python",
            "version": "3.10"  # Modify based on your Python version
        }
    }

    # Write the notebook to a file
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

    # Load the existing notebook
    try:
        with open(filepath, "r") as f:
            notebook = nbformat.read(f, as_version=4)
    except Exception as e:
        print(f"Error reading notebook: {e}")
        return

    # Prompt the user for a title
    title = input("Enter the title for the notebook: ").strip()
    if not title:
        print("Title cannot be empty")
        return

    # Create a markdown cell with the title
    markdown_cell = nbformat.v4.new_markdown_cell(f"# {title}")

    # Add the markdown cell at the beginning of the notebook
    notebook.cells.insert(0, markdown_cell)

    # Write the updated notebook back to the file
    try:
        with open(filepath, "w") as f:
            nbformat.write(notebook, f)
        print(f"'{title}' added to '{filepath}' as title")
    except Exception as e:
        print(f"Error updating notebook: {e}")

def add_about(filepath):
    """Appends the contents of 'about-validmind.ipynb' to the specified notebook if the user agrees."""
    source_notebook_path = os.path.join(os.path.dirname(__file__), "about-validmind.ipynb")

    # Check if the source notebook exists
    if not os.path.exists(source_notebook_path):
        print(f"Source notebook '{source_notebook_path}' does not exist")
        return

    # Ask the user if they want to append the source notebook
    user_input = input("Do you want to include 'about-validmind.ipynb'? (yes/no): ").strip().lower()
    if user_input not in ("yes", "y"):
        print("Skipping appending 'about-validmind.ipynb'")
        return

    # Load the target and source notebooks
    try:
        with open(filepath, "r") as target_file:
            target_notebook = nbformat.read(target_file, as_version=4)

        with open(source_notebook_path, "r") as source_file:
            source_notebook = nbformat.read(source_file, as_version=4)
    except Exception as e:
        print(f"Error reading notebooks: {e}")
        return

    # Append the cells from the source notebook to the target notebook
    target_notebook.cells.extend(source_notebook.cells)

    # Write the updated target notebook back to the file
    try:
        with open(filepath, "w") as target_file:
            nbformat.write(target_notebook, target_file)
        print(f"Contents of 'about-validmind.ipynb' appended to '{filepath}' successfully.")
    except Exception as e:
        print(f"Error appending notebooks: {e}")

# Example usage
if __name__ == "__main__":
    filepath = create_notebook()
    if filepath:
        add_title(filepath)
        add_about(filepath)