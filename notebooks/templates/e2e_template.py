import nbformat
import os

def create_notebook():
    """Creates a new Jupyter Notebook file by asking the user for a filename."""
    # Prompt the user for a filename
    filename = input("Enter the name for the new notebook (without .ipynb extension): ").strip()
    
    # Ensure the filename is valid
    if not filename:
        print("Filename cannot be empty.")
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
        print(f"Notebook '{filepath}' created successfully.")
    except Exception as e:
        print(f"Error creating notebook: {e}")

# Example usage
if __name__ == "__main__":
    create_notebook()
