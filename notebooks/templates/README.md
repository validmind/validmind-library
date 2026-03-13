# Jupyter Notebook template QuickStart

Want to create some code samples for ValidMind? Our **[End-to-end notebook template generation](e2e-notebook.ipynb) notebook** will generate a new file with all the bits and pieces of a standard ValidMind notebook to get you started.

The same functionality is also accesible [in our root directory Makefile](../../Makefile) as a command:

```bash
make notebook
```

## Mini-templates

The template generation script/notebook draws from the following mini-templates, should you need to revise them or grab the information from them manually:

- [`about-validmind/`](about-validmind/): Conceptual overview of ValidMind & prerequisites.
- [`install-validmind/`](install-validmind/): ValidMind Library installation & initialization instructions.
- [`next-steps/`](next-steps/): Directions to review the generated documentation within the ValidMind Platform & additional learning resources.
- [`_upgrade-validmind.ipynb`](_upgrade-validmind.ipynb): Instructions for comparing & upgrading versions of the ValidMind Library.
- [`_copyright.ipynb`](_copyright.ipynb): Copyright and licensing info — mandatory at the bottom of every notebook owned by ValidMind.

Mini-templates are not meant to stand alone, and follow the filename convention of `_filename.ipynb`.

## Add table of contents

For lengthy notebooks, we recommend that you add a table of contents with the [**Simplified table of contents for Jupyter Notebooks extension**](https://github.com/validbeck/jupyter-notebook-toc/tree/main/installation).
