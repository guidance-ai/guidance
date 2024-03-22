import pathlib

import papermill as pm

BASE_NB_PATH = pathlib.Path("./notebooks").absolute()


def run_notebook(notebook_path: pathlib.Path):
    output_nb = notebook_path.stem + ".papermill_out" + notebook_path.suffix
    output_path = TestTutorials.BASE_TUTORIAL_PATH / output_nb

    # Just make sure nothing throws an exception
    pm.execute_notebook(input_path=notebook_path, output_path=output_path)


class TestTutorials:
    BASE_TUTORIAL_PATH = BASE_NB_PATH / "tutorials"

    def test_adding_new_models(self):
        nb_path = TestTutorials.BASE_TUTORIAL_PATH / "adding_new_models.ipynb"
        run_notebook(nb_path)

    def test_regex_constraints(self):
        nb_path = TestTutorials.BASE_TUTORIAL_PATH / "regex_constraints.ipynb"
        run_notebook(nb_path)

    def test_token_healing(self):
        nb_path = TestTutorials.BASE_TUTORIAL_PATH / "token_healing.ipynb"
        run_notebook(nb_path)
