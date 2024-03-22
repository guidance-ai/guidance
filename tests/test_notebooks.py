import pathlib

import papermill as pm

BASE_NB_PATH = pathlib.Path("./notebooks").absolute()


class TestTutorials:
    BASE_TUTORIAL_PATH = BASE_NB_PATH / "tutorials"

    def test_regex_constraints(self):
        nb_path = TestTutorials.BASE_TUTORIAL_PATH / "regex_constraints.ipynb"
        output_nb = nb_path.stem + ".papermill_out" + nb_path.suffix
        output_path = TestTutorials.BASE_TUTORIAL_PATH / output_nb

        pm.execute_notebook(input_path=nb_path, output_path=output_path)

    def test_guidance_acceleration(self):
        nb_path = TestTutorials.BASE_TUTORIAL_PATH / "guidance_acceleration.ipynb"
        output_nb = nb_path.stem + ".papermill_out" + nb_path.suffix
        output_path = TestTutorials.BASE_TUTORIAL_PATH / output_nb

        pm.execute_notebook(input_path=nb_path, output_path=output_path)