# Pull Azure tests into a separate file since we're having issues
# with Azure credentials expiring during the test runs

from ..utils import slowdown
from .nb_test_utils import BASE_NB_PATH, run_notebook


class TestModels:
    BASE_MODEL_PATH = BASE_NB_PATH / "api_examples" / "models"

    def test_azure_openai(self):
        call_delay_secs = slowdown()
        nb_path = TestModels.BASE_MODEL_PATH / "AzureOpenAI.ipynb"
        run_notebook(nb_path, params={"call_delay_secs": call_delay_secs})
