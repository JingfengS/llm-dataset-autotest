import pytest
from src.models.MathVista_ModelTest import MathVista_ModelTest
from src.models.BaseModelTest import ModelTestConfig
from datasets import load_dataset
import pandas as pd
import os

os.environ["OPENAI_API_KEY"] = "your-api-key-here(could be blank if local llm)"


@pytest.fixture()
def dataset_fixture():
    ds = load_dataset("AI4Math/MathVista", split="testmini")
    df = pd.DataFrame(ds).iloc[:10, :]
    yield df


@pytest.mark.usefixtures("dataset_fixture")
class Test_MathVista:
    def test_full_mathvista(self, dataset_fixture):
        mathvista_config = ModelTestConfig(model_name='openai/internvl2_5')
        mathvista_model = MathVista_ModelTest(mathvista_config)
        mathvista_model.make_data(dataset_fixture)
        mathvista_model.make_goldens()
        mathvista_model.set_eval()
        mathvista_model.evaluate_llm()
        success_num, total_num, ratio = mathvista_model.get_success_rate()
        print("Success_num: ", success_num)
        print("total_num: ", total_num)
        print("ratio: ", ratio)
