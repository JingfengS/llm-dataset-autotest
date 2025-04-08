import pytest
from src.models.BaseModelTest import ModelTestConfig
from src.models.MMVE_ModelTest import MMVE_ModelTest
from datasets import load_dataset
import pandas as pd
import os

os.environ["OPENAI_API_KEY"] = "your-api-key-here(could be blank if local llm)"


@pytest.fixture(scope="class")
def dataset_fixture():
    ds = load_dataset("whyu/mm-vet")
    df = pd.DataFrame(ds["test"]).iloc[:5, :]
    yield df


@pytest.mark.usefixtures("dataset_fixture")
class Test_MMVE:
    def test_full_mmve(self, dataset_fixture):
        mmve_test_config = ModelTestConfig(model_name="openai/qwen2")
        mmve_test_model = MMVE_ModelTest(mmve_test_config)
        mmve_test_model.make_data(dataset_fixture)
        mmve_test_model.make_goldens()
        mmve_test_model.save_goldens("./Goldens/mmve_test_goldens.pkl")
        mmve_test_model.set_eval()
        mmve_test_model.evaluate_llm()
        success_num, total_num, ratio = mmve_test_model.get_success_rate()
        print("Success_num: ", success_num)
        print("total_num: ", total_num)
        print("ratio: ", ratio)
