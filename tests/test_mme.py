import pytest
from src.models.MME_TestModel import MME_TestModel
from src.models.BaseModelTest import ModelTestConfig
from datasets import load_dataset
import pandas as pd
import os

os.environ['OPENAI_API_KEY'] = 'your-api-key-here(could be blank if local llm)'

@pytest.fixture(scope='class')
def dataset_fixture():
    MME_URL = "darkyarding/MME"
    dataset_mme = load_dataset(MME_URL)['test']
    mme_dataset_df = pd.DataFrame(dataset_mme).iloc[:5, :]
    yield mme_dataset_df

@pytest.mark.usefixtures('dataset_fixture')
class Test_MME:
    def test_full_mme(self, dataset_fixture):
        mme_test_config = ModelTestConfig(model_name="openai/internvl2_5")
        mme_test_model = MME_TestModel(mme_test_config)
        mme_test_model.make_data(dataset_fixture)
        mme_test_model.make_goldens()
        mme_test_model.set_eval()
        mme_test_model.evaluate_llm()
        success_num, total_num, ratio = mme_test_model.get_success_rate()
        print("Success_num: ", success_num)
        print("total_num: ", total_num)
        print("ratio: ", ratio)
      
