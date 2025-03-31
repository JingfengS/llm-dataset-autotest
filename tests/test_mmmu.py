import pytest
from src.models.MMMU_TestModel import MMMU_Test_Model
from src.models.BaseModelTest import ModelTestConfig
from datasets import load_dataset
import pandas as pd
import os

os.environ['OPENAI_API_KEY'] = 'your-api-key-here(could be blank if local llm)'

@pytest.fixture(scope='class')
def dataset_fixture():
    MMMU_URL = 'MMMU/MMMU'
    dataset_mme = load_dataset(MMMU_URL, 'Accounting')
    mme_dataset_df = pd.DataFrame(dataset_mme['dev'])
    yield mme_dataset_df

@pytest.mark.usefixtures('dataset_fixture')
class Test_MMMU:
    def test_full_mmmu(self, dataset_fixture):
        mmmu_test_config = ModelTestConfig(model_name='openai/internvl2_5')
        mmmu_test_model = MMMU_Test_Model(mmmu_test_config)
        mmmu_test_model.make_data(dataset_fixture)
        mmmu_test_model.make_goldens()
        mmmu_test_model.save_goldens('./Goldens/mmmu_test_goldens.pkl') 
        mmmu_test_model.set_eval()
        mmmu_test_model.evaluate_llm()
        success_num, total_num, ratio = mmmu_test_model.get_success_rate()
        print("Success_num: ", success_num)
        print("total_num: ", total_num)
        print("ratio: ", ratio)
    
    def test_save_load_goldens(self, dataset_fixture):
        mmmu_test_config = ModelTestConfig(model_name='openai/internvl2_5')
        mmmu_test_model = MMMU_Test_Model(mmmu_test_config)
        mmmu_test_model.load_goldens('./tests/Goldens/mmmu_test_goldens.pkl')
        for golden in mmmu_test_model.goldens:
            print(f"Goldens_actual_output: ", golden.actual_output)
            print(f"Goldens_expected_output: ", golden.expected_output)
            print('\n\n')


        