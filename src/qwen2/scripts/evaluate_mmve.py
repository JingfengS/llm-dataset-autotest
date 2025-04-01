from models.BaseModelTest import ModelTestConfig, BaseModelTest
from models.BaseModelTest import ModelTestConfig
from models.MMVE_ModelTest import MMVE_ModelTest
from datasets import load_dataset
import pandas as pd
import os


os.environ['OPENAI_API_KEY'] = 'jingfeng'
os.environ['DEEPEVAL_RESULTS_FOLDER'] = '../mmve_results'
MMVE_MODEL_CONFIG = ModelTestConfig(model_name='openai/internvl2_5')

if __name__ == '__main__':
    ds = load_dataset("whyu/mm-vet")
    df = pd.DataFrame(ds["test"])
    
    mmve_model = MMVE_ModelTest(MMVE_MODEL_CONFIG)
    mmve_model.make_data(df)
    mmve_model.make_goldens()
    mmve_model.save_goldens("../Goldens/mmve_test_goldens.pkl")
    mmve_model.set_eval()
    mmve_model.evaluate_llm()
    success_num, total_num, ratio = mmve_model.get_success_rate()
    print("Success_num: ", success_num)
    print("total_num: ", total_num) 