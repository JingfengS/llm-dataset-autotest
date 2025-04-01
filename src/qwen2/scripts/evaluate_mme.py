from models.BaseModelTest import ModelTestConfig, BaseModelTest
from models.MME_TestModel import MME_TestModel
import json
from datasets import load_dataset, get_dataset_config_names
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import os

os.environ['OPENAI_API_KEY'] = 'jingfeng'
os.environ['DEEPEVAL_RESULTS_FOLDER'] = '../results/mme_results'
MME_MODEL_CONFIG = ModelTestConfig(model_name='openai/qwen2')
MME_URL = 'darkyarding/MME'
MME_GOLDEN_PATH = Path('../Goldens/mme/mme_goldens.pkl')

if __name__ == '__main__':
    ds = load_dataset(MME_URL)['test']
    df = pd.DataFrame(ds)

    mme_model = MME_TestModel(MME_MODEL_CONFIG)
    mme_model.make_data(df)
    mme_model.make_goldens()
    mme_model.save_goldens(MME_GOLDEN_PATH)
    mme_model.set_eval()
    mme_model.evaluate_llm()
