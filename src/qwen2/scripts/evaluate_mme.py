from models.BaseModelTest import ModelTestConfig, BaseModelTest
from models.MME_TestModel import MME_TestModel
import json
from datasets import load_dataset, get_dataset_config_names
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import os

os.environ['OPENAI_API_KEY'] = 'jingfeng'
os.environ['DEEPEVAL_RESULTS_FOLDER'] = '../mme_results'
MME_MODEL_CONFIG = ModelTestConfig(model_name='openai/')

if __name__ == '__main__':
    mme_goldens_path = Path('../../internvl/mme/goldens.pkl')
    mme_model = MME_TestModel(MME_MODEL_CONFIG)
    mme_model.load_goldens(mme_goldens_path)
    mme_model.set_eval()
    mme_model.evaluate_llm()
