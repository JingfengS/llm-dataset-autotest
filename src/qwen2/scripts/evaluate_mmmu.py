from models.BaseModelTest import ModelTestConfig, BaseModelTest
import json
from datasets import load_dataset, get_dataset_config_names
from models.MMMU_TestModel import MMMU_Test_Model
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import os

os.environ['OPENAI_API_KEY'] = 'jingfeng'
os.environ['DEEPEVAL_RESULTS_FOLDER'] = '../mmmu_results'
mmmu_goldens_path = Path('../../internvl/MMMU_Goldens_validation')
MMMU_URL = 'MMMU/MMMU'
MMMU_MODEL_CONFIG = ModelTestConfig(model_name='openai/internvl2_5')

if __name__ == '__main__':
    results = {}
    mmmu_configs = get_dataset_config_names(MMMU_URL)
    for config in tqdm(mmmu_configs, desc='Evaluating mmmu results'):
        mmmu_model = MMMU_Test_Model(MMMU_MODEL_CONFIG)
        mmmu_model.load_goldens(mmmu_goldens_path / (config + '.pkl'))
        mmmu_model.set_eval()
        mmmu_model.evaluate_llm()
        results[config] = mmmu_model.get_success_rate()
    results_json_path = Path(os.environ['DEEPEVAL_RESULTS_FOLDER']) / 'mmmu_results.json'
    with open(results_json_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)

    print(f"Results saved to {results_json_path}")
