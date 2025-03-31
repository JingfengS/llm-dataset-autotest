from models.BaseModelTest import ModelTestConfig, BaseModelTest
from models.MME_TestModel import MME_TestModel
from datasets import load_dataset, get_dataset_config_names
from models.MMMU_TestModel import MMMU_Test_Model
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import os

os.environ['OPENAI_API_KEY'] = 'jingfeng'

if __name__ == '__main__':
    mmmu_goldens = Path('../MMMU_Goldens_validation')
    MMMU_URL = 'MMMU/MMMU'
    mmmu_configs = get_dataset_config_names(MMMU_URL)
    for config in tqdm(mmmu_configs):
        if os.path.exists(mmmu_goldens / (config + '.pkl')):
             print(f'{config + '.pkl'} Already exists, skip')
             continue
        mmmu_ds = load_dataset(MMMU_URL, config)['validation']
        mmmu_df = pd.DataFrame(mmmu_ds)
        mmmu_model_config = ModelTestConfig(model_name='openai/internvl2_5')
        mmmu_model = MMMU_Test_Model(mmmu_model_config)
        mmmu_model.make_data(mmmu_df)
        mmmu_model.make_goldens()
        mmmu_model.save_goldens(mmmu_goldens / (config + '.pkl'))
    
