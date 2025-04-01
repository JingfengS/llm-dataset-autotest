from models.BaseModelTest import ModelTestConfig
from models.MathVista_ModelTest import MathVista_ModelTest
from datasets import load_dataset
import pandas as pd
import os


os.environ["OPENAI_API_KEY"] = "jingfeng"
os.environ["DEEPEVAL_RESULTS_FOLDER"] = "../results/mathvista_results"
VISTA_MODEL_CONFIG = ModelTestConfig(model_name="openai/qwen2")
VISTA_GOLDEN_PATH = "../Goldens/mathvista/mathvista_goldens.pkl"

if __name__ == "__main__":
    ds = load_dataset('AI4Math/MathVista')
    df = pd.DataFrame(ds["testmini"]).iloc[:1000, :]

    vista_model = MathVista_ModelTest(VISTA_MODEL_CONFIG)
    vista_model.make_data(df)
    vista_model.make_goldens()
    vista_model.save_goldens(VISTA_GOLDEN_PATH)
    vista_model.set_eval()
    vista_model.evaluate_llm()
    success_num, total_num, ratio = vista_model.get_success_rate()
    print("Success_num: ", success_num)
    print("total_num: ", total_num)
    print("ratio: ", ratio)
    
