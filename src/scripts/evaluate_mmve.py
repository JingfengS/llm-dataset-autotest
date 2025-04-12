from models.BaseModelTest import ModelTestConfig
from models.MMVE_ModelTest import MMVE_ModelTest
from models.MathVista_ModelTest import MathVista_ModelTest
from models.MME_TestModel import MME_TestModel
from models.MMMU_TestModel import MMMU_TestModel
from datasets import load_dataset
import pandas as pd
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM on MMVE dataset")
    parser.add_argument(
        "--model_name", type=str, required=True, help="Name of the model to evaluate"
    )
    parser.add_argument(
        "--results_folder", type=str, required=True, help="Path to save the results"
    )
    parser.add_argument(
        "--golden_save_path",
        type=str,
        required=True,
        help="Path to save the golden results",
    )
    parser.add_argument(
        "--openai_api_key", type=str, default="your-api-key", help="OpenAI API key"
    )
    parser.add_argument(
        "--api_base",
        type=str,
        default="http://localhost:2000/v1",
        help="Base URL for OpenAI API",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["mathvista", "mm-vet", "mmmu", "mme"],
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "--dataset_url",
        type=str,
        required=True,
        help="URL to the dataset",
    )
    args = parser.parse_args()
    os.environ["OPENAI_API_KEY"] = args.openai_api_key
    os.makedirs(args.results_folder, exist_ok=True)
    os.makedirs(args.golden_save_path, exist_ok=True)
    os.environ["DEEPEVAL_RESULTS_FOLDER"] = args.results_folder
    mmve_model_config = ModelTestConfig(
        model_name="openai/" + args.model_name, api_base=args.api_base
    )
    ds = load_dataset(args.dataset_url)

    


if __name__ == "__main__":
    ds = load_dataset("whyu/mm-vet")
    df = pd.DataFrame(ds["test"])

    mmve_model = MMVE_ModelTest(MMVE_MODEL_CONFIG)
    mmve_model.make_data(df)
    mmve_model.make_goldens()
    mmve_model.save_goldens("../Goldens/mmve/mmve_goldens.pkl")
    mmve_model.set_eval()
    mmve_model.evaluate_llm()
    main()
