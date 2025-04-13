from models.AdditionalTest import AdditionalTest
from models.BaseModelTest import ModelTestConfig
from pathlib import Path
import os
import argparse


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM on Additional dataset")
    parser.add_argument(
        "--model_name", type=str, required=True, help="Name of the model to evaluate"
    )
    parser.add_argument(
        "--remote_data_path",
        type=str,
        default="/home/qwen2-vl-72b-instruct/additional",
        help="Path to the Additional dataset",
    )
    parser.add_argument(
        "--results_folder", type=str, required=True, help="Path to save the results"
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
    args = parser.parse_args()
    os.environ["OPENAI_API_KEY"] = args.openai_api_key
    os.makedirs(args.results_folder, exist_ok=True)

    additional_model_config = ModelTestConfig("openai/" + args.model_name, args.api_base)
    additional_model = AdditionalTest(additional_model_config)
    additional_model.make_data(Path(args.remote_data_path))
    additional_model.make_goldens()
    additional_model.export_results(Path(args.results_folder))


if __name__ == "__main__":
    main()
