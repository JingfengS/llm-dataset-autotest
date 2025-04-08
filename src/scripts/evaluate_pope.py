from models.BaseModelTest import ModelTestConfig
from models.POPE_ModelTest import POPE_ModelTest
from pathlib import Path
import os
import argparse


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM on POPE dataset")
    parser.add_argument(
        "--model_name", type=str, required=True, help="Name of the model to evaluate"
    )
    parser.add_argument(
        "--local_pope_path",
        type=str,
        required=True,
        default="/Users/jingfengsun/Documents/model/models/src/data/pope-coco",
        help="Path to the POPE dataset",
    )
    parser.add_argument(
        "--remote_coco_path",
        type=str,
        required=True,
        default="/home/qwen2-vl-72b-instruct/POPE-main/data/coco",
        help="Path to the COCO dataset",
    )
    parser.add_argument(
        "--pope_type",
        type=str,
        default="all",
        choices=["all", "adversarial", "popular", "random"],
        help="Type of POPE dataset to evaluate on",
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
    args = parser.parse_args()
    os.environ["OPENAI_API_KEY"] = args.openai_api_key
    pope_types = (
        ["adversarial", "popular", "random"]
        if args.pope_type == "all"
        else [args.pope_type]
    )
    for pope_type in pope_types:
        os.makedirs(os.path.join(args.results_folder, pope_type), exist_ok=True)
        os.makedirs(os.path.join(args.golden_save_path, pope_type), exist_ok=True)
        os.environ["DEEPEVAL_RESULTS_FOLDER"] = os.path.join(
            args.results_folder, pope_type
        )
        pope_config = ModelTestConfig(
            model_name="openai/" + args.model_name, api_base=args.api_base
        )
        pope_model = POPE_ModelTest(pope_config)
        pope_model.set_eval()
        pope_model.make_data(args.local_pope_path, args.remote_coco_path, pope_type)
        pope_model.make_goldens()
        pope_model.save_goldens(
            os.path.join(args.golden_save_path, pope_type, "pope_goldens.pkl")
        )
        pope_model.evaluate_llm()


if __name__ == "__main__":
    main()
