from models.POPE_ModelTest import POPE_ModelTest
import pandas as pd
import json
from pathlib import Path
import os
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Calculate POPE results for all models"
    )
    parser.add_argument(
        "--results_path",
        type=str,
        required=True,
        help="Path to the evaluation results file",
    )
    args = parser.parse_args()
    results_path = Path(args.results_path)
    POPE_ModelTest.calculate_evaluation_results(results_path)



if __name__ == "__main__":
    main()
