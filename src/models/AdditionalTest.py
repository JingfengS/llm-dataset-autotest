from .BaseModelTest import BaseModelTest
from pathlib import Path
from deepeval.dataset import EvaluationDataset
import pandas as pd
import json
import os


class AdditionalTest(BaseModelTest):
    def make_data(self, remote_data_path: Path) -> pd.DataFrame:
        data = {
            "testCases": [
                {
                    "input": "Where do you think this photo was taken? What visual clues support your answer?",
                    "expected_output": "",
                    "image_url": "COCO_val2014_000000105156.jpg",
                },
                {
                    "input": "What food item is shown in this image?",
                    "expected_output": "",
                    "image_url": "COCO_val2014_000000022861.jpg",
                },
                {
                    "input": "What type of architecture is shown in this image and from which culture does it originate?",
                    "expected_output": "",
                    "image_url": "COCO_val2014_000000258529.jpg",
                },
                {
                    "input": "Based on the surroundings, in which room of a house was this photo taken?",
                    "expected_output": "",
                    "image_url": "COCO_val2014_000000209747.jpg",
                },
                {
                    "input": "Describe the following image",
                    "expected_output": "",
                    "image_url": "413.jpg",
                },
                {
                    "input": "Describe the following image",
                    "expected_output": "",
                    "image_url": "6070.jpg",
                },
            ]
        }
        self.data = pd.DataFrame(data["testCases"])
        self.data["image_url"] = self.data["image_url"].apply(
            lambda x: os.path.join(remote_data_path, x)
        )
        self.data["context"] = self.data.apply(
            lambda row: [row["input"], row["image_url"]], axis=1
        )
        return self.data

    def export_results(self, output_path: Path):
        """
        Export the results to a JSON file.

        Args:
            output_path (Path): The path where the results will be saved.
        """
        os.makedirs(output_path, exist_ok=True)
        dataset = EvaluationDataset(
            goldens=self.goldens,
        )
        dataset.save_as("json", output_path)
