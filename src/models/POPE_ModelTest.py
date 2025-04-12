import os
from .BaseModelTest import BaseModelTest
from deepeval.test_case import LLMTestCaseParams
from deepeval.metrics import GEval
import pandas as pd
from pathlib import Path
import json


class POPE_ModelTest(BaseModelTest):
    def set_eval(self):
      METRIC_POPE = GEval(
        name='Correctness',
        evaluation_steps=[
            "Check if the choice made ('yes', 'no') in actual_output is match with choice in expected output(note that the output could also be Chinese 是(yes) or 否/不是(no))",
            "if it is, give it full scores, if not, 0 points",
        ],
        strict_mode=True,
        model='gpt-4o-mini',
        evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
                LLMTestCaseParams.CONTEXT,
        ],
      )
      self.metrics.append(METRIC_POPE)

    def make_data(self, local_pope_path: Path, remote_coco_path: Path, pope_type: str) -> pd.DataFrame:
        json_filename = f'formatted_coco_pope_{pope_type}.json'
        json_path = os.path.join(local_pope_path, json_filename)
        raw_data = pd.read_json(json_path)
        raw_data['image_url'] = raw_data['image'].apply(lambda image_filename: self._get_image_path(image_filename, remote_coco_path))
        raw_data['input'] = raw_data['text']
        raw_data['expected_output'] = raw_data['label']
        raw_data['context'] = raw_data['question_id'].apply(lambda question_id: [pope_type, str(question_id)])
        self.data = raw_data
        return raw_data

    @staticmethod
    def _get_image_path(image_filename: Path, coco_path: Path) -> Path:
        return os.path.join(coco_path, "val2014", image_filename)


    @staticmethod
    def output_results(input_path: Path, output_path: Path):
        with open(input_path, 'r') as f:
            data = json.load(f)
        test_data = data['testCases']
        df = pd.DataFrame(test_data)

