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
    def calculate_evaluation_results(input_path: Path) -> None:
        """ 
        Generate the output results for POPE Model Test
        This function reads the input JSON file, processes the test cases, and saves the results to a DataFrame.
        Then it prints out the Accuracy, Precision, Recall, and F1 score

        Args:
            input_path (Path): Path to the input JSON file containing test cases.
        """
        with open(input_path, 'r') as f:
            data = json.load(f)
        test_data = data['testCases']
        df = pd.DataFrame(test_data)
        df['expectedOutput'] = df['expectedOutput'].str.lower()
        df['prediction'] = df['expectedOutput']
        df.loc[~df['success'], 'prediction'] = df.loc[~df['success'], 'expectedOutput'].map({'yes': 'no', 'no': 'yes'})
        tp = ((df['expectedOutput'] == 'yes') & (df['prediction'] == 'yes')).sum()
        fp = ((df['expectedOutput'] == 'no') & (df['prediction'] == 'yes')).sum()
        tn = ((df['expectedOutput'] == 'no') & (df['prediction'] == 'no')).sum()
        
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + tn) if tp + tn > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        yes_ratio = df['prediction'].value_counts(normalize=True).get('yes', 0)

        accuracy = df['success'].mean()

        print("### Accuracy")
        print(f"{accuracy:.4f} (or {accuracy*100:.2f}%)")
        print("### Precision")
        print(f"{precision:.4f} (or {precision*100:.2f}%)")
        print("### Recall")
        print(f"{recall:.4f} (or {recall*100:.2f}%)")
        print("### F1 Score")
        print(f"{f1:.4f} (or {f1*100:.2f}%)")
        print('### Yes Ratio')
        print(f"{yes_ratio:.4f} (or {yes_ratio*100:.2f}%)")
