from .BaseModelTest import BaseModelTest
from deepeval.test_case import LLMTestCaseParams
from deepeval.metrics import GEval
import pandas as pd
from pathlib import Path
import json


class MME_TestModel(BaseModelTest):
    def set_eval(self):
        METRIC_MME = GEval(
            name="Correctness",
            evaluation_steps=[
                "Check if the choice made ('yes', 'no') in actual_output is exactly match with choice in expected output",
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
        self.metrics.append(METRIC_MME)

    def make_data(self, raw_data) -> pd.DataFrame:
        raw_data['image_url'] = raw_data['image'].apply(lambda image: BaseModelTest.pil_to_base64(image, 'JPEG'))
        raw_data['input'] = (
        "For the following question, directly answer 'Yes' or 'No', no need to explain: "
        + raw_data["question"]
        )
        raw_data['expected_output'] = raw_data['answer']
        raw_data['context'] = raw_data.apply(lambda row: [row['category'], row['question_id']], axis=1)
        self.data = raw_data
        return raw_data
    
    @staticmethod
    def output_results(input_file: Path) -> None:
        perception_tasks = [
            "existence", "count", "position", "color", "posters",
            "celebrity", "scene", "landmark", "artwork", "OCR"
        ]
        cognition_tasks = [
            "commonsense_reasoning", "numerical_calculation",
            "text_translation", "code_reasoning"
        ]
        with open(input_file, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data['testCases'])
        
        df['task'] = df['context'].str[0]
        df['image'] = df['context'].str[1]

        acc = df.groupby(['task'])['success'].mean()
        grouped = df.groupby(['task', 'image'])
        both_correct = grouped['success'].all()
        acc_plus = both_correct.groupby(['task']).mean()

        task_scores = (acc + acc_plus) * 100

        # Print results (assuming perception_tasks and cognition_tasks are defined lists)
        print("=========== Perception ===========")
        print("total score:", task_scores.loc[perception_tasks].sum())
        for task in perception_tasks:
            if task in task_scores:
                print(f"\t{task} score:", task_scores[task])

        print("\n=========== Cognition ===========")
        print("total score:", task_scores.loc[cognition_tasks].sum())
        for task in cognition_tasks:
            if task in task_scores:
                print(f"\t{task} score:", task_scores[task])

    

