from .BaseModelTest import BaseModelTest
from deepeval.test_case import LLMTestCaseParams
from deepeval.metrics import GEval
import pandas as pd


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
    
    

