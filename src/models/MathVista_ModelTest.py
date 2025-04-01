from .BaseModelTest import BaseModelTest
from deepeval.test_case import LLMTestCaseParams
from deepeval.metrics import GEval
import pandas as pd
import json

class MathVista_ModelTest(BaseModelTest):
    def set_eval(self):
        METRIC_MATHVISTA = GEval(
            name="Correctness",
            evaluation_steps=[
                "Check if the answer in actual_output is match with the answer in expected output, note that for multiple choice problems, e.g. (A) Yes, (B) No, answer using either the letter (e.g. A) or the content (e.g. Yes) can both considered correct if it match the intended answer",
                "if it is, give it full scores, if not, 0 points",
                'You do not have to concern about the explanation, only looking at final results is fine'
            ],
            model='gpt-4o-mini',
            strict_mode=True,
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
                LLMTestCaseParams.CONTEXT,
            ],
        )
        self.metrics.append(METRIC_MATHVISTA)
    
    def make_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        raw_data['image_url'] = raw_data['decoded_image'].apply(BaseModelTest.pil_to_base64)
        raw_data['input'] = raw_data['query']
        raw_data['expected_output'] = raw_data['answer']
        raw_data['context'] = raw_data['question_type'].apply(lambda question_type: [question_type])
        self.data = raw_data
        return raw_data

    @staticmethod
    def output_results(input_file: str) -> None:
        """
        Generate the output results for MathVista Model Test

        Args:
            input_file (str): 
        """
        with open(input_file, 'r') as f:
            data = json.load(f)
        test_data = data['testCases']
        df = pd.DataFrame(test_data)
        correct_count = df['success'].sum()
        total_count = len(df)
        success_ratio = correct_count / total_count
        print("### Success Count")
        print(f"{correct_count} out of {total_count}")
        print("### Total Count")
        print(f"{total_count}")
        print("### Success Ratio")
        print(f"{success_ratio:.4f} (or {success_ratio*100:.2f}%)")
