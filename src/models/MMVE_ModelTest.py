from .BaseModelTest import BaseModelTest
from deepeval.test_case import LLMTestCaseParams
from deepeval.metrics import GEval
import pandas as pd
import json


class MMVE_ModelTest(BaseModelTest):
    def set_eval(self):
        METRIC_MMVE = GEval(
            name="Correctness",
            evaluation_steps=[
                "Check if the answer in actual_output is match with the answer in expected output, note that for multiple choice problems, e.g. (A) Yes, (B) No, answer using either the letter (e.g. A) or the content (e.g. Yes) can both considered correct if it match the intended answer",
                "if it is, give it full scores, if not, 0 points",
                "You do not have to concern about the explanation, only looking at final results is fine",
            ],
            model="gpt-4o-mini",
            strict_mode=True,
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
                LLMTestCaseParams.CONTEXT,
            ],
        )
        self.metrics.append(METRIC_MMVE)

    def make_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        raw_data["image_url"] = raw_data["image"].apply(
            lambda image: BaseModelTest.pil_to_base64(image, "JPEG")
        )
        raw_data["input"] = raw_data["question"]
        raw_data["expected_output"] = raw_data["answer"]
        raw_data["context"] = raw_data["capability"]
        copies = []
        for i in range(5):
            copy = raw_data.copy()
            copy['context'] = copy['context'].apply(lambda x: x + [f'copy_{i}'])
            copies.append(copy)
        self.data = pd.concat(copies, ignore_index=True)
        self.data = self.data.sample(frac=1, ignore_index=True)
        return self.data

    @staticmethod
    def calculate_overall_capabilities(input_file: str) -> None:
        with open(input_file, 'r') as f:
            data = json.load(f)
        test_data = data['testCases']
        df = pd.DataFrame(test_data)
        df['context'] = df['context'].apply(lambda x: x if isinstance(x, list) else [])
        expanded_df = df.explode('context')
        expanded_df = expanded_df.rename(columns={'context': 'capability'})
        success_ratios = expanded_df.groupby('capability')['success'].mean()
        success_ratios_by_copy = success_ratios.loc[['copy_0', 'copy_1', 'copy_2', 'copy_3', 'copy_4']]
        success_ratios_std = success_ratios_by_copy.std()
        overall_success_ratio = df['success'].mean()
        print("### Success Ratios by Capability")
        print(success_ratios.reset_index().to_string(index=False))

        print("\n### Overall Success Ratio")
        print(f"{overall_success_ratio:.4f} (or {overall_success_ratio*100:.2f}%)")

        print("### Success Ratios Standard Deviation")
        print(f"{success_ratios_std}\n\n")

    @staticmethod
    def calculate_multi_capabilities(input_file: str) -> None:
        """
        Generate the output results for MMVE Model Test by capabilities list

        Args:
            input_file (str): 
                The path to the input file containing the test results in JSON format.
        """
        with open(input_file, 'r') as f:
            data = json.load(f)
        test_data = data['testCases']
        df = pd.DataFrame(test_data)
        df['context'] = df['context'].apply(lambda x: x if isinstance(x, list) else [])
        df['context'] = df['context'].apply(lambda x: x[:-1])
        df['context_str'] = df['context'].apply(lambda x : ', '.join(x))
        success_ratios = df.groupby('context_str')['success'].mean().reset_index()
        overall_success_ratio = df['success'].mean()
        print("### Success Ratios by Context")
        print(success_ratios.to_string(index=False))
        print("\n### Overall Success Ratio")
        print(f"{overall_success_ratio:.4f} (or {overall_success_ratio*100:.2f}%)")
        
