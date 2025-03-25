from .BaseModelTest import BaseModelTest
from deepeval.test_case import LLMTestCaseParams
from deepeval.metrics import GEval
import pandas as pd
import ast

class MMMU_Test_Model(BaseModelTest):
    def set_eval(self):
        METRIC_MME = GEval(
            name="Correctness",
            evaluation_steps=[
                "Check if the choice made ('A', 'B', 'C', 'D') in actual_output is exactly match with choice in expected output",
                "if it is, give it full scores, if not, 0 points",
                'You do not have to concern about the explanation, only looking at final results are fine'
            ],
            strict_mode=True,
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
                LLMTestCaseParams.CONTEXT,
            ],
        )
        self.metrics.append(METRIC_MME)

    def make_data(self, raw_data):
        def make_input(row) -> str:
            prompt = ''
            prompt += f"Question: {row['question']}"
            choices_text = ['Choices:']
            for i, choice in enumerate(ast.literal_eval(row['options'])):
                choices_text.append(f"({chr(ord('A')+i)}) {choice}")
            prompt += '\n' + '\n'.join(choices_text)
            # prompt += '\n' + 'HINT: You only need to return choices, no need to explain'
            return prompt
        raw_data['input'] = raw_data.apply(make_input, axis=1)
        raw_data['image_url'] = raw_data['image_1'].apply(lambda image: BaseModelTest.pil_to_base64(image, 'PNG'))
        raw_data['expected_output'] = raw_data['answer']
        raw_data['context'] = raw_data['subfield'].apply(lambda topic: [topic])
        self.data = raw_data
        return raw_data
