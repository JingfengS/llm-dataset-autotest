import asyncio
import json
from dataclasses import dataclass
from litellm import completion
from PIL import Image
import base64
from io import BytesIO
import pandas as pd
from deepeval.dataset import EvaluationDataset, Golden
from typing import List
from tqdm import tqdm
from deepeval.evaluate import EvaluationResult, evaluate
from deepeval.test_case import LLMTestCase
from pathlib import Path
import pickle


@dataclass
class ModelTestConfig:
    model_name: str
    api_base: str = "http://localhost:2000/v1"
    api_key: str = "jingfeng"


class BaseModelTest:
    """
    The is the base class for model test, all the other specific test will be based on this
    """

    def __init__(self, model_test_config: ModelTestConfig):
        self.model_name: str = model_test_config.model_name
        self.api_base: str = model_test_config.api_base
        self.api_key: str = model_test_config.api_key
        self.data: pd.DataFrame = None
        self.goldens: List[Golden] = []
        self.metrics = []
        self.evaluation_result: EvaluationResult = None

    def set_eval(self):
        raise NotImplementedError("Subclass must implement this method")

    async def get_answer(self, message: str, image_url: str = "", **kwargs) -> str:
        """
        The request from the llm that we deployed
        @param message: The message we sent to the llm
        @param image_url: The image_url we sent to the llm to understand

        return the response from the llm
        """
        try:
            content = []
            if image_url:
                content.append({"type": "image_url", "image_url": image_url})
            content.append({"type": "text", "text": message})

            response = await asyncio.to_thread(
                completion,
                model=self.model_name,
                messages=[{"role": "user", "content": content}],
                api_base=self.api_base,
                api_key=self.api_key,
                **kwargs,
            )

            if "choices" in response and len(response["choices"]) > 0:
                return response["choices"][0]["message"]["content"]
            else:
                return "No valid response received"
        except Exception as e:
            print(f"An error occurred: {e}")
            return f"An error occurred: {e}"

    def make_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Make structured data from raw_data, the data better have these columns:
        @input
        @expected_output
        @image_url: Optional
        @context: Optional
        """
        raise NotImplementedError("Subclasses must have this method")

    def make_goldens(
        self,
        input: str = "input",
        expected_output: str = "expected_output",
        has_image: bool = True,
        has_context: bool = True,
        image_url: str = "image_url",
        context: str = "context",
        comments: str = None,
    ) -> List[Golden]:
        """
        Interact with llm to generate actual_output
        And make goldens for llm testing
        @param input: the column name of inputs in dataframe
        @param expected_output: the column name of expected_outputs in dataframe
        @param has_image: True if has image to upload, False otherwise, default True
        @param has_context: True if there is a context column, False otherwise, default True
        @param image_url: the column name of image_url in dataframe
        @param context: the column name of context in dataframe
        @param comments: The comments for these goldens
        """
        if self.data is None:
            raise ValueError("Must use make_data() first to create self.data.")
        self.goldens = []
        if not has_image:
            self.data[image_url] = ""
        if not has_context:
            self.data[context] = [""]

        async def generate_goldens():
            actual_outputs = await asyncio.gather(
                *(
                    self.get_answer(row[input], row[image_url])
                    for _, row in self.data.iterrows()
                )
            )

            for actual_output, (_, instance) in zip(
                actual_outputs, self.data.iterrows()
            ):
                golden = Golden(
                    input=instance[input],
                    actual_output=actual_output,
                    expected_output=instance[expected_output],
                    context=instance[context],
                    comments=comments,
                )
                self.goldens.append(golden)

        asyncio.run(generate_goldens())
        return self.goldens

    def evaluate_llm(
        self, is_push: bool = False, alias: str = "tmp"
    ) -> EvaluationResult:
        """
        Evaluate the llm based on the Goldens and metrics
        @param is_push: Push the results to confidentai or not, default False
        @param alias: the alias of the dataset if pushed to confidentai
        """
        if len(self.metrics) == 0:
            raise ValueError("Execute self.set_eval() to add metric to self.metrics")
        if len(self.goldens) == 0:
            raise ValueError(
                "Execute self.make_goldens() to add goldens to self.goldens"
            )

        dataset = EvaluationDataset(goldens=self.goldens)
        if is_push:
            dataset.push(alias, overwrite=True)
        test_cases = []
        for golden in self.goldens:
            test_case = LLMTestCase(
                input=golden.input,
                actual_output=golden.actual_output,
                expected_output=golden.expected_output,
                comments=golden.comments,
                context=golden.context,
            )
            test_cases.append(test_case)
        dataset.test_cases = test_cases
        self.evaluation_result = evaluate(
            dataset,
            metrics=self.metrics,
            ignore_errors=True,
            throttle_value=0.3,
            max_concurrent=10,
        )

    def get_success_rate(self):
        if self.evaluation_result == None:
            raise ValueError("Please evaluate the llm first")
        success_num = 0
        for test_case in self.evaluation_result.test_results:
            if test_case.success:
                success_num += 1
        return (
            success_num,
            len(self.evaluation_result.test_results),
            success_num / len(self.evaluation_result.test_results),
        )

    def save_goldens(self, output_path: Path) -> None:
        """
        Save goldens to pickles for future validation
        @param output_path: The path to save the pickle
        """
        if len(self.goldens) == 0:
            raise ValueError("Please make goldens first")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(self.goldens, f)
        print(f"Goldens saved to {output_path}")

    def load_goldens(self, input_path: Path) -> None:
        """
        Load the goldens pickle for validation
        @param input_path: path for the input pickle
        """
        if len(self.goldens) != 0:
            raise ValueError("self.goldens should have length of 0")
        with open(input_path, "rb") as f:
            self.goldens = pickle.load(f)
        print(f"Load pickle from {input_path}")

    def add_feature(self, golden_feature: str, data_feature: str) -> None:
        if len(self.goldens) == 0 or len(self.data) == 0:
            raise ValueError("Goldens or Data cannot be empty")
        for golden, (_, row) in zip(self.goldens, self.data.iterrows()):
            match golden_feature:
                case "input":
                    golden.input = row[data_feature]
                case "expected_output":
                    golden.expected_output = row[data_feature]
                case "context":
                    golden.context = row[data_feature]
                case "comments":
                    golden.comments = row[data_feature]

    @staticmethod
    def test_results_json2csv(input_path: Path, output_path: Path) -> pd.DataFrame:
        with open(input_path, "r") as results_json:
            json_data = json.load(results_json)
        df = pd.DataFrame(json_data["testCases"])
        df["context_str"] = df["context"].str[0]
        stats = df.groupby("context_str")["success"].agg(
            successful="sum", total="count"
        )
        stats["ratio"] = (stats["successful"] / stats["total"]) * 100
        stats = stats.reset_index()
        stats.to_csv(output_path, index=False)
        return stats

    @staticmethod
    def pil_to_base64(image: Image.Image, format: str = "JPEG") -> str:
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        buffered = BytesIO()
        image.save(buffered, format=format)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/{format.lower()};base64,{img_str}"


if __name__ == "__main__":
    base_config = ModelTestConfig(model_name="openai/internvl2_5")
    base_model = BaseModelTest(base_config)
    test1 = base_model.get_answer(
        "explain the content in this image", "/home/qwen2-vl-72b-instruct/yukino.jpeg"
    )
    print("TEST: The first response is describing yukino: ", test1)
    test2 = base_model.get_answer("what is 12 + 9?")
    print("TEST: The second problem is 12 + 9 = ?", test2)
