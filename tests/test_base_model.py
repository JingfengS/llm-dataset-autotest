from src.models.BaseModelTest import ModelTestConfig, BaseModelTest
import pytest


@pytest.fixture(scope="class")
def basemodel_fixture():
    base_config = ModelTestConfig(model_name="openai/internvl2_5")
    base_model = BaseModelTest(base_config)
    yield base_model
    print("Test Finish!")


@pytest.mark.usefixtures("basemodel_fixture")
class Test_Base_Model:
    def test_get_answer(self, basemodel_fixture):
        test1 = basemodel_fixture.get_answer(
            "explain the content in this image",
            "/home/qwen2-vl-72b-instruct/yukino.jpeg",
        )
        print("TEST: The first response is describing yukino: ", test1)
        test2 = basemodel_fixture.get_answer("what is 12 + 9?")
        print("TEST: The second problem is 12 + 9 = ?", test2)

    def test_make_goldens_error(self, basemodel_fixture):
        with pytest.raises(ValueError):
            basemodel_fixture.make_goldens()
            
    def test_evaluate_llm_error(self, basemodel_fixture):
        with pytest.raises(ValueError):
            basemodel_fixture.evaluate_llm()
