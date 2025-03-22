import pytest
from src.models.MME_TestModel import MME_TestModel
from src.models.BaseModelTest import ModelTestConfig

@pytest.fixture(scope='class')
def mme_test_fixture():
    mme_test_config = ModelTestConfig(model_name="openai/internvl2_5")