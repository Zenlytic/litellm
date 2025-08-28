import os
import sys

# Add the parent directory to the system path
sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.types.router import GenericLiteLLMParams
from litellm.utils import get_llm_provider


def test_azure_model_name_unchanged_when_deployment_name_absent():
    litellm_params = GenericLiteLLMParams(custom_llm_provider="azure")
    model, custom_llm_provider, _, _ = get_llm_provider(model="azure/gpt-4.1-deployment", litellm_params=litellm_params)
    assert custom_llm_provider == "azure"
    assert model == "gpt-4.1-deployment"


def test_azure_model_name_unchanged_when_deployment_name_different():
    litellm_params = GenericLiteLLMParams(custom_llm_provider="azure", azure_deployment_name="gpt-4.1-deployment")
    model, custom_llm_provider, _, _ = get_llm_provider(model="azure/gpt-4.1", litellm_params=litellm_params)
    assert custom_llm_provider == "azure"
    assert model == "gpt-4.1"
