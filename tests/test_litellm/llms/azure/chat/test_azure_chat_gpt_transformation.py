import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
)

from litellm.llms.azure.chat.gpt_transformation import AzureOpenAIConfig


@pytest.fixture
def config():
    return AzureOpenAIConfig()


class TestAzureOpenAIConfig:
    def test_is_response_format_supported_model(self, config: AzureOpenAIConfig):
        # New logic: Azure deployment names with suffixes and prefixes
        assert config._is_response_format_supported_model("azure/gpt-4.1-suffix")
        assert config._is_response_format_supported_model("gpt-4.1-suffix")
        assert config._is_response_format_supported_model("azure/gpt-4-1-suffix")
        assert config._is_response_format_supported_model("gpt-4-1-suffix")
        # 4o models (should always be supported)
        assert config._is_response_format_supported_model("gpt-4o")
        assert config._is_response_format_supported_model("azure/gpt-4o-custom")
        # Backwards compatibility: base names
        assert config._is_response_format_supported_model("gpt-4.1")
        assert config._is_response_format_supported_model("gpt-4-1")
        # Negative test: clearly unsupported model
        assert not config._is_response_format_supported_model("gpt-3.5-turbo")
        assert not config._is_response_format_supported_model("gpt-3-5-turbo")
        assert not config._is_response_format_supported_model("gpt-3-5-turbo-suffix")
        assert not config._is_response_format_supported_model("gpt-35-turbo-suffix")
        assert not config._is_response_format_supported_model("gpt-35-turbo")


def test_azure_transform_request(config: AzureOpenAIConfig):
    request = config.transform_request(
        model="gpt-4o",
        messages=[],
        optional_params={},
        litellm_params={},
        headers={},
    )
    assert request["model"] == "gpt-4o"


def test_azure_transform_request_with_deployment_name(config: AzureOpenAIConfig):
    request = config.transform_request(
        model="gpt-4o",
        messages=[],
        optional_params={},
        litellm_params={"azure_deployment_name": "gpt-4o-deployment"},
        headers={},
    )
    assert request["model"] == "gpt-4o-deployment"


def test_map_openai_params_with_preview_api_version(config: AzureOpenAIConfig):
    non_default_params = {
        "response_format": {"type": "json_object"},
    }
    optional_params = {}
    model = "azure/gpt-4-1"
    drop_params = False
    api_version = "preview"
    assert config.map_openai_params(
        non_default_params, optional_params, model, drop_params, api_version
    )
