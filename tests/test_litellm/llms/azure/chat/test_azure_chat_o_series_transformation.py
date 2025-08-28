import json
import os
import sys
import traceback
from typing import Callable, Optional
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path
import litellm
from litellm.llms.azure.chat.o_series_transformation import AzureOpenAIO1Config


@pytest.fixture
def config():
    return AzureOpenAIO1Config()


def test_azure_chat_o_series_transform_request(config: AzureOpenAIO1Config):
    request = config.transform_request(
        model="o_series/web-interface-o1-mini",
        messages=[],
        optional_params={},
        litellm_params={},
        headers={},
    )
    assert request["model"] == "web-interface-o1-mini"


def test_azure_chat_o_series_transform_request_with_deployment_name(config: AzureOpenAIO1Config):
    request = config.transform_request(
        model="o_series/web-interface-o1-mini",
        messages=[],
        optional_params={},
        litellm_params={"azure_deployment_name": "web-interface-o1-mini-deployment"},
        headers={},
    )
    assert request["model"] == "web-interface-o1-mini-deployment"


@pytest.mark.asyncio
async def test_azure_chat_o_series_transformation(config: AzureOpenAIO1Config):
    model = "o_series/web-interface-o1-mini"
    messages = [{"role": "user", "content": "Hello, how are you?"}]
    optional_params = {}
    litellm_params = {}
    headers = {}

    response = await config.async_transform_request(
        model, messages, optional_params, litellm_params, headers
    )
    print(response)
    assert response["model"] == "web-interface-o1-mini"
