from typing import Literal

from pydantic import Field

from letta.log import get_logger
from letta.schemas.enums import ProviderCategory, ProviderType
from letta.schemas.llm_config import LLMConfig
from letta.schemas.providers.openai import OpenAIProvider

logger = get_logger(__name__)


class MoonshotProvider(OpenAIProvider):
    """
    Moonshot AI (Kimi) provider.

    OpenAI-compatible API at https://api.moonshot.ai/v1.
    Models: kimi-k2.6, kimi-k2.5, moonshot-v1-{8k,32k,128k}.
    """

    provider_type: Literal[ProviderType.moonshot] = Field(ProviderType.moonshot, description="The type of the provider.")
    provider_category: ProviderCategory = Field(ProviderCategory.base, description="The category of the provider (base or byok)")
    api_key: str | None = Field(None, description="API key for the Moonshot AI API.", deprecated=True)
    base_url: str = Field("https://api.moonshot.ai/v1", description="Base URL for the Moonshot AI API.")

    def get_default_max_output_tokens(self, model_name: str) -> int:
        """K2.x models default to 32k output tokens, v1 models to 8k."""
        if model_name.startswith("kimi-k2"):
            return 32768
        return 8192

    async def list_llm_models_async(self) -> list[LLMConfig]:
        from letta.llm_api.openai import openai_get_model_list_async

        api_key = await self.api_key_enc.get_plaintext_async() if self.api_key_enc else None
        response = await openai_get_model_list_async(self.base_url, api_key=api_key)
        data = response.get("data", response)

        configs = []
        for model in data:
            assert "id" in model, f"Moonshot model missing 'id' field: {model}"
            model_name = model["id"]

            # Moonshot's /v1/models returns context_length directly
            context_window_size = model.get("context_length")
            if not context_window_size:
                logger.warning(f"Couldn't find context window size for Moonshot model {model_name}, skipping")
                continue

            configs.append(
                LLMConfig(
                    model=model_name,
                    model_endpoint_type="moonshot",
                    model_endpoint=self.base_url,
                    context_window=context_window_size,
                    handle=self.get_handle(model_name),
                    max_tokens=self.get_default_max_output_tokens(model_name),
                    provider_name=self.name,
                    provider_category=self.provider_category,
                )
            )

        return configs
