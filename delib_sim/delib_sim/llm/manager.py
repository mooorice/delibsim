from typing import List, Dict, Any, Optional, Union, Literal
import os
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import AsyncOpenAI

from ..utils.config import LLMSetup, LLMProvider

OPENROUTER_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

# Map provider enum to OpenRouter provider prefix
PROVIDER_PREFIX_MAP = {
    LLMProvider.OPENAI: "openai",
    LLMProvider.ANTHROPIC: "anthropic",
    LLMProvider.XAI: "x-ai",  # xAI models use "x-ai" prefix on OpenRouter
    LLMProvider.GEMINI: "google",  # Gemini models use "google" prefix
    LLMProvider.DEEPSEEK: "deepseek",
    LLMProvider.MOONSHOT: "moonshotai",
    LLMProvider.ALIBABA: "qwen",
    LLMProvider.META: "meta-llama",
    LLMProvider.GOOGLE: "google",
}


def _build_openrouter_headers() -> Optional[Dict[str, str]]:
    """Build optional OpenRouter HTTP headers if env vars are provided."""
    headers: Dict[str, str] = {}
    referer = os.getenv("OPENROUTER_SITE_URL")
    if referer:
        headers["HTTP-Referer"] = referer
    title = os.getenv("OPENROUTER_SITE_NAME")
    if title:
        headers["X-Title"] = title
    return headers or None


# Mapping from config model names to actual OpenRouter API model names
# This allows using different config keys for the same model (e.g., with/without reasoning)
MODEL_NAME_MAPPING = {
    # GPT models
    "openai/gpt-5.1-reasoning": "openai/gpt-5.1",
    "openai/gpt-5.1-no-reasoning": "openai/gpt-5.1",
    # Google models
    "google/gemini-3-pro-preview-03-25-reasoning": "google/gemini-3-pro-preview",
    "google/gemini-3-pro-preview-03-25-no-reasoning": "google/gemini-3-pro-preview",
    # Deepseek models
    "deepseek/deepseek-v3.2-exp-reasoning": "deepseek/deepseek-v3.2-exp",
    "deepseek/deepseek-v3.2-exp-no-reasoning": "deepseek/deepseek-v3.2-exp",
    # Moonshot models
    "moonshotai/kimi-k2-thinking-reasoning": "moonshotai/kimi-k2-thinking",
    "moonshotai/kimi-k2-thinking-no-reasoning": "moonshotai/kimi-k2-thinking",
    # Meta models (no mapping needed, but keeping for consistency)
    # Alibaba models (no mapping needed, but keeping for consistency)
}

def _normalize_model_name(model_name: str, provider: LLMProvider) -> str:
    """Normalize model name to OpenRouter format (provider/model-name).
    
    First checks if there's a mapping from config name to API name.
    If model_name already contains a '/', assume it's already in the correct format.
    Otherwise, prefix it with the provider prefix.
    """
    # Check if there's a mapping for this config model name
    if model_name in MODEL_NAME_MAPPING:
        model_name = MODEL_NAME_MAPPING[model_name]
    
    if "/" in model_name:
        # Already has provider prefix, use as-is
        return model_name
    
    prefix = PROVIDER_PREFIX_MAP.get(provider)
    if prefix:
        return f"{prefix}/{model_name}"
    
    # Fallback: return as-is if provider not in map
    return model_name


class LLMInstance:
    """Wrapper for an LLM client instance accessed via OpenRouter."""

    def __init__(self, instance_id: str, setup: LLMSetup):
        self.instance_id = instance_id
        self.setup = setup
        self.client: Optional[AsyncOpenAI] = None
        self.normalized_model_name = _normalize_model_name(setup.model_name, setup.provider)
        self._initialize_client()

    def _initialize_client(self):
        api_key = os.getenv(self.setup.api_key_env)
        if not api_key:
            raise ValueError(f"API key not found for environment variable {self.setup.api_key_env}")

        base_url = (
            os.getenv("OPENROUTER_BASE_URL")
            or self.setup.api_endpoint
            or OPENROUTER_DEFAULT_BASE_URL
        )

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=_build_openrouter_headers(),
        )
        print(
            f"[INFO] ({self.instance_id}) OpenRouter client ready for model "
            f"{self.normalized_model_name} (original: {self.setup.model_name}) via {base_url}"
        )

    def _extract_response_text(self, response: Any) -> str:
        """Normalize chat completion output into a plain string."""
        try:
            choice = response.choices[0]
            message_content = choice.message.content
        except (AttributeError, IndexError):
            return "[No Response]"

        if message_content is None:
            return "[No Response]"

        if isinstance(message_content, str):
            return message_content

        parts: List[str] = []
        for item in message_content:
            if isinstance(item, str):
                parts.append(item)
                continue
            text_value = getattr(item, "text", None)
            if text_value:
                parts.append(text_value)
                continue
            if isinstance(item, dict):
                text_value = item.get("text")
                if text_value:
                    parts.append(text_value)
        return "\n".join(parts) if parts else "[No Response]"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        tool_choice: Optional[Union[Literal["auto", "none", "required"], Dict[str, Any]]] = "auto",
    ) -> str:
        """Generate a response from the LLM using chat.completions with retry logic."""
        if not self.client:
            raise RuntimeError(f"LLM client not initialized for {self.instance_id}")

        request_args: Dict[str, Any] = {
            "model": self.normalized_model_name,
            "messages": messages,
        }

        if temperature is not None:
            request_args["temperature"] = temperature

        # Only send tool_choice if tools were configured. Some providers error when
        # tool_choice is supplied without any declared tools.
        # Gemini/Google models don't support tool_choice parameter at all, so skip it for them.
        if (tool_choice is not None 
            and self.setup.assistant_tools 
            and self.setup.provider not in [LLMProvider.GOOGLE, LLMProvider.GEMINI]):
            request_args["tool_choice"] = tool_choice

        # Add reasoning parameter if configured
        # Only include reasoning if both enabled and effort is specified (matching OpenRouter API format)
        if self.setup.reasoning_enabled is True and self.setup.reasoning_effort:
            effort = self.setup.reasoning_effort.lower()
            # Map effort values to valid OpenRouter values
            effort_map = {
                "low": "low",
                "medium": "medium",
                "high": "high",
                "minimal": "minimal",
            }
            mapped_effort = effort_map.get(effort, effort)
            
            # Pass reasoning via extra_body (OpenRouter supports this format)
            # The structure matches: {"reasoning": {"effort": "high"}}
            request_args["extra_body"] = {
                "reasoning": {
                    "effort": mapped_effort
                }
            }
            print(f"[INFO] ({self.instance_id}) Requesting reasoning with effort: {mapped_effort}")

        response = await self.client.chat.completions.create(**request_args)
        
        # Extract reasoning if available (supports both 'reasoning' and 'reasoning_details' attributes)
        try:
            msg = response.choices[0].message
            reasoning = getattr(msg, "reasoning", None)
            reasoning_details = getattr(msg, "reasoning_details", None)
            if reasoning:
                print(f"[INFO] ({self.instance_id}) Received reasoning content (length: {len(str(reasoning))} chars)")
            elif reasoning_details:
                print(f"[INFO] ({self.instance_id}) Received reasoning_details (length: {len(str(reasoning_details))} chars)")
        except (AttributeError, IndexError):
            pass
        
        final_content = self._extract_response_text(response)
        print(f"[INFO] ({self.instance_id}) Received response from {self.normalized_model_name}")
        return final_content


class LLMManager:
    """Manages multiple LLM instances for deliberation simulations."""

    def __init__(self, config: List[LLMSetup]):
        self.config = config
        self.instances: List[LLMInstance] = []

    async def initialize_instances(self):
        """Initialize all LLM instances based on configuration."""
        instance_count = 0

        for setup in self.config:
            if not setup.enabled:
                continue

            api_key = os.getenv(setup.api_key_env)
            if not api_key:
                print(f"Warning: API key for {setup.model_name} not found. Skipping.")
                continue

            for _ in range(setup.count):
                instance_id = f"{setup.model_name}-{instance_count}"
                try:
                    instance = LLMInstance(instance_id, setup)
                    self.instances.append(instance)
                    instance_count += 1
                except Exception as e:
                    print(f"Error initializing {instance_id}: {e}")

        print(f"[INFO] LLM Instance initialization complete. {len(self.instances)} instances ready.")

    def get_instances(self) -> List[LLMInstance]:
        """Get all LLM instances."""
        return self.instances
    
    def get_instances_by_model(self, model_name: str) -> List[LLMInstance]:
        """Get all instances of a specific model."""
        return [instance for instance in self.instances 
                if instance.setup.model_name == model_name]
    
    async def generate_responses(self, prompt: str, instances: List[LLMInstance] = None) -> Dict[str, str]:
        """Generate responses from multiple instances in parallel."""
        if instances is None:
            instances = self.instances
            
        responses = {}
        for instance in instances:
            try:
                response = await instance.generate(prompt)
                responses[instance.instance_id] = response
            except Exception as e:
                print(f"Failed to get response from {instance.instance_id}: {str(e)}")
                responses[instance.instance_id] = None
                
        return responses 

    async def cleanup_assistants(self):
        """Placeholder for legacy cleanup hooks."""
        print("[INFO] No assistant cleanup required when using OpenRouter chat completions.")