from typing import List, Optional, Dict, Any, Union, Literal
from pydantic import BaseModel, Field, validator, root_validator, model_validator
from enum import Enum

from .surveys import get_survey_topic

class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    XAI = "xai"
    GEMINI = "gemini"
    DEEPSEEK = "deepseek"
    MOONSHOT = "moonshot"
    ALIBABA = "alibaba"
    META = "meta"
    GOOGLE = "google"
    # Add other providers as needed

class LLMSetup(BaseModel):
    model_name: str
    api_endpoint: str
    api_key_env: str
    count: int = Field(gt=0)
    provider: LLMProvider
    enabled: bool = True  # New field to enable/disable models
    reasoning_enabled: Optional[bool] = Field(default=None, description="Enable or disable reasoning for OpenRouter models. If None, reasoning is disabled.")
    reasoning_effort: Optional[str] = Field(default=None, description="Reasoning effort level for OpenRouter models. Valid values: 'low', 'medium', 'high', 'minimal'. Only used when reasoning_enabled is True.")
    thinking_config: Optional[Dict[str, Any]] = Field(default=None, description="Configuration for Anthropic's thinking feature (e.g., {'type': 'enabled', 'budget_tokens': 1000})")
    
    @model_validator(mode="before")
    @classmethod
    def migrate_reasoning_config(cls, data: Any) -> Any:
        """Migrate old reasoning_effort: 'enabled' format to new reasoning_enabled/reasoning_effort structure."""
        if isinstance(data, dict):
            # Handle backward compatibility: if reasoning_effort is "enabled", convert to reasoning_enabled=True
            if "reasoning_effort" in data and isinstance(data["reasoning_effort"], str):
                if data["reasoning_effort"].lower() == "enabled":
                    # Set reasoning_enabled to True if not already set
                    if "reasoning_enabled" not in data:
                        data["reasoning_enabled"] = True
                    # Remove the old "enabled" value from reasoning_effort
                    data["reasoning_effort"] = None
                elif data["reasoning_effort"].lower() == "none":
                    # "none" means disabled, ensure reasoning_enabled is False/None
                    if "reasoning_enabled" not in data:
                        data["reasoning_enabled"] = False
                    data["reasoning_effort"] = None
                elif data["reasoning_effort"].lower() in ["low", "medium", "high", "minimal"]:
                    # Valid effort value - ensure reasoning_enabled is True if not set
                    if "reasoning_enabled" not in data:
                        data["reasoning_enabled"] = True
        return data
    # --- OpenAI Assistant Specific Config --- 
    assistant_instructions: Optional[str] = Field(default="You are a helpful participant in a deliberation.", description="Optional template instructions included in prompts or metadata.")
    assistant_name_prefix: Optional[str] = Field(default="DelibSimAgent", description="Optional label prefix when logging agent personas.")
    assistant_tools: Optional[List[Dict[str, Any]]] = Field(default=None, description="Optional OpenAI Chat Completions tool definitions passed when supported.")
    assistant_tool_resources: Optional[Dict[str, Any]] = Field(default=None, description="Legacy assistant resource settings (not used with OpenRouter chat completions).")

class DeliberationProtocol(str, Enum):
    FREE_FLOW = "free_flow"
    GUIDED = "guided"
    # Add other protocols as needed

class GroupAssignmentStrategy(str, Enum):
    RANDOM = "random"
    BY_MODEL = "by_model"
    MIXED = "mixed"  # Same as by_model but more explicit naming
    INDIVIDUAL_MODEL = "individual_model"  # All agents in a group use the same model
    # Add other strategies as needed

class SimulationConfig(BaseModel):
    llm_setup: List[LLMSetup]
    deliberation_protocol: DeliberationProtocol = DeliberationProtocol.FREE_FLOW
    num_deliberation_turns: int = Field(gt=0, default=20)
    num_repetitions: int = Field(gt=0, default=1)
    group_size: int = Field(gt=0, default=5)
    group_assignment_strategy: GroupAssignmentStrategy = GroupAssignmentStrategy.RANDOM
    random_seed: Optional[int] = None
    display_conversation: bool = Field(default=True)
    pre_deliberation_only: bool = Field(default=False)
    
    # New survey and topic config
    survey_name: Optional[str] = Field(default=None, description="Name of the survey sheet in data/dri_surveys/surveys.xlsx. Required if DRI assessment is enabled.")
    deliberation_topic: Optional[str] = Field(default=None, description="The topic or question for the deliberation. If omitted, it will be inferred from the survey metadata.")
    # --- Unified Context File Config --- 
    context_document_path: Optional[str] = Field(None, description="Path to the single context document for deliberation.")
    openai_vector_store_auto_delete_days: int = Field(
        default=1,
        gt=0,
        description="Legacy setting (unused with OpenRouter chat completions)."
    )
    global_temperature: Optional[float] = Field(None, description="Global temperature setting for LLMs, if applicable and not overridden by sweep.")
    deliberation_tool_choice: Union[Literal["auto", "none", "required"], Dict[str, Any]] = Field(
        default="auto",
        description="Tool choice for OpenAI Assistant runs during deliberation/post-survey. 'auto', 'none', 'required', or e.g. {'type': 'file_search'} to force."
    )

    # Add parallelization flag
    parallelize_repetitions: bool = Field(default=False, description="Run simulation repetitions in parallel using asyncio.gather.")
    
    # Deliberation prompt type: "treatment" (complex prompt), "control_1" or "control_2" (minimal prompt)
    deliberation_prompt_type: Optional[str] = Field(default="control_1", description="Type of deliberation prompt: 'treatment' for complex prompt, 'control_1' or 'control_2' for minimal prompt.")

    # New field for concurrency limit
    max_concurrent_runs: Optional[int] = Field(None, description=
        "Maximum number of simulation runs to execute in parallel during a sweep. "
        "If None or a non-positive value, concurrency is not explicitly limited by a semaphore "
        "(though it might be limited by other factors like number of configurations)."
    )
    # New field for repetition concurrency limit
    max_parallel_repetitions: Optional[int] = Field(None, description=
        "Maximum number of repetitions to execute in parallel within a single simulation run "
        "when parallelize_repetitions is True. If None or non-positive, all repetitions run concurrently."
    )

    dri_conversation_enabled: bool = Field(default=True, description="Enable or disable deliberation conversation between pre and post DRI surveys.")
    assess_dqi: bool = Field(default=False, description="Enable automatic DQI assessment of transcripts using AQuA.")

    @validator('group_size')
    def check_group_size(cls, value):
        if value <= 0:
            raise ValueError("Group size must be greater than 0")
        return value

    @model_validator(mode="after")
    def check_survey_name_if_dri_enabled(cls, values):
        if getattr(values, "dri_conversation_enabled", True):
            survey_name = values.survey_name
            if not survey_name:
                raise ValueError("survey_name must be set when DRI deliberation is enabled")

            if values.deliberation_topic is None:
                try:
                    values.deliberation_topic = get_survey_topic(survey_name)
                except Exception as exc:
                    raise ValueError(
                        f"Could not infer deliberation_topic for survey '{survey_name}': {exc}"
                    ) from exc
        return values

    class Config:
        # Remove this line to ensure enum members are used internally
        # use_enum_values = True
        pass # Add pass if Config becomes empty

def load_config(config_path: str) -> SimulationConfig:
    """Load simulation configuration from a YAML file."""
    import yaml
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    return SimulationConfig(**config_data) 