"""
Pydantic models for experiment execution configuration.

These types define the structure of provider, task, and evaluator configs stored in the database.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from phoenix.db.types.annotation_configs import OutputConfigType
from phoenix.db.types.evaluators import InputMapping
from phoenix.db.types.model_provider import ModelProvider
from phoenix.db.types.prompts import (
    PromptChatTemplate,
    PromptInvocationParameters,
    PromptResponseFormat,
    PromptTemplateFormat,
    PromptTemplateType,
    PromptTools,
)

# =============================================================================
# Prompt Version Config (mirrors models.PromptVersion structure)
# =============================================================================


class PromptVersionConfig(BaseModel):
    """
    Prompt version configuration for experiment execution.

    This structure mirrors models.PromptVersion to maintain consistency
    between stored prompts and experiment task configurations.
    """

    model_config = ConfigDict(frozen=True)

    # Template definition
    template_type: PromptTemplateType  # CHAT or STR
    template_format: PromptTemplateFormat  # F_STRING, MUSTACHE, or NONE
    template: PromptChatTemplate

    # Model configuration
    model_provider: ModelProvider
    model_name: str
    invocation_parameters: PromptInvocationParameters
    tools: PromptTools | None = None
    response_format: PromptResponseFormat | None = None

    # Custom provider ID (if using custom provider instead of builtin)
    # None = use builtin provider with secrets/env vars
    custom_provider_id: int | None = None


# =============================================================================
# Task Config Types
# =============================================================================


class TaskConfig(BaseModel):
    """Configuration for running LLM tasks in an experiment."""

    model_config = ConfigDict(frozen=True)

    # Prompt configuration (mirrors PromptVersion)
    prompt_version_config: PromptVersionConfig

    # Experiment-specific settings (not part of PromptVersion)
    template_variables_path: str | None = None
    appended_messages_path: str | None = None


# =============================================================================
# Evaluator Config Types
# =============================================================================


class EvaluatorConfig(BaseModel):
    """Configuration for a single evaluator."""

    model_config = ConfigDict(frozen=True)

    # Dataset evaluator database ID (numeric)
    dataset_evaluator_id: int

    input_mapping: InputMapping
    output_configs: Sequence[OutputConfigType] = Field(min_length=1)


class EvaluatorConfigs(BaseModel):
    """Configuration for all evaluators in an experiment."""

    model_config = ConfigDict(frozen=True)

    evaluators: Sequence[EvaluatorConfig]
