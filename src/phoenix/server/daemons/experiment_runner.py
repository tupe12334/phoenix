"""
Background experiment runner daemon.

Three concepts:
- ExperimentRunner: Daemon orchestrator with fair dispatch and concurrency control
- RunningExperiment: Per-experiment state with queues and rate limit awareness
- TaskJob / EvalJob: Self-executing command objects

Key patterns:
- Non-blocking rate limit check: Experiment checks capacity before returning work
- Jobs are self-executing with callbacks (Command pattern)
- Semaphore-first: Acquire concurrency slot before looking for work
"""

from __future__ import annotations

import asyncio
import heapq
import json
import logging
import random
import traceback
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import cached_property
from secrets import token_hex
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Hashable,
    Literal,
    Mapping,
    Protocol,
    Sequence,
    overload,
)

import anyio
from anyio import Semaphore
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from cachetools import LRUCache
from opentelemetry.context import Context as OtelContext
from sqlalchemy import select, update
from sqlalchemy.exc import SQLAlchemyError
from strawberry.relay import GlobalID
from typing_extensions import TypeAlias, override

from phoenix.config import EXPERIMENT_STALE_CLAIM_TIMEOUT
from phoenix.db import models
from phoenix.db.helpers import SupportedSQLDialect, get_experiment_incomplete_runs_query
from phoenix.db.insertion.helpers import OnConflict, insert_on_conflict
from phoenix.db.types.annotation_configs import OutputConfigType
from phoenix.db.types.evaluators import InputMapping
from phoenix.db.types.prompts import get_raw_invocation_parameters
from phoenix.evals.models.rate_limiters import AdaptiveTokenBucket, UnavailableTokensError
from phoenix.server.api.evaluators import (
    LLMEvaluator,
    evaluation_result_to_model,
    get_evaluators,
)
from phoenix.server.api.helpers.message_helpers import (
    build_template_variables,
    extract_and_convert_example_messages,
    formatted_messages,
    prompt_chat_template_to_playground_messages,
)
from phoenix.server.api.helpers.playground_clients import get_playground_client
from phoenix.server.api.helpers.playground_experiment_runs import (
    get_db_experiment_run,
)
from phoenix.server.api.types.ChatCompletionSubscriptionPayload import (
    ChatCompletionSubscriptionError,
    ChatCompletionSubscriptionPayload,
    ChatCompletionSubscriptionResult,
    EvaluationChunk,
)
from phoenix.server.api.types.DatasetExample import DatasetExample
from phoenix.server.api.types.Evaluator import DatasetEvaluator as DatasetEvaluatorNode
from phoenix.server.api.types.ExperimentRun import ExperimentRun
from phoenix.server.api.types.ExperimentRun import ExperimentRun as GqlExperimentRun
from phoenix.server.api.types.ExperimentRunAnnotation import ExperimentRunAnnotation
from phoenix.server.api.types.Span import Span as GqlSpan
from phoenix.server.api.types.Trace import Trace
from phoenix.server.types import DaemonTask, DbSessionFactory
from phoenix.tracers import Tracer
from phoenix.utilities.template_formatters import TemplateFormatterError

if TYPE_CHECKING:
    from phoenix.db.types.experiment_config import (
        EvaluatorConfig,
        TaskConfig,
    )
    from phoenix.server.api.evaluators import BaseEvaluator
    from phoenix.server.api.helpers.playground_clients import PlaygroundStreamingClient
    from phoenix.server.api.input_types.GenerativeCredentialInput import GenerativeCredentialInput


class TokenBucket(Protocol):
    def on_rate_limit_error(self, request_start_time: float, verbose: bool = False) -> None: ...
    def make_request_if_ready(self) -> None: ...


class _NoOpTokenBucket:
    def on_rate_limit_error(self, request_start_time: float, verbose: bool = False) -> None: ...
    def make_request_if_ready(self) -> None: ...


_NO_OP_TOKEN_BUCKET = _NoOpTokenBucket()


class TokenBucketRegistry(Protocol):
    """Read-only interface for accessing rate limit buckets by key."""

    def __getitem__(self, key: Hashable) -> TokenBucket: ...


class AutoCreateTokenBucketRegistry:
    """LRU cache that auto-creates AdaptiveTokenBucket on access."""

    def __init__(self, maxsize: int = 100) -> None:
        self._cache: LRUCache[Hashable, TokenBucket] = LRUCache(maxsize=maxsize)

    def __getitem__(self, key: Hashable) -> TokenBucket:
        if key is _NO_OP_TOKEN_BUCKET:
            return _NO_OP_TOKEN_BUCKET
        if key not in self._cache:
            self._cache[key] = AdaptiveTokenBucket(
                initial_per_second_request_rate=5.0,
                cooldown_seconds=0,
            )
        return self._cache[key]


logger = logging.getLogger(__name__)
# TODO: Remove before shipping - forces debug logging for development
logger.setLevel(logging.DEBUG)  # Override parent's level
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(_handler)

ExperimentId: TypeAlias = int

# =============================================================================
# Jobs (Command Pattern)
# =============================================================================


class Job(ABC):
    """
    Base class for self-executing jobs.

    Jobs are Commands (Gang of Four pattern) - they carry everything needed
    for execution and report results to their owning RunningExperiment.
    """

    # Owner - reports results back to this experiment
    _running_experiment: RunningExperiment

    @cached_property
    def running_experiment(self) -> RunningExperiment:
        """Owning experiment; reports results back to this."""
        return self._running_experiment

    @property
    @abstractmethod
    def debug_identifier(self) -> str:
        """Human-readable identifier for logging."""
        ...

    @property
    @abstractmethod
    def experiment_id(self) -> int:
        """Return the experiment ID this job belongs to."""
        ...

    @abstractmethod
    async def execute(self) -> None:
        """Execute this job. Results reported to owning RunningExperiment."""
        ...

    @abstractmethod
    def get_rate_limit_key(self) -> Hashable:
        """Return the rate limit key for this job's LLM client."""
        ...


class TaskJob(Job):
    """
    Task job: run LLM completion for one dataset example × one repetition.

    Carries everything needed for execution:
    - Data: example input, repetition number, messages
    - Config: prompt_version for creating client on demand
    - Behavior: timeout, db, decrypt
    - RunningExperiment reference for reporting results
    """

    def __init__(
        self,
        *,
        # Owner - reports results back to this experiment
        running_experiment: RunningExperiment,
        # Identity
        experiment: models.Experiment,
        dataset_example_revision: models.DatasetExampleRevision,
        repetition_number: int,
        # Execution context
        task_config: TaskConfig,
        llm_client: PlaygroundStreamingClient[Any],
        db: DbSessionFactory,
        decrypt: Callable[[bytes], bytes],
        tracer_factory: Callable[[], Tracer],
        project_id: int,
        # Optional parameters with defaults
        credentials: Sequence[GenerativeCredentialInput] | None = None,
        timeout: float = 120.0,
        retry_count: int = 0,
    ) -> None:
        # Owner (private; base class and _run_and_release use _running_experiment)
        self._running_experiment = running_experiment

        # Identity
        self._experiment = experiment
        self._dataset_example_revision = dataset_example_revision
        self._repetition_number = repetition_number

        # Execution context
        self._task_config = task_config
        self._llm_client = llm_client
        self._db = db
        self._decrypt = decrypt
        self._tracer_factory = tracer_factory
        self._project_id = project_id
        self._credentials = credentials
        self._timeout = timeout

        # Retry tracking
        self.retry_count = retry_count

    @cached_property
    def dataset_example_revision(self) -> models.DatasetExampleRevision:
        """Dataset example revision this task runs for."""
        return self._dataset_example_revision

    @cached_property
    def repetition_number(self) -> int:
        """Repetition index for this task (0-based)."""
        return self._repetition_number

    @cached_property
    def debug_identifier(self) -> str:
        return (
            f"task:experiment_id={self._experiment.id}, "
            f"dataset_example_id={self._dataset_example_revision.dataset_example_id}, "
            f"repetition={self._repetition_number}"
        )

    @cached_property
    def experiment_id(self) -> int:
        return self._experiment.id

    @override
    def get_rate_limit_key(self) -> Hashable:
        """Return the rate limit key for this job's LLM client."""
        return self._llm_client.get_rate_limit_key()

    @override
    async def execute(self) -> None:
        """Execute the task, write to DB, and report results."""
        logger.debug(f"TaskJob {self.debug_identifier} starting execution")
        tracer = self._tracer_factory()
        example_id = GlobalID(
            DatasetExample.__name__,
            str(self._dataset_example_revision.dataset_example_id),
        )
        revision = self._dataset_example_revision
        template_variables_path = self._task_config.template_variables_path
        appended_messages_path = self._task_config.appended_messages_path
        config = self._task_config.prompt_version_config
        template = config.template
        messages = prompt_chat_template_to_playground_messages(template)
        template_format = config.template_format
        format_start_time = datetime.now(timezone.utc)
        try:
            # Build template variables using shared helper
            template_variables = build_template_variables(
                input_data=revision.input,
                output_data=revision.output,
                metadata=revision.metadata_,
                template_variables_path=template_variables_path,
            )
            messages = list(
                formatted_messages(
                    messages=messages,
                    template_format=template_format,
                    template_variables=template_variables,
                )
            )
            # Append messages from dataset example if path is specified
            if appended_messages_path:
                appended = extract_and_convert_example_messages(
                    revision.input, appended_messages_path
                )
                messages.extend(appended)
        except (TemplateFormatterError, KeyError, TypeError, ValueError) as error:
            format_end_time = datetime.now(timezone.utc)
            error_payload = ChatCompletionSubscriptionError(
                message=str(error),
                dataset_example_id=example_id,
                repetition_number=self._repetition_number,
            )
            await self._running_experiment.on_task_chunk(error_payload)
            db_run = models.ExperimentRun(
                experiment_id=self._experiment.id,
                dataset_example_id=revision.dataset_example_id,
                trace_id=None,
                output={},
                repetition_number=self._repetition_number,
                start_time=format_start_time,
                end_time=format_end_time,
                error=str(error),
                trace=None,
            )

            with anyio.fail_after(30, shield=True):
                async with self._db() as session:
                    session.add(db_run)
            result_payload = ChatCompletionSubscriptionResult(
                span=None,
                experiment_run=ExperimentRun(id=db_run.id, db_record=db_run),
                dataset_example_id=GlobalID(
                    DatasetExample.__name__, str(revision.dataset_example_id)
                ),
                repetition_number=self._repetition_number,
            )
            await self._running_experiment.on_task_chunk(result_payload)
            await self._running_experiment.on_task_failure(self, error)
            return

        try:
            logger.debug(
                f"TaskJob {self.debug_identifier}: starting streaming (timeout={self._timeout}s)"
            )
            with anyio.fail_after(self._timeout):
                async for chunk in self._llm_client.chat_completion_create(
                    messages=messages,
                    tools=config.tools,
                    response_format=config.response_format,
                    invocation_parameters=get_raw_invocation_parameters(
                        config.invocation_parameters
                    ),
                    tracer=tracer,
                    otel_context=OtelContext(),
                ):
                    chunk.dataset_example_id = example_id
                    chunk.repetition_number = self._repetition_number
                    await self._running_experiment.on_task_chunk(chunk)

                logger.debug(f"TaskJob {self.debug_identifier}: stream finished, building traces")
                task_db_traces = tracer.get_db_traces(project_id=self._project_id)
                task_db_trace = task_db_traces[0] if task_db_traces else None
                if not task_db_trace or not task_db_trace.spans:
                    logger.warning(
                        f"TaskJob {self.debug_identifier}: no trace recorded "
                        "(stream may have completed without emitting spans)"
                    )
                    no_trace_error = RuntimeError(
                        "No trace recorded for completion (stream completed without spans)"
                    )
                    await self._running_experiment.on_task_failure(self, no_trace_error)
                    return
                db_span = task_db_trace.spans[0]
                db_run = get_db_experiment_run(
                    db_span,
                    task_db_trace,
                    experiment_id=self._experiment.id,
                    example_id=self._dataset_example_revision.dataset_example_id,
                    repetition_number=self._repetition_number,
                )

                with anyio.fail_after(30, shield=True):
                    async with self._db() as session:
                        session.add_all(task_db_traces)
                        stmt = insert_on_conflict(
                            {
                                "experiment_id": db_run.experiment_id,
                                "dataset_example_id": db_run.dataset_example_id,
                                "trace_id": db_run.trace_id,
                                "output": db_run.output,
                                "repetition_number": db_run.repetition_number,
                                "start_time": db_run.start_time,
                                "end_time": db_run.end_time,
                                "error": db_run.error,
                                "prompt_token_count": db_run.prompt_token_count,
                                "completion_token_count": db_run.completion_token_count,
                            },
                            table=models.ExperimentRun,
                            dialect=self._db.dialect,
                            unique_by=[
                                "experiment_id",
                                "dataset_example_id",
                                "repetition_number",
                            ],
                            on_conflict=OnConflict.DO_UPDATE,
                        ).returning(models.ExperimentRun)
                        db_run_result = await session.scalar(stmt)
                        assert db_run_result is not None
                        db_run = db_run_result

                result_payload = ChatCompletionSubscriptionResult(
                    span=GqlSpan(id=db_span.id, db_record=db_span),
                    experiment_run=GqlExperimentRun(id=db_run.id, db_record=db_run),
                    dataset_example_id=example_id,
                    repetition_number=self._repetition_number,
                )
                await self._running_experiment.on_task_chunk(result_payload)

                logger.debug(f"TaskJob {self.debug_identifier} completed successfully")
                await self._running_experiment.on_task_success(self, db_run)

        except TimeoutError:
            logger.warning(f"TaskJob {self.debug_identifier} timed out")
            await self._running_experiment.on_task_timeout(self)

        except anyio.get_cancelled_exc_class():
            # Task was canceled (e.g., server shutdown) - re-raise to propagate
            logger.debug(f"TaskJob {self.debug_identifier} cancelled")
            raise

        except Exception as e:
            # Classify error type using client's provider-specific logic
            err_type = type(e).__name__
            if self._is_rate_limit_error(e):
                logger.debug(f"TaskJob {self.debug_identifier} hit rate limit ({err_type})")
                await self._running_experiment.on_task_rate_limit(self)
            elif self._is_transient_error(e):
                # Network/transient errors - retry with backoff
                logger.warning(
                    f"TaskJob {self.debug_identifier} hit transient error ({err_type}): {e}"
                )
                await self._running_experiment.on_task_network_error(self, e)
            else:
                # Permanent failure - don't retry
                logger.exception(f"TaskJob {self.debug_identifier} failed ({err_type}): {e}")
                await self._running_experiment.on_task_failure(self, e)

    def _is_rate_limit_error(self, e: Exception) -> bool:
        """Check if exception is a rate limit error using client's provider-specific logic."""
        return bool(self._llm_client.is_rate_limit_error(e))

    def _is_transient_error(self, e: Exception) -> bool:
        """Check if exception is a transient/network error (should retry with backoff)."""
        if isinstance(e, SQLAlchemyError):
            return True
        return bool(self._llm_client.is_transient_error(e))


class EvalJob(Job):
    """
    Eval job: run one evaluator on one task result.

    No streaming - evaluators run silently.
    Results written to experiment_run_annotations table.
    """

    def __init__(
        self,
        *,
        # Owner - reports results back to this experiment
        running_experiment: RunningExperiment,
        # Identity
        experiment_run: models.ExperimentRun,
        dataset_example_revision: models.DatasetExampleRevision,
        # Evaluator config
        evaluator: BaseEvaluator,
        # Execution context
        db: DbSessionFactory,
        tracer_factory: Callable[[], Tracer],
        project_id: int,
        # Optional parameters with defaults
        input_mapping: InputMapping,
        output_configs: Sequence[OutputConfigType],
        timeout: float = 60.0,
        retry_count: int = 0,
    ) -> None:
        # Owner
        self._running_experiment = running_experiment

        # Identity
        self._experiment_run = experiment_run
        self._dataset_example_revision = dataset_example_revision

        # Evaluator config
        self._evaluator = evaluator
        self._input_mapping = input_mapping
        self._output_configs = output_configs

        # Execution context
        self._db = db
        self._tracer_factory = tracer_factory
        self._project_id = project_id
        self._timeout = timeout

        # Retry tracking
        self.retry_count = retry_count

    @cached_property
    def dataset_example_revision(self) -> models.DatasetExampleRevision:
        """Dataset example revision this eval runs for."""
        return self._dataset_example_revision

    @cached_property
    def experiment_run(self) -> models.ExperimentRun:
        """Experiment run this eval annotates."""
        return self._experiment_run

    @cached_property
    def evaluator(self) -> BaseEvaluator:
        return self._evaluator

    @cached_property
    def debug_identifier(self) -> str:
        return (
            f"eval:experiment_id={self._experiment_run.experiment_id}, "
            f"run_id={self._experiment_run.id}, evaluator_name={self._evaluator.name}"
        )

    @cached_property
    def experiment_id(self) -> ExperimentId:
        return self._experiment_run.experiment_id

    @override
    def get_rate_limit_key(self) -> Hashable:
        """Return the rate limit key for this job's evaluator.

        For LLM evaluators, delegates to the LLM client's rate limit key.
        For built-in evaluators (which run locally), returns a unique key
        so they can run without external rate limiting.
        """

        if isinstance(self._evaluator, LLMEvaluator):
            return self._evaluator.llm_client.get_rate_limit_key()
        return _NO_OP_TOKEN_BUCKET

    @override
    async def execute(self) -> None:
        """Execute the evaluation, write to DB, and report results."""
        logger.debug(f"EvalJob {self.debug_identifier}: starting (timeout={self._timeout}s)")
        try:
            with anyio.fail_after(self._timeout):
                context_dict: dict[str, Any] = {
                    "input": self._dataset_example_revision.input,
                    "reference": self._dataset_example_revision.output,
                    "output": self._experiment_run.output.get("task_output"),
                    "metadata": self._dataset_example_revision.metadata_,
                }

                tracer = self._tracer_factory()
                eval_results = await self._evaluator.evaluate(
                    context=context_dict,
                    input_mapping=self._input_mapping,
                    name=self._output_configs[0].name,
                    output_configs=self._output_configs,
                    tracer=tracer,
                )
                logger.debug(
                    f"EvalJob {self.debug_identifier}: evaluator returned "
                    f"{len(eval_results)} result(s)"
                )
                db_traces: list[models.Trace] = list(
                    tracer.get_db_traces(project_id=self._project_id)
                )
                annotations: list[models.ExperimentRunAnnotation] = []
                for result in eval_results:
                    annotation = evaluation_result_to_model(
                        result,
                        experiment_run_id=self._experiment_run.id,
                    )
                    annotations.append(annotation)

                with anyio.fail_after(30, shield=True):
                    async with self._db() as session:
                        session.add_all(db_traces)
                        session.add_all(annotations)

                # Check for evaluation error - treat as permanent failure for circuit breaker
                if any(r.get("error") is not None for r in eval_results):
                    error_msg = next(r["error"] for r in eval_results if r.get("error") is not None)
                    logger.warning(f"EvalJob {self.debug_identifier} returned error: {error_msg}")
                    await self._running_experiment.on_eval_failure(self, Exception(error_msg))
                else:
                    logger.debug(
                        f"EvalJob {self.debug_identifier}: success, "
                        f"wrote {len(annotations)} annotation(s)"
                    )
                    await self._running_experiment.on_eval_success(self, annotations, db_traces)

        except TimeoutError:
            logger.warning(f"EvalJob {self.debug_identifier} timed out")
            await self._running_experiment.on_eval_timeout(self)

        except Exception as e:
            # Classify error type
            err_type = type(e).__name__
            if self._is_rate_limit_error(e):
                logger.debug(f"EvalJob {self.debug_identifier} hit rate limit ({err_type})")
                await self._running_experiment.on_eval_rate_limit(self)
            elif self._is_transient_error(e):
                logger.warning(
                    f"EvalJob {self.debug_identifier} hit transient error ({err_type}): {e}"
                )
                await self._running_experiment.on_eval_network_error(self, e)
            else:
                logger.exception(f"EvalJob {self.debug_identifier} failed ({err_type}): {e}")
                await self._running_experiment.on_eval_failure(self, e)

    def _is_rate_limit_error(self, e: Exception) -> bool:
        """Check if exception is a rate limit error using evaluator's client."""

        if isinstance(self._evaluator, LLMEvaluator):
            return self._evaluator.llm_client.is_rate_limit_error(e)
        return False

    def _is_transient_error(self, e: Exception) -> bool:
        """Check if exception is a transient/network error using evaluator's client."""

        if isinstance(e, SQLAlchemyError):
            return True
        if isinstance(self._evaluator, LLMEvaluator):
            return self._evaluator.llm_client.is_transient_error(e)
        return False


# =============================================================================
# Circuit Breaker
# =============================================================================


class CircuitBreaker:
    """
    Simple consecutive-failure circuit breaker.

    Trips after N consecutive failures. One success resets the counter.
    Once tripped, stays tripped (experiment must be restarted).
    """

    def __init__(self, *, threshold: int = 5) -> None:
        self.threshold = threshold
        self._consecutive_failures: int = 0
        self._tripped: bool = False
        self._trip_reason: str | None = None

    def record_failure(self, error: Exception) -> bool:
        """Record a failure. Returns True if circuit just tripped."""
        if self._tripped:
            logger.debug("CircuitBreaker: already tripped, ignoring failure")
            return False
        self._consecutive_failures += 1
        logger.debug(
            f"CircuitBreaker: recorded failure "
            f"{self._consecutive_failures}/{self.threshold}: {error}"
        )
        if self._consecutive_failures >= self.threshold:
            self._tripped = True
            self._trip_reason = str(error)
            logger.warning(f"CircuitBreaker: TRIPPED after {self.threshold} consecutive failures")
            return True
        return False

    def record_success(self) -> None:
        """Record a success. Resets failure counter."""
        if self._consecutive_failures > 0:
            logger.debug(
                f"CircuitBreaker: success, resetting counter from {self._consecutive_failures}"
            )
        self._consecutive_failures = 0

    @property
    def is_tripped(self) -> bool:
        return self._tripped

    @property
    def trip_reason(self) -> str | None:
        return self._trip_reason


# =============================================================================
# Retry Item (for heap-based retry queue)
# =============================================================================


@dataclass(order=True)
class RetryItem:
    """Item in the retry queue, ordered by ready_at time."""

    ready_at: datetime
    job: Job = field(compare=False)


# =============================================================================
# Running Experiment
# =============================================================================


class OnExperimentDone(Protocol):
    def __call__(
        self,
        experiment_id: ExperimentId,
        *,
        last_error: str | None = None,
    ) -> None: ...


class RunningExperiment:
    """
    Per-experiment state: queues, rate limit check, creates Jobs.

    Key behavior:
    - try_get_ready_job() checks rate limit and returns a Job or None
    - If None (rate limited or no work), Daemon tries another experiment
    - Rate limit check returns immediately, never blocks waiting for capacity

    Priority: evals > retries > tasks
    """

    def __init__(
        self,
        *,
        experiment: models.Experiment,
        execution_config: models.ExperimentExecutionConfig,
        llm_client: PlaygroundStreamingClient[Any],
        db: DbSessionFactory,
        decrypt: Callable[[bytes], bytes],
        tracer_factory: Callable[[], Tracer],
        token_buckets: TokenBucketRegistry,
        on_done: OnExperimentDone,
        credentials: Sequence[GenerativeCredentialInput] | None = None,
        evaluator_by_id: Mapping[int, BaseEvaluator] = MappingProxyType({}),
        dataset_evaluator_name_by_id: Mapping[int, str] = MappingProxyType({}),
        max_retries: int = 3,
        base_backoff_seconds: float = 1.0,
    ) -> None:
        # Required dependencies
        self._experiment = experiment
        self._execution_config = execution_config
        self._llm_client = llm_client
        self._db = db
        self._decrypt = decrypt
        self._tracer_factory = tracer_factory
        self._token_buckets = token_buckets
        self._on_done = on_done

        # Optional configuration
        self._credentials = credentials
        self._evaluator_by_id = evaluator_by_id or {}
        self._dataset_evaluator_name_by_id = dataset_evaluator_name_by_id or {}
        self._max_retries = max_retries
        self._base_backoff_seconds = base_backoff_seconds

        # Queues (priority: evals > retries > tasks)
        self._task_queue: deque[TaskJob] = deque()
        self._eval_queue: deque[EvalJob] = deque()
        self._retry_heap: list[RetryItem] = []
        self._in_flight: dict[int, Job] = {}
        self._cancel_scopes: dict[int, anyio.CancelScope] = {}

        # UI subscribers for streaming
        self._subscribers: list[MemoryObjectSendStream[ChatCompletionSubscriptionPayload]] = []

        # Fairness tracking
        self.last_served_at: datetime = datetime.min.replace(tzinfo=timezone.utc)

        # Internal state
        self._active: bool = True

        # Counters for completion summary
        self._tasks_succeeded: int = 0
        self._tasks_failed: int = 0
        self._evals_succeeded: int = 0
        self._evals_failed: int = 0

        # Circuit breakers (separate for tasks and evals since they may use different providers)
        self._task_circuit_breaker = CircuitBreaker()
        self._eval_circuit_breaker = CircuitBreaker()  # TODO: per evaluator

        # Pagination: load tasks in batches to avoid memory exhaustion
        self._task_batch_size: int = 10
        self._task_db_offset: int = 0
        self._task_db_exhausted: bool = False

        # Cached project_id to avoid repeated DB lookups
        self._project_id: int | None = None

    def has_work(self) -> bool:
        """Does this experiment have pending work?"""
        if not self._active:
            logger.debug(f"Experiment {self._experiment.id}: has_work() -> False (inactive)")
            return False

        has_evals = bool(self._eval_queue)
        has_retries = bool(self._retry_heap)
        has_tasks = bool(self._task_queue)
        has_in_flight = bool(self._in_flight)
        has_more_in_db = not self._task_db_exhausted

        result = has_evals or has_retries or has_tasks or has_in_flight or has_more_in_db

        # Log detailed state when we have no work (helps debug stalls)
        if not result:
            logger.debug(
                f"Experiment {self._experiment.id}: has_work() -> False "
                f"[evals={len(self._eval_queue)}, retries_total={len(self._retry_heap)}, "
                f"tasks={len(self._task_queue)}, in_flight={len(self._in_flight)}, "
                f"db_exhausted={self._task_db_exhausted}]"
            )
        return result

    def _has_ready_retries(self) -> bool:
        """Check if any retries are ready."""
        if not self._retry_heap:
            return False
        return self._retry_heap[0].ready_at <= datetime.now(timezone.utc)

    async def try_get_ready_job(self) -> Job | None:
        """
        Return a Job if work is available AND rate limit allows, else None.

        Ensures the task buffer is filled from DB, peeks at the next job,
        checks rate limit using its key, then pops if allowed.
        """
        await self._ensure_task_buffer()
        if not self._active:
            logger.debug(
                f"Experiment {self._experiment.id}: try_get_ready_job() -> None (inactive)"
            )
            return None

        if not self._has_pending_work():
            logger.debug(
                f"Experiment {self._experiment.id}: try_get_ready_job() -> None (no pending work)"
            )
            return None

        # Peek at next job without popping
        job, pop = self._peek_next_job()
        if job is None:
            logger.debug(
                f"Experiment {self._experiment.id}: try_get_ready_job() -> None "
                f"(peek returned None despite has_pending_work=True)"
            )
            return None

        # Check rate limit using job's key
        key = job.get_rate_limit_key()
        bucket = self._token_buckets[key]
        try:
            bucket.make_request_if_ready()
        except UnavailableTokensError:
            # Rate limited - return None, Daemon tries another experiment
            logger.debug(
                f"Experiment {self._experiment.id}: try_get_ready_job() -> None "
                f"(rate limited for key={key}, job={job.debug_identifier})"
            )
            return None

        # Allowed - pop and return (update fairness so we're served last next time)
        self.last_served_at = datetime.now(timezone.utc)
        pop()
        return job

    def _peek_next_job(self) -> tuple[Job | None, Callable[[], Any]]:
        """Peek at the next job in priority order without removing it."""
        # Priority 1: Evals
        if self._eval_queue:
            return self._eval_queue[0], lambda: self._eval_queue.popleft()

        # Priority 2: Ready retries
        if self._has_ready_retries():
            return self._retry_heap[0].job, lambda: heapq.heappop(self._retry_heap)

        # Priority 3: Tasks
        if self._task_queue:
            return self._task_queue[0], lambda: self._task_queue.popleft()

        return None, lambda: None

    def register_cancel_scope(self, job: Job, scope: anyio.CancelScope) -> None:
        """Register a cancel scope for an in-flight job. Called by Runner when execution starts."""
        self._cancel_scopes[id(job)] = scope
        self._in_flight[id(job)] = job

    def unregister_cancel_scope(self, job: Job) -> None:
        """Unregister a cancel scope when job completes. Called by Runner when execution ends."""
        self._cancel_scopes.pop(id(job), None)
        self._in_flight.pop(id(job), None)
        self._check_completion()

    def _has_pending_work(self) -> bool:
        """Check if any work is pending (ignoring rate limits)."""
        has_evals = bool(self._eval_queue)
        has_ready_retries = self._has_ready_retries()
        has_tasks = bool(self._task_queue)
        has_more_in_db = not self._task_db_exhausted
        result = has_evals or has_ready_retries or has_tasks or has_more_in_db

        # Log detailed state when we have no pending work (helps debug stalls)
        if not result:
            logger.debug(
                f"Experiment {self._experiment.id}: _has_pending_work() -> False "
                f"[evals={len(self._eval_queue)}, retries={len(self._retry_heap)}, "
                f"tasks={len(self._task_queue)}, db_exhausted={self._task_db_exhausted}, "
                f"in_flight={len(self._in_flight)}]"
            )
        return result

    async def _ensure_task_buffer(self) -> None:
        """Load more tasks from DB if buffer is empty to avoid memory exhaustion."""
        # Early exit if no task config (nothing to dispatch)
        task_config = self._execution_config.task_config
        if task_config is None:
            self._task_db_exhausted = True
            return
        if self._task_queue:
            logger.debug(
                f"Experiment {self._experiment.id}: _ensure_task_buffer() skipped "
                f"(task_queue has {len(self._task_queue)} items)"
            )
            return
        if self._task_db_exhausted:
            logger.debug(
                f"Experiment {self._experiment.id}: _ensure_task_buffer() skipped "
                f"(db_exhausted=True, cursor={self._task_db_offset})"
            )
            return

        logger.debug(
            f"Experiment {self._experiment.id}: _ensure_task_buffer() loading batch "
            f"(cursor={self._task_db_offset}, batch_size={self._task_batch_size})"
        )

        try:
            with anyio.fail_after(30, shield=True):
                async with self._db() as session:
                    # Determine SQL dialect
                    assert session.bind is not None
                    dialect = SupportedSQLDialect(session.bind.dialect.name)

                    # Get project_id from cache or DB
                    if self._project_id is None:
                        self._project_id = await session.scalar(
                            select(models.Project.id).where(
                                models.Project.name == self._experiment.project_name
                            )
                        )
                        if self._project_id is None:
                            logger.error(
                                f"Project '{self._experiment.project_name}' "
                                "not found for experiment "
                                f"{self._experiment.id}"
                            )
                            self._task_db_exhausted = True
                            return
                    project_id = self._project_id

                    # Query incomplete runs with pagination
                    stmt = get_experiment_incomplete_runs_query(
                        self._experiment,
                        dialect,
                        cursor_example_rowid=self._task_db_offset
                        if self._task_db_offset > 0
                        else None,
                        limit=self._task_batch_size,
                    )
                    result = await session.execute(stmt)
                    rows = result.all()
        except SQLAlchemyError as e:
            # DB error - log and return without marking exhausted so we retry next cycle
            logger.warning(
                f"Experiment {self._experiment.id}: _ensure_task_buffer() DB error, "
                f"will retry next cycle: {e}"
            )
            return

        # Check if we've exhausted all tasks
        has_more = len(rows) > self._task_batch_size
        logger.debug(
            f"Experiment {self._experiment.id}: _ensure_task_buffer() query returned "
            f"{len(rows)} rows (batch_size={self._task_batch_size}, has_more={has_more})"
        )
        if has_more:
            # Cursor points to first item of NEXT page (the extra row we fetched)
            next_page_first_item = rows[self._task_batch_size][0]
            next_cursor = next_page_first_item.dataset_example_id
            rows = rows[: self._task_batch_size]
        else:
            next_cursor = None
            logger.info(
                f"Experiment {self._experiment.id}: setting _task_db_exhausted=True "
                f"(rows={len(rows)}, cursor={self._task_db_offset})"
            )
            self._task_db_exhausted = True

        if not rows:
            logger.debug(f"Experiment {self._experiment.id}: no rows returned, returning early")
            # Check completion: experiment may be done if DB exhausted and no tasks in queue
            self._check_completion()
            return

        # Update cursor for next batch
        if next_cursor is not None:
            old_cursor = self._task_db_offset
            self._task_db_offset = next_cursor
            logger.debug(
                f"Experiment {self._experiment.id}: cursor advanced {old_cursor} -> "
                f"{self._task_db_offset}"
            )

        # Process each incomplete run
        for revision, successful_count, incomplete_reps in rows:
            # Parse incomplete repetitions
            # (SQLite returns JSON string, PostgreSQL returns array)
            if successful_count == 0:
                # Completely missing - need all repetitions
                incomplete = list(range(1, self._experiment.repetitions + 1))
            elif dialect is SupportedSQLDialect.POSTGRESQL:
                incomplete = [r for r in incomplete_reps if r is not None]
            else:
                incomplete = [r for r in json.loads(incomplete_reps) if r is not None]

            # Create a TaskJob for each incomplete repetition
            for repetition_number in incomplete:
                job = self._create_task_job(
                    dataset_example_revision=revision,
                    repetition_number=repetition_number,
                    task_config=task_config,
                    project_id=project_id,
                )
                self._task_queue.append(job)

        logger.debug(f"Loaded {len(self._task_queue)} tasks for experiment {self._experiment.id}")

    def _create_task_job(
        self,
        dataset_example_revision: models.DatasetExampleRevision,
        repetition_number: int,
        task_config: TaskConfig,
        project_id: int,
    ) -> TaskJob:
        """Create a TaskJob owned by this experiment."""
        return TaskJob(
            running_experiment=self,
            experiment=self._experiment,
            dataset_example_revision=dataset_example_revision,
            repetition_number=repetition_number,
            task_config=task_config,
            llm_client=self._llm_client,
            db=self._db,
            decrypt=self._decrypt,
            tracer_factory=self._tracer_factory,
            project_id=project_id,
            credentials=self._credentials,
        )

    def _create_eval_job(
        self,
        experiment_run: models.ExperimentRun,
        dataset_example_revision: models.DatasetExampleRevision,
        evaluator_config: EvaluatorConfig,
        project_id: int,
    ) -> EvalJob | None:
        """Create an EvalJob owned by this experiment.

        Returns None if the evaluator is not available (e.g. not in evaluator_by_id).
        """
        evaluator = self._evaluator_by_id.get(evaluator_config.dataset_evaluator_id)
        if evaluator is None:
            logger.warning(
                "Evaluator id=%s not found, skipping",
                evaluator_config.dataset_evaluator_id,
            )
            return None

        return EvalJob(
            running_experiment=self,
            experiment_run=experiment_run,
            dataset_example_revision=dataset_example_revision,
            evaluator=evaluator,
            input_mapping=evaluator_config.input_mapping,
            output_configs=evaluator_config.output_configs,
            db=self._db,
            tracer_factory=self._tracer_factory,
            project_id=project_id,
        )

    # === Task Event Handlers ===

    async def on_task_chunk(self, chunk: ChatCompletionSubscriptionPayload) -> None:
        """Stream chunk to UI subscribers."""
        if not self._subscribers:
            return

        closed_streams: list[MemoryObjectSendStream[ChatCompletionSubscriptionPayload]] = []
        for send_stream in self._subscribers:
            try:
                send_stream.send_nowait(chunk)
            except anyio.WouldBlock:
                # Subscriber buffer full, drop chunk
                pass
            except (anyio.ClosedResourceError, anyio.BrokenResourceError):
                # ClosedResourceError: subscriber explicitly closed their receiving stream
                # BrokenResourceError: subscribers receive stream was garbage collected
                closed_streams.append(send_stream)
        # Clean up closed streams
        for stream in closed_streams:
            self._subscribers.remove(stream)

    async def on_task_success(
        self,
        job: TaskJob,
        experiment_run: models.ExperimentRun,
    ) -> None:
        """Task completed. Queue eval jobs for each evaluator (feedback loop)."""
        self._tasks_succeeded += 1
        self._task_circuit_breaker.record_success()

        # Discard results if experiment was stopped - avoid writing stale data
        if not self._active:
            return

        # Feedback loop: queue eval jobs for this task's result
        evaluator_configs = self._execution_config.evaluator_configs
        if evaluator_configs is None:
            return

        queued = 0
        for eval_config in evaluator_configs.evaluators:
            eval_job = self._create_eval_job(
                experiment_run=experiment_run,
                dataset_example_revision=job.dataset_example_revision,
                evaluator_config=eval_config,
                project_id=job._project_id,
            )
            if eval_job is not None:
                self._eval_queue.append(eval_job)
                queued += 1
        if queued:
            logger.debug(
                f"Experiment {self._experiment.id}: task success run_id="
                f"{experiment_run.id}, queued {queued} eval job(s)"
            )

    async def on_task_rate_limit(self, job: TaskJob) -> None:
        """Task hit rate limit. Update token bucket and requeue with backoff."""
        # Inform token bucket of rate limit
        key = job.get_rate_limit_key()
        bucket = self._token_buckets[key]
        bucket.on_rate_limit_error(
            request_start_time=datetime.now(timezone.utc).timestamp(), verbose=False
        )

        # Requeue with exponential backoff
        if job.retry_count < self._max_retries:
            job.retry_count += 1
            backoff = self._base_backoff_seconds * (2 ** (job.retry_count - 1))
            ready_at = datetime.now(timezone.utc) + timedelta(seconds=backoff)
            logger.debug(
                f"{job.debug_identifier} hit rate limit, retry "
                f"{job.retry_count}/{self._max_retries} "
                f"in {backoff:.1f}s (ready_at={ready_at.isoformat()})"
            )
            heapq.heappush(self._retry_heap, RetryItem(ready_at=ready_at, job=job))
        else:
            self._tasks_failed += 1
            logger.warning(f"{job.debug_identifier} exceeded max retries ({self._max_retries})")
            # Yield error to subscribers
            example_id = GlobalID(
                DatasetExample.__name__,
                str(job.dataset_example_revision.dataset_example_id),
            )
            error_payload = ChatCompletionSubscriptionError(
                message=f"Rate limit exceeded after {self._max_retries} retries",
                dataset_example_id=example_id,
                repetition_number=job.repetition_number,
            )
            await self.on_task_chunk(error_payload)

    async def on_task_network_error(self, job: TaskJob, error: Exception) -> None:
        """Task hit transient/network error. Requeue with backoff (no token bucket update)."""
        # Check circuit breaker
        if self._task_circuit_breaker.record_failure(error):
            await self._handle_circuit_trip(
                "task", self._task_circuit_breaker.trip_reason or str(error)
            )
            return

        # Requeue with exponential backoff (but don't update token bucket - not a rate issue)
        if job.retry_count < self._max_retries:
            job.retry_count += 1
            backoff = self._base_backoff_seconds * (2 ** (job.retry_count - 1))
            ready_at = datetime.now(timezone.utc) + timedelta(seconds=backoff)
            heapq.heappush(self._retry_heap, RetryItem(ready_at=ready_at, job=job))
        else:
            self._tasks_failed += 1
            logger.warning(
                f"{job.debug_identifier} exceeded max retries "
                f"({self._max_retries}) due to network errors"
            )
            # Yield error to subscribers
            example_id = GlobalID(
                DatasetExample.__name__,
                str(job.dataset_example_revision.dataset_example_id),
            )
            error_payload = ChatCompletionSubscriptionError(
                message=f"Network error after {self._max_retries} retries: {error}",
                dataset_example_id=example_id,
                repetition_number=job.repetition_number,
            )
            await self.on_task_chunk(error_payload)

    async def on_task_failure(self, job: TaskJob, error: Exception) -> None:
        """Task failed with non-retryable error. Yield error to subscribers."""
        self._tasks_failed += 1
        logger.warning(f"{job.debug_identifier} failed with non-retryable error: {error}")

        # Check circuit breaker (permanent failures also count)
        if self._task_circuit_breaker.record_failure(error):
            await self._handle_circuit_trip(
                "task", self._task_circuit_breaker.trip_reason or str(error)
            )
            return

        # Yield error to subscribers (same as old implementation)
        example_id = GlobalID(
            DatasetExample.__name__,
            str(job.dataset_example_revision.dataset_example_id),
        )
        error_payload = ChatCompletionSubscriptionError(
            message=str(error),
            dataset_example_id=example_id,
            repetition_number=job.repetition_number,
        )
        await self.on_task_chunk(error_payload)

    async def on_task_timeout(self, job: TaskJob) -> None:
        """Task timed out. Requeue with backoff or yield error if max retries exceeded."""
        # Timeouts are transient - requeue with backoff
        if job.retry_count < self._max_retries:
            job.retry_count += 1
            backoff = self._base_backoff_seconds * (2 ** (job.retry_count - 1))
            ready_at = datetime.now(timezone.utc) + timedelta(seconds=backoff)
            heapq.heappush(self._retry_heap, RetryItem(ready_at=ready_at, job=job))
        else:
            self._tasks_failed += 1
            logger.warning(
                f"{job.debug_identifier} exceeded max retries ({self._max_retries}) due to timeouts"
            )
            # Yield error to subscribers (same as old implementation)
            example_id = GlobalID(
                DatasetExample.__name__,
                str(job.dataset_example_revision.dataset_example_id),
            )
            error_payload = ChatCompletionSubscriptionError(
                message="Playground task timed out",
                dataset_example_id=example_id,
                repetition_number=job.repetition_number,
            )
            await self.on_task_chunk(error_payload)

    # === Eval Event Handlers ===

    async def on_eval_success(
        self,
        job: EvalJob,
        annotations: Sequence[models.ExperimentRunAnnotation],
        db_traces: Sequence[models.Trace] = (),
    ) -> None:
        """Eval completed - yield result to UI."""
        self._evals_succeeded += 1
        self._eval_circuit_breaker.record_success()

        # Discard results if experiment was stopped - avoid writing stale data
        if not self._active:
            return

        traces_by_trace_id = {t.trace_id: t for t in db_traces}
        example_id = GlobalID(
            DatasetExample.__name__, str(job.dataset_example_revision.dataset_example_id)
        )
        closed_streams: list[MemoryObjectSendStream["ChatCompletionSubscriptionPayload"]] = []
        for send_stream in self._subscribers:
            for annotation in annotations:
                eval_db_trace = (
                    traces_by_trace_id.get(annotation.trace_id) if annotation.trace_id else None
                )
                eval_chunk = EvaluationChunk(
                    evaluator_name=job.evaluator.name,
                    experiment_run_evaluation=(
                        ExperimentRunAnnotation(id=annotation.id, db_record=annotation)
                        if not annotation.error
                        else None
                    ),
                    dataset_example_id=example_id,
                    repetition_number=job.experiment_run.repetition_number,
                    trace=(
                        Trace(id=eval_db_trace.id, db_record=eval_db_trace)
                        if eval_db_trace
                        else None
                    ),
                    error=annotation.error,
                )

                try:
                    send_stream.send_nowait(eval_chunk)
                except anyio.WouldBlock:
                    logger.debug(
                        f"Experiment {self._experiment.id}: subscriber slow, dropping chunk"
                    )
                    pass
                except (anyio.ClosedResourceError, anyio.BrokenResourceError):
                    # ClosedResourceError: subscriber explicitly closed their receive stream
                    # BrokenResourceError: subscriber's receive stream was garbage collected
                    closed_streams.append(send_stream)
        # Clean up closed streams
        for stream in closed_streams:
            self._subscribers.remove(stream)

    async def on_eval_rate_limit(self, job: EvalJob) -> None:
        """Eval hit rate limit. Requeue with backoff."""
        key = job.get_rate_limit_key()
        bucket = self._token_buckets[key]
        bucket.on_rate_limit_error(
            request_start_time=datetime.now(timezone.utc).timestamp(), verbose=False
        )

        if job.retry_count < self._max_retries:
            job.retry_count += 1
            backoff = self._base_backoff_seconds * (2 ** (job.retry_count - 1))
            ready_at = datetime.now(timezone.utc) + timedelta(seconds=backoff)
            logger.debug(
                f"{job.debug_identifier} hit rate limit, retry "
                f"{job.retry_count}/{self._max_retries} "
                f"in {backoff:.1f}s"
            )
            heapq.heappush(self._retry_heap, RetryItem(ready_at=ready_at, job=job))
        else:
            self._evals_failed += 1
            logger.warning(f"{job.debug_identifier} exceeded max retries ({self._max_retries})")

    async def on_eval_network_error(self, job: EvalJob, error: Exception) -> None:
        """Eval hit transient/network error. Requeue with backoff (no token bucket update)."""
        # Check circuit breaker
        if self._eval_circuit_breaker.record_failure(error):
            await self._handle_circuit_trip(
                "eval", self._eval_circuit_breaker.trip_reason or str(error)
            )
            return

        # Requeue with exponential backoff (but don't update token bucket - not a rate issue)
        if job.retry_count < self._max_retries:
            job.retry_count += 1
            backoff = self._base_backoff_seconds * (2 ** (job.retry_count - 1))
            ready_at = datetime.now(timezone.utc) + timedelta(seconds=backoff)
            heapq.heappush(self._retry_heap, RetryItem(ready_at=ready_at, job=job))
        else:
            self._evals_failed += 1
            logger.warning(
                f"{job.debug_identifier} exceeded max retries "
                f"({self._max_retries}) due to network errors"
            )

    async def on_eval_failure(self, job: EvalJob, error: Exception) -> None:
        """Eval failed with non-retryable error."""
        self._evals_failed += 1
        logger.warning(f"{job.debug_identifier} failed with non-retryable error: {error}")

        # Check circuit breaker (permanent failures also count)
        if self._eval_circuit_breaker.record_failure(error):
            await self._handle_circuit_trip(
                "eval", self._eval_circuit_breaker.trip_reason or str(error)
            )
            return

    async def on_eval_timeout(self, job: EvalJob) -> None:
        """Eval timed out. Treat as transient - requeue with backoff."""
        # Timeouts are transient - requeue with backoff
        if job.retry_count < self._max_retries:
            job.retry_count += 1
            backoff = self._base_backoff_seconds * (2 ** (job.retry_count - 1))
            ready_at = datetime.now(timezone.utc) + timedelta(seconds=backoff)
            heapq.heappush(self._retry_heap, RetryItem(ready_at=ready_at, job=job))
        else:
            self._evals_failed += 1
            logger.warning(
                f"{job.debug_identifier} exceeded max retries ({self._max_retries}) due to timeouts"
            )

    # === Circuit Breaker & Completion ===

    async def _handle_circuit_trip(self, job_type: str, reason: str) -> None:
        """
        Handle circuit breaker trip - stop experiment and notify subscribers.

        Args:
            job_type: "task" or "eval" - which circuit breaker tripped
            reason: The error message that caused the trip
        """
        logger.warning(
            f"Experiment {self._experiment.id}: circuit breaker tripped ({job_type}): {reason}"
        )

        # Send error to all subscribers before cancelling
        error_payload = ChatCompletionSubscriptionError(
            message=f"Experiment stopped: {reason} (5 consecutive failures)",
            dataset_example_id=None,  # Experiment-level error, not task-specific
            repetition_number=None,
        )

        for send_stream in self._subscribers:
            try:
                send_stream.send_nowait(error_payload)
                logger.debug(
                    f"Experiment {self._experiment.id}: sent circuit trip error to subscriber"
                )
            except anyio.WouldBlock:
                logger.warning(
                    f"Experiment {self._experiment.id}: subscriber buffer full, dropping error"
                )
            except (anyio.ClosedResourceError, anyio.BrokenResourceError):
                logger.debug(f"Experiment {self._experiment.id}: subscriber already disconnected")

        # Close all subscriber streams so they receive EndOfStream
        logger.debug(
            f"Experiment {self._experiment.id}: "
            f"closing {len(self._subscribers)} subscriber stream(s)"
        )
        for send_stream in self._subscribers:
            try:
                await send_stream.aclose()
                logger.debug(f"Experiment {self._experiment.id}: closed subscriber stream")
            except Exception as e:
                logger.debug(f"Experiment {self._experiment.id}: error closing stream: {e}")
        self._subscribers.clear()
        logger.debug(f"Experiment {self._experiment.id}: all subscriber streams closed")

        # Stop the experiment (stops in-flight jobs, clears queues)
        self.stop()

        # Notify daemon to remove us and update DB status (with error)
        error_message = f"Circuit breaker tripped ({job_type}): {reason}"
        self._on_done(self._experiment.id, last_error=error_message)

    def _check_completion(self) -> None:
        """Check if experiment is complete (no pending work, nothing in-flight)."""
        if not self._active:
            logger.debug(f"Experiment {self._experiment.id}: _check_completion() skip (inactive)")
            return

        # Log current state for debugging
        has_work_result = self.has_work()
        has_more_db = not self._task_db_exhausted
        logger.debug(
            f"Experiment {self._experiment.id}: _check_completion() "
            f"has_work={has_work_result}, has_more_db={has_more_db}"
        )

        # Check both current work and potential work from DB pagination
        if not has_work_result and not has_more_db:
            self._active = False
            # Close subscriber streams so consumers get EndOfStream
            # (This is the canonical signal that the producer is done)
            for stream in self._subscribers:
                stream.close()
            self._subscribers.clear()
            logger.info(
                f"Experiment {self._experiment.id} completed: "
                f"tasks={self._tasks_succeeded} succeeded, {self._tasks_failed} failed; "
                f"evals={self._evals_succeeded} succeeded, {self._evals_failed} failed"
            )
            self._on_done(self._experiment.id)

    def stop(self) -> None:
        """
        Stop this experiment.

        NOTE: Does NOT call on_done. The caller (stop mutation, heartbeat, etc.)
        is responsible for handling status updates. on_done is only for natural
        completion via _check_completion().
        """
        # Idempotency guard: if already stopped, no-op
        if not self._active:
            return

        # Log stack trace to identify what triggered stop (helps debug mysterious stops)
        stack = "".join(traceback.format_stack()[:-1])  # Exclude this frame
        logger.info(f"Experiment {self._experiment.id} stop() called from:\n{stack}")

        self._active = False
        pending_tasks = len(self._task_queue)
        pending_evals = len(self._eval_queue)
        pending_retries = len(self._retry_heap)
        in_flight = len(self._in_flight)

        # Cancel all in-flight jobs via their scopes
        cancelled_count = 0
        for scope in self._cancel_scopes.values():
            scope.cancel()
            cancelled_count += 1

        # Clear all queues to release memory and prevent any further processing
        # (Queued work is transient - derived from DB state, reconstructed on resume)
        self._task_queue.clear()
        self._eval_queue.clear()
        self._retry_heap.clear()

        # Close all subscriber streams so they receive EndOfStream
        # (MemoryObjectSendStream supports sync close())
        subscriber_count = len(self._subscribers)
        for stream in self._subscribers:
            stream.close()
        self._subscribers.clear()

        logger.info(
            f"Experiment {self._experiment.id} stopped: "
            f"tasks={self._tasks_succeeded} succeeded, {self._tasks_failed} failed; "
            f"evals={self._evals_succeeded} succeeded, {self._evals_failed} failed; "
            f"dropped: {pending_tasks} tasks, {pending_evals} evals, {pending_retries} retries, "
            f"{in_flight} in-flight ({cancelled_count} cancelled), "
            f"{subscriber_count} subscribers closed"
        )
        # NOTE: on_done is NOT called here. See docstring.

    # === Subscription ===

    def subscribe(
        self,
    ) -> MemoryObjectReceiveStream[ChatCompletionSubscriptionPayload]:
        """Subscribe to experiment progress updates.

        Returns a receive stream. Close it when done; cleanup happens automatically.
        """
        send_stream, receive_stream = anyio.create_memory_object_stream[
            ChatCompletionSubscriptionPayload
        ](max_buffer_size=1000)
        self._subscribers.append(send_stream)
        logger.debug(
            f"Experiment {self._experiment.id}: new subscriber, total={len(self._subscribers)}"
        )
        return receive_stream


# =============================================================================
# Daemon (Singleton Orchestrator)
# =============================================================================


class ExperimentRunner(DaemonTask):
    """
    Singleton daemon that orchestrates background experiment execution.

    Key patterns:
    - Semaphore-first dispatch: acquire slot before looking for work
    - Round-robin fairness: least-recently-served experiment gets priority
    - Non-blocking rate limits: experiments return None if rate limited
    - Auto-resume: orphaned experiments resumed on startup
    """

    MAX_CONCURRENT = 20
    POLL_INTERVAL = 0.1  # seconds
    MAX_CONSECUTIVE_ERRORS = 50  # Stop daemon after this many consecutive internal errors

    # Timeout for detecting stale claims from crashed replicas
    STALE_CLAIM_TIMEOUT = EXPERIMENT_STALE_CLAIM_TIMEOUT
    HEARTBEAT_INTERVAL = EXPERIMENT_STALE_CLAIM_TIMEOUT / 2

    # Orphan scan: check for crashed replicas' experiments periodically
    # Base interval + random jitter to avoid thundering herd across replicas
    ORPHAN_SCAN_INTERVAL = EXPERIMENT_STALE_CLAIM_TIMEOUT
    ORPHAN_SCAN_JITTER = timedelta(seconds=EXPERIMENT_STALE_CLAIM_TIMEOUT.seconds * 0.1)

    def __init__(
        self,
        db: DbSessionFactory,
        *,
        decrypt: Callable[[bytes], bytes],
        tracer_factory: Callable[[], Tracer],
    ) -> None:
        super().__init__()
        self._db = db
        self._decrypt = decrypt
        self._tracer_factory = tracer_factory
        self._experiments: dict[ExperimentId, RunningExperiment] = {}
        self._seats = Semaphore(self.MAX_CONCURRENT)
        self._work_available = anyio.Event()
        # Unique replica ID for coordinating experiment ownership across replicas
        self._replica_id = token_hex(8)
        # Rate limit buckets keyed by client's rate_limit_key.
        self._token_buckets: TokenBucketRegistry = AutoCreateTokenBucketRegistry(maxsize=100)
        # Cancel scope for forceful shutdown - set during _run()
        self._task_group_cancel_scope: anyio.CancelScope | None = None
        # Prevent GC from silently cancelling fire-and-forget tasks
        self._background_tasks: set[asyncio.Task[None]] = set()

    async def _run(self) -> None:
        """
        Main dispatch loop (semaphore-first pattern).

        1. Wait for experiments if none exist
        2. Acquire semaphore slot (wait if all busy)
        3. Round-robin through experiments for ready job
        4. If job found, dispatch; else release slot and sleep
        """
        logger.debug(f"ExperimentRunner daemon starting (replica_id={self._replica_id})")

        async with anyio.create_task_group() as tg:
            # Save cancel scope so graceful_shutdown can force-cancel if needed
            self._task_group_cancel_scope = tg.cancel_scope
            # Start background loops
            tg.start_soon(self._heartbeat_loop)
            tg.start_soon(self._orphan_scan_loop)
            # Track consecutive errors to detect programming bugs (e.g., AttributeError in a loop).
            # Without this, internal errors would spin infinitely. After MAX_CONSECUTIVE_ERRORS,
            # we stop the daemon rather than burn CPU and fill logs.
            consecutive_errors = 0
            try:
                while self._running:
                    acquired = False
                    try:
                        # Wait for experiments if none exist
                        if not self._experiments:
                            logger.debug("No experiments, waiting for work...")
                            self._work_available = anyio.Event()
                            await self._work_available.wait()
                            logger.debug("Work available, resuming dispatch loop")
                            continue

                        # Semaphore-first: acquire slot before looking for work
                        await self._seats.acquire()
                        acquired = True
                        logger.debug(
                            "Dispatch loop: semaphore acquired, "
                            f"active_experiments={len(self._experiments)}"
                        )

                        # Round-robin through experiments for fairness
                        job = await self._try_get_ready_job()

                        if job:
                            logger.debug(f"Dispatching job: {job.debug_identifier}")
                            tg.start_soon(self._run_and_release, job)
                            acquired = False  # Ownership transferred to job
                        else:
                            self._seats.release()
                            acquired = False
                            logger.debug(
                                "Dispatch loop: no ready job, sleeping %.2fs",
                                self.POLL_INTERVAL,
                            )
                            await anyio.sleep(self.POLL_INTERVAL)

                        consecutive_errors = 0
                    except anyio.get_cancelled_exc_class():
                        raise
                    except Exception:
                        consecutive_errors += 1
                        if acquired:
                            try:
                                self._seats.release()
                            except ValueError:
                                pass

                        logger.exception(
                            f"Dispatch loop error "
                            f"({consecutive_errors}/{self.MAX_CONSECUTIVE_ERRORS})"
                        )

                        if consecutive_errors >= self.MAX_CONSECUTIVE_ERRORS:
                            logger.critical(
                                f"Dispatch loop hit {self.MAX_CONSECUTIVE_ERRORS} "
                                "consecutive errors, stopping daemon. "
                                "This is likely a programming bug."
                            )
                            self._running = False
                            break

                        # Exponential backoff: 0.1s, 0.2s, 0.4s, ... capped at 30s
                        backoff = min(0.1 * (2**consecutive_errors), 30.0)
                        await anyio.sleep(backoff)
            except anyio.get_cancelled_exc_class():
                logger.debug("Dispatch loop cancelled")
                raise
            finally:
                # On shutdown, wait for in-flight jobs to complete
                logger.debug(
                    f"Dispatch loop ending (_running={self._running}), starting graceful shutdown"
                )
                await self._graceful_shutdown()

    async def _try_get_ready_job(self) -> Job | None:
        """
        Try to get a ready job from any experiment (round-robin for fairness).

        Sorts experiments by last_served_at so least-recently-served gets priority.
        Each experiment's try_get_ready_job() checks rate limit non-blocking.
        """
        # Sort by fairness: least recently served first
        candidates = sorted(
            self._experiments.values(),
            key=lambda e: e.last_served_at,
        )

        if not candidates:
            logger.debug("_try_get_ready_job: no experiments in registry")
            return None

        for candidate in candidates:
            if job := await candidate.try_get_ready_job():
                return job

        # Log why we couldn't find work (helps debug stalled experiments)
        states = [
            (
                e._experiment.id,
                e._active,
                len(e._task_queue),
                len(e._eval_queue),
                len(e._in_flight),
                len(e._retry_heap),
                e._task_db_exhausted,
            )
            for e in candidates
        ]
        logger.debug(
            f"_try_get_ready_job: checked {len(candidates)} experiments, none had ready work. "
            f"States (id, active, tasks, evals, in_flight, retries, db_exhausted): {states}"
        )
        return None

    async def _run_and_release(self, job: Job) -> None:
        """Execute job and release semaphore."""
        try:
            # Create cancel scope so experiment can cancel this job
            with anyio.CancelScope() as scope:
                job.running_experiment.register_cancel_scope(job, scope)
                await job.execute()
        except anyio.get_cancelled_exc_class():
            logger.debug(f"Job {job.debug_identifier} was cancelled")
        except Exception:
            logger.exception(f"Job {job.debug_identifier} raised unhandled exception")
        finally:
            job.running_experiment.unregister_cancel_scope(job)
            logger.debug(f"Job {job.debug_identifier} finished, releasing semaphore")
            self._seats.release()

    async def _graceful_shutdown(self, timeout: float = 5.0) -> None:
        """
        Gracefully shut down all experiments.

        This does in-memory cleanup only (no DB update) so experiments can resume
        on restart. Ownership is preserved in the DB via claimed_by/claimed_at.

        Steps:
        1. Stop all experiments (cancels jobs, clears queues, closes subscribers)
        2. Wait for shielded DB writes to complete (bounded by timeout)
        3. Force-cancel if timeout exceeded

        See appendix-stopping-deep-dive.md for rationale on not updating DB.
        """
        experiment_count = len(self._experiments)
        in_flight_count = sum(len(exp._in_flight) for exp in self._experiments.values())

        if experiment_count == 0:
            logger.debug("Graceful shutdown: no experiments running")
            return

        logger.info(
            f"Graceful shutdown: stopping {experiment_count} experiments "
            f"with {in_flight_count} in-flight jobs"
        )

        # Stop all experiments (in-memory only - preserves DB ownership for resume)
        for exp_id, exp in list(self._experiments.items()):
            logger.debug(f"Graceful shutdown: stopping experiment {exp_id}")
            exp.stop()
        # Don't clear _experiments here - let shielded operations reference them

        if in_flight_count == 0:
            logger.debug("Graceful shutdown: no in-flight jobs to wait for")
            self._experiments.clear()
            return

        logger.debug(f"Graceful shutdown: waiting up to {timeout}s for shielded DB writes")

        # Wait for semaphore to be fully released (all shielded operations complete)
        try:
            with anyio.fail_after(timeout):
                # Acquire all semaphore slots = all jobs have finished their shielded writes
                for _ in range(self.MAX_CONCURRENT):
                    await self._seats.acquire()
            logger.debug("Graceful shutdown: all shielded operations completed")
        except TimeoutError:
            logger.warning(f"Graceful shutdown: timed out after {timeout}s, force-cancelling")
            # Force-cancel the task group to kill httpx connections
            if self._task_group_cancel_scope:
                self._task_group_cancel_scope.cancel()

        # Clear registry after shutdown
        self._experiments.clear()

    async def _resume_orphaned(self) -> None:
        """
        Resume orphaned experiments on startup.

        Finds experiments that were running but their owner crashed:
        - claimed_at is NOT NULL (was running) AND claimed_at is stale

        This handles:
        - Server crash during experiment execution
        - Network partition where replica became unreachable
        """
        cutoff = datetime.now(timezone.utc) - self.STALE_CLAIM_TIMEOUT

        async with self._db() as session:
            # Find orphaned experiments (claimed but stale)
            stmt = (
                select(models.ExperimentExecutionConfig)
                .where(models.ExperimentExecutionConfig.claimed_at.is_not(None))
                .where(models.ExperimentExecutionConfig.claimed_at < cutoff)
            )
            result = await session.execute(stmt)
            orphaned_configs = result.scalars().all()

            if not orphaned_configs:
                logger.debug("No orphaned experiments found")
                return

            logger.info(f"Found {len(orphaned_configs)} orphaned experiments, resuming...")

            # Collect configs to start AFTER releasing session lock
            configs_to_start: list[models.ExperimentExecutionConfig] = []

            for config in orphaned_configs:
                try:
                    # Eager load experiment relationship
                    await session.refresh(config, ["experiment"])

                    # Try to claim it (atomic to prevent race with other replicas)
                    # NOTE: Don't set toggled_at - orphan resume is automatic, not user-initiated.
                    # User should be able to stop immediately after orphan recovery.
                    now = datetime.now(timezone.utc)
                    claim_stmt = (
                        update(models.ExperimentExecutionConfig)
                        .where(models.ExperimentExecutionConfig.id == config.id)
                        .where(models.ExperimentExecutionConfig.claimed_at < cutoff)
                        .values(
                            claimed_at=now,
                            claimed_by=self._replica_id,
                            last_error=None,  # Clear previous error on resume
                        )
                        .returning(models.ExperimentExecutionConfig.id)
                    )
                    claim_result = await session.execute(claim_stmt)
                    claimed_row = claim_result.scalar_one_or_none()

                    if claimed_row is None:
                        # Another replica claimed it first
                        logger.debug(f"Experiment {config.id} claimed by another replica")
                        continue

                    # Mark for starting after session releases
                    configs_to_start.append(config)

                except Exception:
                    logger.exception(f"Failed to claim orphaned experiment {config.id}")

        # Start experiments OUTSIDE session to avoid deadlock
        # (start_experiment also updates the row, needs separate transaction)
        for config in configs_to_start:
            try:
                await self.start_experiment(config, subscribe=False)
                logger.info(f"Resumed orphaned experiment {config.id}")
            except Exception:
                logger.exception(f"Failed to start orphaned experiment {config.id}")

    # === Public API ===

    @overload
    async def start_experiment(
        self,
        config: models.ExperimentExecutionConfig,
        *,
        credentials: Sequence[GenerativeCredentialInput] | None = None,
        subscribe: Literal[True] = True,
    ) -> tuple[RunningExperiment, MemoryObjectReceiveStream[ChatCompletionSubscriptionPayload]]: ...

    @overload
    async def start_experiment(
        self,
        config: models.ExperimentExecutionConfig,
        *,
        credentials: Sequence[GenerativeCredentialInput] | None = None,
        subscribe: Literal[False],
    ) -> RunningExperiment: ...

    async def start_experiment(
        self,
        config: models.ExperimentExecutionConfig,
        *,
        credentials: Sequence[GenerativeCredentialInput] | None = None,
        subscribe: bool = False,
    ) -> (
        tuple[RunningExperiment, MemoryObjectReceiveStream[ChatCompletionSubscriptionPayload]]
        | RunningExperiment
    ):
        """Register and start a new experiment.

        Args:
            config: The experiment execution config.
            credentials: Ephemeral API credentials (not stored, passed at runtime).
            subscribe: If True (default), returns a subscription stream to receive chunks.
                       Subscribe before work starts to avoid missing early chunks.

        Returns:
            If subscribe=True: (experiment, receive_stream) tuple
            If subscribe=False: just the experiment
        """
        logger.info(
            f"start_experiment({config.id}) called, "
            f"subscribe={subscribe}, has_credentials={credentials is not None}"
        )

        task_config = config.task_config
        if task_config is None:
            raise ValueError(
                "ExperimentExecutionConfig.task_config is required to start experiment"
            )
        pv = task_config.prompt_version_config
        model_provider = pv.model_provider
        model_name = pv.model_name
        custom_provider_id = pv.custom_provider_id

        # Mark this replica as owner (claimed_at NOT NULL = running)
        # Clear last_error on start/resume (fresh start)
        # NOTE: Don't set toggled_at here - it's only set by user-initiated stop/resume
        # mutations to enforce cooldown between toggles. Initial start shouldn't block
        # the first stop.
        now = datetime.now(timezone.utc)
        async with self._db() as session:
            stmt = (
                update(models.ExperimentExecutionConfig)
                .where(models.ExperimentExecutionConfig.id == config.id)
                .values(
                    claimed_at=now,
                    claimed_by=self._replica_id,
                    last_error=None,  # Clear previous error on resume
                )
            )
            await session.execute(stmt)
            experiment = await session.get(
                models.Experiment,
                config.id,
            )
            if experiment is None:
                # Race condition: experiment was deleted after config was created
                raise ValueError(f"Experiment {config.id} no longer exists")

            llm_client = await get_playground_client(
                model_provider=model_provider,
                model_name=model_name,
                custom_provider_id=custom_provider_id,
                session=session,
                decrypt=self._decrypt,
                credentials=credentials,
            )

            # Load evaluators from config
            # get_evaluators expects DatasetEvaluator node IDs
            # (it resolves to LLM/Builtin internally)
            evaluator_node_ids: list[GlobalID] = []
            evaluator_configs = config.evaluator_configs
            dataset_evaluator_name_by_id: dict[int, str] = {}
            if evaluator_configs is None:
                evaluator_configs_list = []
            else:
                evaluator_configs_list = list(evaluator_configs.evaluators)

            for eval_config in evaluator_configs_list:
                evaluator_node_ids.append(
                    GlobalID(
                        type_name=DatasetEvaluatorNode.__name__,
                        node_id=str(eval_config.dataset_evaluator_id),
                    )
                )

            # Load dataset evaluator names for display (prefer over base evaluator name)
            if evaluator_configs_list:
                dataset_evaluator_ids = [c.dataset_evaluator_id for c in evaluator_configs_list]
                names_result = await session.execute(
                    select(
                        models.DatasetEvaluators.id,
                        models.DatasetEvaluators.name,
                    ).where(models.DatasetEvaluators.id.in_(dataset_evaluator_ids))
                )
                for row in names_result:
                    dataset_evaluator_name_by_id[row.id] = row.name.root

            evaluators = await get_evaluators(
                dataset_evaluator_node_ids=evaluator_node_ids,
                session=session,
                decrypt=self._decrypt,
                credentials=credentials,
            )
            evaluator_by_id = {
                cfg.dataset_evaluator_id: ev for cfg, ev in zip(evaluator_configs_list, evaluators)
            }
            logger.debug(
                f"Loaded {len(evaluators)} evaluators for experiment {config.id} "
                f"(config has {len(evaluator_configs_list)} total)"
            )

        exp = RunningExperiment(
            experiment=experiment,
            execution_config=config,
            llm_client=llm_client,
            db=self._db,
            decrypt=self._decrypt,
            tracer_factory=self._tracer_factory,
            token_buckets=self._token_buckets,
            on_done=self._on_experiment_done,
            credentials=credentials,
            evaluator_by_id=evaluator_by_id,
            dataset_evaluator_name_by_id=dataset_evaluator_name_by_id,
        )

        # Subscribe BEFORE registering - guarantees no missed chunks
        receive_stream = exp.subscribe() if subscribe else None

        self._experiments[config.id] = exp
        self._work_available.set()  # Wake dispatch loop
        logger.info(
            f"Started experiment {config.id} (replica={self._replica_id}, "
            f"subscribe={subscribe}, total_active={len(self._experiments)})"
        )

        if subscribe:
            assert receive_stream is not None
            return exp, receive_stream
        return exp

    async def _heartbeat_loop(self) -> None:
        """
        Periodically refresh claimed_at for running experiments we own.

        Also detects lost ownership (cross-replica stop, deleted, etc.)
        and cancels local experiments we no longer own.
        """
        while True:
            await asyncio.sleep(self.HEARTBEAT_INTERVAL.total_seconds())
            if not self._experiments:
                logger.debug("Heartbeat: no experiments in memory, skipping")
                continue

            try:
                experiment_ids = list(self._experiments.keys())
                now = datetime.now(timezone.utc)
                logger.debug(
                    f"Heartbeat: refreshing {len(experiment_ids)} experiments: {experiment_ids}"
                )

                # UPDATE only experiments we own that are still running
                # RETURNING tells us which ones were actually updated
                with anyio.fail_after(30, shield=True):
                    async with self._db() as session:
                        stmt = (
                            update(models.ExperimentExecutionConfig)
                            .where(models.ExperimentExecutionConfig.id.in_(experiment_ids))
                            .where(models.ExperimentExecutionConfig.claimed_by == self._replica_id)
                            .where(models.ExperimentExecutionConfig.claimed_at.is_not(None))
                            .values(claimed_at=now)
                            .returning(models.ExperimentExecutionConfig.id)
                        )
                        result = await session.execute(stmt)
                        updated_ids = {row.id for row in result}

                logger.debug(
                    f"Heartbeat: UPDATE affected {len(updated_ids)} rows: {updated_ids} "
                    f"(WHERE id IN {experiment_ids} AND claimed_by={self._replica_id} "
                    f"AND claimed_at IS NOT NULL)"
                )

                # Any experiment in memory but NOT updated = we lost ownership
                lost_ownership = set(experiment_ids) - updated_ids
                if lost_ownership:
                    logger.warning(
                        f"Heartbeat: detected lost ownership for "
                        f"{len(lost_ownership)} experiments: "
                        f"{lost_ownership} (in memory but not updated by heartbeat)"
                    )
                for exp_id in lost_ownership:
                    logger.info(
                        f"Lost ownership of experiment {exp_id} "
                        f"(not in updated_ids), cancelling local state"
                    )
                    if exp := self._experiments.pop(exp_id, None):
                        exp.stop()

                if updated_ids:
                    logger.debug(
                        f"Heartbeat: successfully refreshed {len(updated_ids)} experiments"
                    )
            except Exception:
                logger.exception("Heartbeat failed, will retry next interval")

    async def _orphan_scan_loop(self) -> None:
        """
        Periodically scan for orphaned experiments from crashed replicas.

        Unlike startup recovery, this runs continuously to catch experiments
        orphaned by other replicas that crash while this server is running.

        Uses jitter to avoid thundering herd when multiple replicas scan simultaneously.
        """
        while True:
            # Random jitter to spread out scans across replicas
            jitter_seconds = random.uniform(0, self.ORPHAN_SCAN_JITTER.total_seconds())
            sleep_seconds = self.ORPHAN_SCAN_INTERVAL.total_seconds() + jitter_seconds
            logger.debug(
                f"Orphan scan: sleeping {sleep_seconds:.1f}s "
                f"(base={self.ORPHAN_SCAN_INTERVAL.total_seconds()}s, jitter={jitter_seconds:.1f}s)"
            )
            await asyncio.sleep(sleep_seconds)

            try:
                await self._resume_orphaned()
            except Exception:
                logger.exception("Orphan scan failed, will retry next interval")

    def stop_experiment(self, experiment_id: int) -> bool:
        """Stop a running experiment.

        Removes experiment from _experiments dict so heartbeat doesn't try to
        refresh it (which would fail since claimed_at=NULL in DB).
        """
        logger.info(
            f"stop_experiment({experiment_id}) called, "
            f"in_memory={experiment_id in self._experiments}, "
            f"all_experiments={list(self._experiments.keys())}"
        )
        if exp := self._experiments.pop(experiment_id, None):
            logger.info(f"stop_experiment({experiment_id}): found in registry, calling stop()")
            exp.stop()
            return True
        logger.warning(
            f"stop_experiment({experiment_id}): NOT FOUND in registry (already stopped/completed?)"
        )
        return False

    def _on_experiment_done(self, experiment_id: int, *, last_error: str | None = None) -> None:
        """
        Callback when experiment stops (naturally or due to error).

        Called from _check_completion() when all work is done,
        or from _handle_circuit_trip() when circuit breaker trips.
        Sets is_running=False in the database, optionally with error.
        """
        self._experiments.pop(experiment_id, None)
        logger.debug(f"Experiment {experiment_id} stopped, removed from registry")
        task = asyncio.create_task(
            self._set_experiment_stopped(experiment_id, last_error=last_error)
        )
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _set_experiment_stopped(
        self, experiment_id: int, *, last_error: str | None = None
    ) -> None:
        """
        Set experiment to stopped (claimed_at=NULL) in database.

        Uses CONDITIONAL update (WHERE claimed_by = self._replica_id) to avoid
        clobbering another replica's running experiment during ownership transitions.
        If we've lost ownership, the update affects 0 rows - which is correct.

        See appendix-stopping-deep-dive.md for detailed rationale.
        """
        # Log error locally before attempting DB update (in case conditional update fails)
        if last_error:
            logger.warning(f"Experiment {experiment_id} stopping with error: {last_error}")

        values: dict[str, Any] = {
            "claimed_at": None,
            "claimed_by": None,
        }
        if last_error is not None:
            values["last_error"] = last_error

        # CONDITIONAL update: only if we still own it
        # This prevents clobbering another replica's running experiment
        stmt = (
            update(models.ExperimentExecutionConfig)
            .where(models.ExperimentExecutionConfig.id == experiment_id)
            .where(models.ExperimentExecutionConfig.claimed_by == self._replica_id)
            .values(**values)
            .returning(models.ExperimentExecutionConfig.id)
        )
        try:
            with anyio.fail_after(30, shield=True):
                async with self._db() as session:
                    updated = await session.scalar(stmt)

            if updated:
                logger.debug(f"Set claimed_at=NULL for experiment {experiment_id}")
            else:
                # We lost ownership - another replica owns it or it was already stopped
                # The error was logged above, so debugging is still possible
                logger.debug(
                    f"Conditional update for experiment {experiment_id} affected 0 rows "
                    f"(lost ownership or already stopped)"
                )
        except Exception:
            logger.exception(f"Failed to stop experiment {experiment_id}")

    @property
    def replica_id(self) -> str:
        return self._replica_id

    def get_experiment(self, experiment_id: int) -> RunningExperiment | None:
        """Get a running experiment by ID."""
        return self._experiments.get(experiment_id)

    async def subscribe(
        self, experiment_id: int
    ) -> AsyncIterator[ChatCompletionSubscriptionPayload]:
        """
        Subscribe to experiment progress updates.

        Yields payloads as the daemon processes the experiment.
        """
        exp = self._experiments.get(experiment_id)
        if not exp:
            logger.debug(f"Daemon.subscribe({experiment_id}): experiment not found")
            return

        logger.debug(f"Daemon.subscribe({experiment_id}): starting subscription loop")
        receive_stream = exp.subscribe()
        try:
            while exp.has_work():
                with anyio.move_on_after(1.0) as cancel_scope:
                    payload = await receive_stream.receive()
                    logger.debug(
                        f"Daemon.subscribe({experiment_id}): received payload type "
                        f"{type(payload).__name__}"
                    )
                    yield payload
                if cancel_scope.cancelled_caught:
                    continue  # Timeout - check if experiment still has work
            logger.debug(f"Daemon.subscribe({experiment_id}): loop exited (has_work=False)")
        except anyio.EndOfStream:
            logger.debug(f"Daemon.subscribe({experiment_id}): EndOfStream received")
        finally:
            logger.debug(f"Daemon.subscribe({experiment_id}): closing receive stream")
            await receive_stream.aclose()
