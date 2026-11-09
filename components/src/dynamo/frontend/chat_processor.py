#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Contract for custom chat processors (``--dyn-chat-processor <name>``).

A custom chat processor replaces the Rust preprocessor for chat requests, as
the built-in ``vllm`` and ``sglang`` processors do. ``--dyn-chat-processor``
names it either by an entry point in the ``dynamo.frontend.chat_processors``
group or by a ``module:function`` path; both resolve to a
:class:`ChatProcessorProvider`.

The frontend calls the provider once at startup, inside its event loop, with
the parsed :class:`~dynamo.frontend.frontend_args.FrontendConfig` and the
``--dyn-chat-processor-config`` JSON object (empty when unset). It returns the
:class:`ChatEngineFactory` the frontend calls for each served model.
"""

from typing import Any, Protocol

from dynamo._internal import ModelDeploymentCard
from dynamo.llm import ModelCardInstanceId, PythonAsyncEngine, RoutedEngine

from .frontend_args import FrontendConfig


class ChatEngineFactory(Protocol):
    """Builds the chat engine for one model's workers.

    Called once per worker set (a model's workers that share a deployment
    card), when its first worker registers, and again if the card changes. If
    it raises, the frontend retries with backoff and serves the model only once
    it succeeds. The engine serves ``/v1/chat/completions``, ``/v1/responses``
    and ``/v1/messages``; ``/v1/completions`` keeps the Rust preprocessor.

    ``mdc.runtime_config()`` returns what the worker published at registration
    (its ``runtime_data`` included), and ``mdc.local_dir()`` the downloaded
    model files. The engine sends preprocessed requests to the workers through
    ``routed_engine.generate(request, context=context)``.
    """

    async def __call__(
        self,
        instance_id: ModelCardInstanceId,
        mdc: ModelDeploymentCard,
        routed_engine: RoutedEngine,
    ) -> PythonAsyncEngine:
        ...


class ChatProcessorProvider(Protocol):
    """What a custom ``--dyn-chat-processor`` name resolves to."""

    def __call__(
        self, config: FrontendConfig, options: dict[str, Any]
    ) -> ChatEngineFactory:
        ...
