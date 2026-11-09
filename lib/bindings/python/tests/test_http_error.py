# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo._core import _map_python_exception, _raise_routed_http_error
from dynamo.llm import HttpError
from dynamo.llm.exceptions import RoutedEngineError

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.pre_merge,
    pytest.mark.unit,
]


def test_raise_http_error():
    with pytest.raises(HttpError):
        raise HttpError(404, "Not Found")
    with pytest.raises(Exception):
        raise HttpError(500, "Internal Server Error")


def test_invalid_http_error_code():
    with pytest.raises(ValueError):
        HttpError(1700, "Invalid Code")


def test_invalid_http_error_message():
    with pytest.raises(ValueError):
        # The second argument must be a string, not bytes.
        HttpError(400, b"Bad Request")


def test_long_http_error_message():
    message = ("A" * 8192) + "B"
    error = HttpError(400, message)
    assert len(error.message) == 8192

    # Ensure the exception string uses the truncated message too.
    assert message[:8189] in str(error)
    assert "B" not in str(error)


def test_routed_engine_error_keeps_backend_status():
    with pytest.raises(RoutedEngineError) as raised:
        _raise_routed_http_error(503, "worker busy")

    error = raised.value
    # Callers that caught the ValueError raised before still catch it.
    assert isinstance(error, ValueError)
    assert (error.error_class, error.status_code) == ("Unavailable", 503)
    # Escaping a chat engine restores the backend's error instead of a 400.
    assert _map_python_exception(error) == str(error)
