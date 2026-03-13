import os
import sys
import unittest
from datetime import datetime
from unittest.mock import patch, AsyncMock, MagicMock

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import litelm
from litelm.integrations.langfuse.langfuse_otel import LangfuseOtelLogger
from litelm.integrations.opentelemetry import OpenTelemetry
from litelm.types.services import ServiceTypes
from litelm._service_logger import ServiceLogging


class TestServiceLoggerOTEL(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Reset callbacks before each test
        litelm.service_callback = []
        os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-123"
        os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-123"

    @patch("litelm.integrations.opentelemetry.OpenTelemetry._init_tracing")
    @patch("litelm.integrations.opentelemetry.OpenTelemetry._init_metrics")
    @patch("litelm.integrations.opentelemetry.OpenTelemetry._init_logs")
    async def test_langfuse_otel_ignores_service_logs(
        self, mock_logs, mock_metrics, mock_tracing
    ):
        """
        Test that LangfuseOtelLogger overrides the service logging hooks with 'pass'.
        """
        logger = LangfuseOtelLogger()

        # Verify hooks are overriden
        self.assertEqual(
            logger.async_service_success_hook.__qualname__,
            "LangfuseOtelLogger.async_service_success_hook",
        )
        self.assertEqual(
            logger.async_service_failure_hook.__qualname__,
            "LangfuseOtelLogger.async_service_failure_hook",
        )

    @patch("litelm.integrations.opentelemetry.OpenTelemetry._init_tracing")
    @patch("litelm.integrations.opentelemetry.OpenTelemetry._init_metrics")
    @patch("litelm.integrations.opentelemetry.OpenTelemetry._init_logs")
    async def test_langfuse_otel_does_not_create_proxy_request_span(
        self, mock_logs, mock_metrics, mock_tracing
    ):
        """
        Test that LangfuseOtelLogger returns None for create_litelm_proxy_request_started_span.

        This prevents empty proxy request spans from being sent to Langfuse when
        requests don't result in actual LLM calls (e.g., auth failures, health checks).
        """
        logger = LangfuseOtelLogger()

        # Verify the method is overridden
        self.assertEqual(
            logger.create_litelm_proxy_request_started_span.__qualname__,
            "LangfuseOtelLogger.create_litelm_proxy_request_started_span",
        )

        # Verify it returns None
        result = logger.create_litelm_proxy_request_started_span(
            start_time=datetime.now(),
            headers={"Authorization": "Bearer test"},
        )
        self.assertIsNone(result)

    @patch("litelm.integrations.opentelemetry.OpenTelemetry._init_tracing")
    @patch("litelm.integrations.opentelemetry.OpenTelemetry._init_metrics")
    @patch("litelm.integrations.opentelemetry.OpenTelemetry._init_logs")
    async def test_service_logging_shadowing_fix(
        self, mock_logs, mock_metrics, mock_tracing
    ):
        """
        Test the architectural fix: multiple OTEL loggers should receive logs independently.
        """
        # 1. Initialize two loggers
        langfuse_logger = LangfuseOtelLogger()
        otel_logger = OpenTelemetry()

        # 2. Setup service_callback list
        litelm.service_callback = [langfuse_logger, otel_logger]

        service_logging = ServiceLogging()

        # 3. Mock the base OpenTelemetry hook
        with patch.object(
            OpenTelemetry, "async_service_success_hook", new_callable=AsyncMock
        ) as mock_base_hook:
            # Trigger a service event
            await service_logging.async_service_success_hook(
                service=ServiceTypes.DB,
                call_type="success",
                duration=0.1,
                parent_otel_span=MagicMock(),
                start_time=0.0,
                end_time=1.0,
            )

            # The architectural fix ensures we call each correctly.
            self.assertEqual(
                mock_base_hook.call_count,
                1,
                "Generic OTEL logger should have received the log exactly once.",
            )


if __name__ == "__main__":
    unittest.main()
