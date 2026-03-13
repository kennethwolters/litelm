import asyncio
import os
import sys
from unittest.mock import Mock, patch, AsyncMock
import pytest
from fastapi import Request
from litelm.proxy.utils import _get_redoc_url, _get_docs_url
from datetime import datetime

sys.path.insert(0, os.path.abspath("../.."))
import litelm


@pytest.mark.asyncio
async def test_disable_spend_logs():
    """
    Test that the spend logs are not written to the database when disable_spend_logs is True
    """
    # Mock the necessary components
    import asyncio
    mock_prisma_client = Mock()
    mock_prisma_client.spend_log_transactions = []
    # Add lock for spend_log_transactions (matches real PrismaClient)
    mock_prisma_client._spend_log_transactions_lock = asyncio.Lock()

    with patch("litelm.proxy.proxy_server.disable_spend_logs", True), patch(
        "litelm.proxy.proxy_server.prisma_client", mock_prisma_client
    ):
        from litelm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter
        db_spend_update_writer = DBSpendUpdateWriter()

        # Call update_database with disable_spend_logs=True
        await db_spend_update_writer.update_database(
            token="fake-token",
            response_cost=0.1,
            user_id="user123",
            completion_response=None,
            start_time=datetime.now(),
            end_time=datetime.now(),
            end_user_id="end_user_id",
            team_id="team_id",
            org_id="org_id",
            kwargs={},
        )
        # Verify no spend logs were added
        assert len(mock_prisma_client.spend_log_transactions) == 0
