import io
import os
import sys


sys.path.insert(0, os.path.abspath("../.."))

import asyncio
import gzip
import json
import logging
import time
from unittest.mock import AsyncMock, patch

import pytest

import litelm
from litelm import completion
from litelm._logging import verbose_logger
from litelm.integrations.datadog.datadog import *
from datetime import datetime, timedelta
from litelm.types.utils import (
    StandardLoggingPayload,
    StandardLoggingModelInformation,
    StandardLoggingMetadata,
    StandardLoggingHiddenParams,
)
from litelm.integrations.azure_storage.azure_storage import AzureBlobStorageLogger

verbose_logger.setLevel(logging.DEBUG)


@pytest.mark.asyncio
async def test_azure_blob_storage():
    azure_storage_logger = AzureBlobStorageLogger(flush_interval=1)
    litelm.callbacks = [azure_storage_logger]

    response = await litelm.acompletion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello, world!"}],
    )
    print(response)

    await asyncio.sleep(3)
    pass
