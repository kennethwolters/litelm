"""
Tests for litelm.litelm_core_utils.redact_messages.should_redact_message_logging

Covers the proxy flow where headers arrive in litelm_params["metadata"]["headers"]
but litelm_params["litelm_metadata"] is None.
"""

import pytest

import litelm
from litelm.litelm_core_utils.redact_messages import should_redact_message_logging


@pytest.fixture(autouse=True)
def _reset_global_redaction():
    """Ensure the global setting is off for every test."""
    original = litelm.turn_off_message_logging
    litelm.turn_off_message_logging = False
    yield
    litelm.turn_off_message_logging = original


def _make_model_call_details(
    metadata_headers=None,
    litelm_metadata=None,
    metadata=None,
    standard_callback_dynamic_params=None,
):
    """Build a model_call_details dict that mimics real proxy/SDK flows."""
    litelm_params = {}
    if metadata is not None:
        litelm_params["metadata"] = metadata
    elif metadata_headers is not None:
        litelm_params["metadata"] = {"headers": metadata_headers}
    else:
        litelm_params["metadata"] = {}

    # get_litelm_params always sets this key (even when value is None)
    litelm_params["litelm_metadata"] = litelm_metadata

    details = {"litelm_params": litelm_params}
    if standard_callback_dynamic_params is not None:
        details["standard_callback_dynamic_params"] = standard_callback_dynamic_params
    return details


class TestShouldRedactMessageLogging:
    """Unit tests for should_redact_message_logging()."""

    # ---- proxy flow: headers in metadata, litelm_metadata is None ----

    def test_enable_redaction_via_x_header_proxy_flow(self):
        """x-litelm-enable-message-redaction header should enable redaction
        even when litelm_metadata is None (proxy path)."""
        details = _make_model_call_details(
            metadata_headers={"x-litelm-enable-message-redaction": "true"},
            litelm_metadata=None,
        )
        assert should_redact_message_logging(details) is True

    def test_enable_redaction_via_old_header_proxy_flow(self):
        """litelm-enable-message-redaction header should enable redaction
        even when litelm_metadata is None (proxy path)."""
        details = _make_model_call_details(
            metadata_headers={"litelm-enable-message-redaction": "true"},
            litelm_metadata=None,
        )
        assert should_redact_message_logging(details) is True

    def test_disable_redaction_via_header_proxy_flow(self):
        """litelm-disable-message-redaction should suppress redaction
        even when global setting is on, and litelm_metadata is None."""
        litelm.turn_off_message_logging = True
        details = _make_model_call_details(
            metadata_headers={"litelm-disable-message-redaction": "true"},
            litelm_metadata=None,
        )
        assert should_redact_message_logging(details) is False

    # ---- SDK direct-call flow: headers in litelm_metadata ----

    def test_enable_redaction_via_header_in_litelm_metadata(self):
        """Headers inside litelm_metadata (SDK direct call) should work."""
        details = _make_model_call_details(
            litelm_metadata={"headers": {"x-litelm-enable-message-redaction": "true"}},
        )
        assert should_redact_message_logging(details) is True

    # ---- no headers at all ----

    def test_no_headers_defaults_to_global_off(self):
        """Without headers, falls back to global setting (False)."""
        details = _make_model_call_details(
            metadata_headers=None,
            litelm_metadata=None,
        )
        assert should_redact_message_logging(details) is False

    def test_no_headers_global_on(self):
        """Without headers, respects global turn_off_message_logging=True."""
        litelm.turn_off_message_logging = True
        details = _make_model_call_details(
            metadata_headers=None,
            litelm_metadata=None,
        )
        assert should_redact_message_logging(details) is True

    # ---- dynamic params take precedence ----

    def test_dynamic_param_enables_redaction(self):
        """Dynamic turn_off_message_logging=True should enable redaction."""
        details = _make_model_call_details(
            metadata_headers={},
            litelm_metadata=None,
            standard_callback_dynamic_params={"turn_off_message_logging": True},
        )
        assert should_redact_message_logging(details) is True

    def test_dynamic_param_false_overrides_header(self):
        """Dynamic turn_off_message_logging=False should take precedence over enable header."""
        details = _make_model_call_details(
            metadata_headers={"x-litelm-enable-message-redaction": "true"},
            litelm_metadata=None,
            standard_callback_dynamic_params={"turn_off_message_logging": False},
        )
        assert should_redact_message_logging(details) is False

    # ---- non-dict metadata safety ----

    def test_both_metadata_fields_none(self):
        """When both litelm_metadata and metadata are None, should not raise."""
        details = _make_model_call_details(
            metadata=None,
            litelm_metadata=None,
        )
        assert should_redact_message_logging(details) is False

    def test_both_metadata_fields_none_global_on(self):
        """When both metadata fields are None but global is on, should still return True."""
        litelm.turn_off_message_logging = True
        details = _make_model_call_details(
            metadata=None,
            litelm_metadata=None,
        )
        assert should_redact_message_logging(details) is True
