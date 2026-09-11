from scripts.upstream_watch import _is_actionable, diff_snapshots


def test_exported_imported_function_signature_change_is_actionable():
    old = {
        "api_surface": {
            "exports": ["stream_chunk_builder"],
            "functions": {"stream_chunk_builder": ["chunks"]},
        }
    }
    new = {
        "api_surface": {
            "exports": ["stream_chunk_builder"],
            "functions": {"stream_chunk_builder": ["chunks", "count_prompt_tokens"]},
        }
    }
    ours = {
        "exports": {"stream_chunk_builder"},
        "functions": {},
        "type_classes": set(),
        "exception_classes": set(),
        "providers": set(),
    }

    changes = diff_snapshots(old, new, ours)

    assert len(changes) == 1
    assert changes[0]["impact"] == "May need passthrough"
    assert _is_actionable(changes[0])
