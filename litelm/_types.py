"""Lightweight types replacing openai.types for zero-dep operation.

Provides dict-like access wrappers and compatibility constructors so test code
can construct mock responses with the same API as litellm.
"""

import json
import uuid


def _to_dict(obj):
    """Recursively convert to dict for JSON serialization."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_to_dict(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    # Handle both __slots__ and __dict__ based classes
    if hasattr(obj, "__slots__"):
        return {k: _to_dict(getattr(obj, k)) for k in obj.__slots__}
    if hasattr(obj, "__dict__"):
        return {k: _to_dict(v) for k, v in obj.__dict__.items() if not k.startswith("_")}
    return obj


# ---------------------------------------------------------------------------
# Tool call types
# ---------------------------------------------------------------------------


class Function:
    __slots__ = ("name", "arguments")

    def __init__(self, name=None, arguments=""):
        self.name = name
        self.arguments = arguments

    def __repr__(self):
        return f"Function(name={self.name!r}, arguments={self.arguments!r})"


class ChatCompletionMessageToolCall:
    __slots__ = ("id", "type", "function")

    def __init__(self, id=None, type="function", function=None):
        self.id = id if id is not None else f"call_{uuid.uuid4().hex[:24]}"
        self.type = type
        self.function = Function(**function) if isinstance(function, dict) else function

    def __repr__(self):
        return f"ChatCompletionMessageToolCall(id={self.id!r}, type={self.type!r}, function={self.function!r})"


class ChoiceDeltaToolCallFunction:
    __slots__ = ("name", "arguments")

    def __init__(self, name=None, arguments=""):
        self.name = name
        self.arguments = arguments


class ChoiceDeltaToolCall:
    __slots__ = ("index", "id", "type", "function")

    def __init__(self, index=0, id=None, type="function", function=None):
        self.index = index
        self.id = id
        self.type = type
        if isinstance(function, dict):
            self.function = ChoiceDeltaToolCallFunction(**function)
        else:
            self.function = function or ChoiceDeltaToolCallFunction()


def _coerce_delta_tool_call(d):
    """Coerce a dict to ChoiceDeltaToolCall."""
    fn = d.get("function")
    if isinstance(fn, dict):
        fn = ChoiceDeltaToolCallFunction(**fn)
    return ChoiceDeltaToolCall(
        index=d.get("index", 0),
        id=d.get("id"),
        type=d.get("type", "function"),
        function=fn,
    )


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------


class CompletionUsage:
    __slots__ = (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "completion_tokens_details",
        "prompt_tokens_details",
    )

    def __init__(
        self,
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        completion_tokens_details=None,
        prompt_tokens_details=None,
        **kwargs,
    ):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
        self.completion_tokens_details = completion_tokens_details
        self.prompt_tokens_details = prompt_tokens_details

    def __iter__(self):
        """Support dict(usage) — DSPy access pattern."""
        for k in self.__slots__:
            yield k, getattr(self, k)


# ---------------------------------------------------------------------------
# Non-streaming types
# ---------------------------------------------------------------------------


class ChatCompletionMessage:
    __slots__ = (
        "role",
        "content",
        "tool_calls",
        "reasoning_content",
        "images",
        "thinking_blocks",
        "provider_specific_fields",
    )

    def __init__(
        self,
        role="assistant",
        content=None,
        tool_calls=None,
        reasoning_content=None,
        images=None,
        thinking_blocks=None,
        provider_specific_fields=None,
        **kwargs,
    ):
        self.role = role
        self.content = content
        if tool_calls is not None:
            self.tool_calls = [ChatCompletionMessageToolCall(**tc) if isinstance(tc, dict) else tc for tc in tool_calls]
        else:
            self.tool_calls = tool_calls
        self.reasoning_content = reasoning_content
        self.images = images
        self.thinking_blocks = thinking_blocks
        self.provider_specific_fields = provider_specific_fields

    def __getitem__(self, key):
        return getattr(self, key)


class Choice:
    __slots__ = ("index", "message", "finish_reason", "logprobs")

    def __init__(self, index=0, message=None, finish_reason="stop", logprobs=None):
        self.index = index
        self.message = ChatCompletionMessage(**message) if isinstance(message, dict) else message
        self.finish_reason = finish_reason
        self.logprobs = logprobs

    def __getitem__(self, key):
        return getattr(self, key)


class ChatCompletion:
    __slots__ = ("id", "model", "choices", "created", "object", "usage")

    def __init__(self, id="", model="", choices=None, created=0, object="chat.completion", usage=None, **kwargs):
        self.id = id
        self.model = model
        self.choices = [Choice(**c) if isinstance(c, dict) else c for c in (choices or [])]
        self.created = created
        self.object = object
        self.usage = CompletionUsage(**usage) if isinstance(usage, dict) else usage

    def model_dump(self):
        return _to_dict(self)

    def model_dump_json(self):
        return json.dumps(_to_dict(self))


# ---------------------------------------------------------------------------
# Streaming types
# ---------------------------------------------------------------------------


class ChoiceDelta:
    __slots__ = (
        "role",
        "content",
        "tool_calls",
        "reasoning_content",
        "images",
        "thinking_blocks",
        "provider_specific_fields",
    )

    def __init__(
        self,
        role=None,
        content=None,
        tool_calls=None,
        reasoning_content=None,
        images=None,
        thinking_blocks=None,
        provider_specific_fields=None,
        **kwargs,
    ):
        self.role = role
        self.content = content
        if tool_calls is not None:
            self.tool_calls = [_coerce_delta_tool_call(tc) if isinstance(tc, dict) else tc for tc in tool_calls]
        else:
            self.tool_calls = tool_calls
        self.reasoning_content = reasoning_content
        self.images = images
        self.thinking_blocks = thinking_blocks
        self.provider_specific_fields = provider_specific_fields

    def __getitem__(self, key):
        return getattr(self, key)


class ChunkChoice:
    __slots__ = ("index", "delta", "finish_reason")

    def __init__(self, index=0, delta=None, finish_reason=None):
        self.index = index
        self.delta = ChoiceDelta(**delta) if isinstance(delta, dict) else (delta or ChoiceDelta())
        self.finish_reason = finish_reason

    def __getitem__(self, key):
        return getattr(self, key)


class ChatCompletionChunk:
    __slots__ = ("id", "model", "choices", "created", "object", "usage")

    def __init__(self, id="", model="", choices=None, created=0, object="chat.completion.chunk", usage=None, **kwargs):
        self.id = id
        self.model = model
        self.choices = [ChunkChoice(**c) if isinstance(c, dict) else c for c in (choices or [])]
        self.created = created
        self.object = object
        self.usage = CompletionUsage(**usage) if isinstance(usage, dict) else usage

    def model_dump(self):
        return _to_dict(self)

    def model_dump_json(self):
        return json.dumps(_to_dict(self))


# ---------------------------------------------------------------------------
# Coercion helper
# ---------------------------------------------------------------------------


def _coerce_choice(choice):
    """Coerce a dict to a Choice object, filling in defaults for missing fields."""
    if isinstance(choice, Choice):
        return choice
    if isinstance(choice, dict):
        msg = choice.get("message", {})
        if isinstance(msg, dict):
            msg.setdefault("role", "assistant")
            msg = ChatCompletionMessage(**msg)
        choice.setdefault("finish_reason", "stop")
        choice.setdefault("index", 0)
        return Choice(
            message=msg,
            finish_reason=choice["finish_reason"],
            index=choice["index"],
            logprobs=choice.get("logprobs"),
        )
    return choice


# ---------------------------------------------------------------------------
# Wrapper types with dict-like access
# ---------------------------------------------------------------------------


class ModelResponse:
    """Wraps ChatCompletion with dict-like access and DSPy-expected attrs."""

    def __init__(self, completion=None, cache_hit=False, **kwargs):
        if completion is not None and not kwargs:
            self._completion = completion
        else:
            build_kwargs = dict(kwargs)
            build_kwargs.setdefault("id", "chatcmpl-mock")
            build_kwargs.setdefault("created", 0)
            build_kwargs.setdefault("model", "mock")
            build_kwargs.setdefault("object", "chat.completion")
            build_kwargs.setdefault("choices", [])
            usage = build_kwargs.get("usage")
            if usage is None:
                build_kwargs["usage"] = CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
            elif isinstance(usage, dict):
                usage.setdefault("prompt_tokens", 0)
                usage.setdefault("completion_tokens", 0)
                usage.setdefault("total_tokens", 0)
                build_kwargs["usage"] = CompletionUsage(**usage)
            build_kwargs["choices"] = [_coerce_choice(c) for c in build_kwargs["choices"]]
            self._completion = ChatCompletion(**build_kwargs)
        self.cache_hit = cache_hit
        self._hidden_params = {}

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return getattr(self._completion, name)

    def __getitem__(self, key):
        return getattr(self._completion, key)

    def model_dump(self):
        return _to_dict(self._completion)

    def json(self):
        return self._completion.model_dump_json()

    def __repr__(self):
        return f"ModelResponse({self._completion!r})"


class ModelResponseStream:
    """Wraps ChatCompletionChunk with dict-like access."""

    def __init__(self, chunk=None, **kwargs):
        if chunk is not None and not kwargs:
            self._chunk = chunk
        else:
            build_kwargs = dict(kwargs)
            build_kwargs.setdefault("id", "chatcmpl-chunk-mock")
            build_kwargs.setdefault("created", 0)
            build_kwargs.setdefault("model", "mock")
            build_kwargs.setdefault("object", "chat.completion.chunk")
            build_kwargs.setdefault("choices", [])
            self._chunk = ChatCompletionChunk(**build_kwargs)
        self.predict_id = None

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return getattr(self._chunk, name)

    def __getitem__(self, key):
        return getattr(self._chunk, key)

    def json(self):
        return self._chunk.model_dump_json()

    def __repr__(self):
        return f"ModelResponseStream({self._chunk!r})"


# ---------------------------------------------------------------------------
# Compatibility constructors matching litellm's API
# ---------------------------------------------------------------------------


class Message(ChatCompletionMessage):
    """ChatCompletionMessage with role defaulting to 'assistant'."""

    def __init__(self, role="assistant", **kwargs):
        super().__init__(role=role, **kwargs)


class Choices(Choice):
    """Choice with finish_reason defaulting to 'stop' and index to 0."""

    def __init__(self, finish_reason="stop", index=0, **kwargs):
        super().__init__(finish_reason=finish_reason, index=index, **kwargs)


class Usage(CompletionUsage):
    """CompletionUsage with all fields defaulting to 0."""

    def __init__(self, prompt_tokens=0, completion_tokens=0, total_tokens=0, **kwargs):
        super().__init__(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            **kwargs,
        )


# Streaming compat
Delta = ChoiceDelta


class StreamingChoices(ChunkChoice):
    """ChunkChoice with index defaulting to 0."""

    def __init__(self, index=0, **kwargs):
        super().__init__(index=index, **kwargs)
