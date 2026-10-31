"""Typed native System One data; no generated text or application thresholds."""

import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

Probability = Annotated[float, Field(ge=0, le=1)]


class _Data(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False, hide_input_in_errors=True)


class NoulQuestion(_Data):
    type: Literal["noul"] = "noul"
    instructions: JsonValue | None = None
    criteria: dict[Literal["true", "false"], JsonValue] | None = None

    @model_validator(mode="after")
    def require_question(self):
        if self.instructions is None and not self.criteria:
            raise ValueError("A noul question requires instructions or criteria")
        return self


class ChoiceQuestion(_Data):
    type: Literal["choice"] = "choice"
    instructions: JsonValue | None = None
    criteria: dict[str, JsonValue] = Field(min_length=2, max_length=255)


class ScoreQuestion(_Data):
    type: Literal["score"] = "score"
    instructions: JsonValue | None = None
    criteria: list[JsonValue] = Field(min_length=2, max_length=10)


DecisionQuestion = Annotated[NoulQuestion | ChoiceQuestion | ScoreQuestion, Field(discriminator="type")]


class SystemOneRequest(_Data):
    model: str = Field(min_length=1)
    state: JsonValue
    questions: dict[Annotated[str, Field(min_length=1)], DecisionQuestion] = Field(min_length=1, max_length=128)


class NoulAnswer(_Data):
    type: Literal["noul"] = "noul"
    noul: Probability


class ChoiceAnswer(_Data):
    type: Literal["choice"] = "choice"
    choice: str
    confidence: Probability
    probabilities: dict[str, Probability] = Field(min_length=2)


class ScoreAnswer(_Data):
    type: Literal["score"] = "score"
    score: float = Field(ge=0)
    confidence: Probability
    legend: dict[str, JsonValue]
    probabilities: dict[str, Probability] = Field(min_length=2)


DecisionAnswer = Annotated[NoulAnswer | ChoiceAnswer | ScoreAnswer, Field(discriminator="type")]


class DecisionUsage(_Data):
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    cached_tokens: int | None = Field(default=None, ge=0)
    cache_write_tokens: int | None = Field(default=None, ge=0)


class SystemOneResponse(_Data):
    model: str | None = None
    answers: dict[str, DecisionAnswer] = Field(min_length=1)
    usage: DecisionUsage | None = None
    cache_hit: bool = False

    def validate_for(self, request: SystemOneRequest):
        """Validate answer identity and distributions against the actual question space."""
        if self.answers.keys() != request.questions.keys():
            raise ValueError("Decision answer names do not match questions")
        for name, question in request.questions.items():
            answer = self.answers[name]
            if answer.type != question.type:
                raise ValueError("Decision answer type does not match question")
            if isinstance(answer, NoulAnswer):
                continue
            if isinstance(question, ChoiceQuestion):
                keys = set(question.criteria)
                if answer.choice not in keys:
                    raise ValueError("Decision selected an unknown choice")
            else:
                keys = {str(i) for i in range(len(question.criteria))}
                if set(answer.legend) != keys or answer.score > len(keys) - 1:
                    raise ValueError("Decision score does not match rubric")
            if set(answer.probabilities) != keys or not math.isclose(
                sum(answer.probabilities.values()), 1, abs_tol=0.001
            ):
                raise ValueError("Decision distribution does not match question")
        return self
