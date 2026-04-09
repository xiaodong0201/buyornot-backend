from pydantic import BaseModel, Field
from typing import Any, Optional


class AnalyzeRequest(BaseModel):
    user_id: str
    text: Optional[str] = None
    image_base64: Optional[str] = None
    analysis_mode: str = "decision"


class PreviewImageRequest(BaseModel):
    text: str

class FollowupAnswer(BaseModel):
    question: str
    answer: str


class FollowupRequest(BaseModel):
    session_id: str
    user_id: str
    question: Optional[str] = None
    answer: Optional[str] = None
    answers: Optional[list[FollowupAnswer]] = None


class ProfileUpdateRequest(BaseModel):
    updates: dict[str, Any] = Field(default_factory=dict)


class FollowupOption(BaseModel):
    question: str
    options: list[str]
    question_type: str = "multiple_choice"   # "multiple_choice" | "open_text"
    open_text_placeholder: str = ""


class ScoreBreakdown(BaseModel):
    cost_value: int = 3   # 1-5
    quality: int = 3
    brand: int = 3
    fit: int = 3
    reviews: int = 3
    safety: int = 3


class DecisionResult(BaseModel):
    verdict: str          # worth_buying | cautious | not_recommended
    verdict_label: str    # Worth Buying | Buy with Caution | Don't Recommend
    summary: str
    reasons: list[str]
    scores: Optional[ScoreBreakdown] = None
    dimension_priority: Optional[list[str]] = None
    followup: Optional[FollowupOption] = None
    product_name: Optional[str] = None
    category: Optional[str] = None
