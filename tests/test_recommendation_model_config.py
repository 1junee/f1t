"""모델별 추천 이유 요청 설정이 제공자 API와 호환되는지 검사한다."""
import json
from io import BytesIO

import pytest

from pipeline.target_description import recommendation_explanation as explanation


@pytest.mark.parametrize(
    ("model", "expected_thinking"),
    [
        ("gemini-3.5-flash-lite", {"thinkingLevel": "minimal"}),
        ("models/gemini-3.5-flash-lite", {"thinkingLevel": "minimal"}),
        ("gemini-2.5-flash", {"thinkingBudget": 0}),
    ],
)
def test_recommendation_request_uses_supported_thinking_config(
    monkeypatch, model, expected_thinking
):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    captured = {}

    def fake_urlopen(request, timeout):
        captured.update(json.loads(request.data))
        return BytesIO(json.dumps({
            "candidates": [{"content": {"parts": [{"text": "추천 문구입니다. 조건을 반영했습니다."}]}}]
        }).encode())

    monkeypatch.setattr(explanation.urllib.request, "urlopen", fake_urlopen)
    result = explanation._generate_gemini_recommendation_reason(
        prompt="검정 후드티", model_name=model
    )

    assert captured["generationConfig"]["thinkingConfig"] == expected_thinking
    assert result == "추천 문구입니다. 조건을 반영했습니다."
