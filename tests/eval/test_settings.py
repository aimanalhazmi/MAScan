from mascan.core.settings import Settings


def test_eval_judge_model_defaults_to_gpt4o(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    s = Settings()
    assert s.eval_judge_model == "gpt-4o"


def test_eval_models_overridable(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("EVAL_JUDGE_MODEL", "gpt-4o-mini")
    s = Settings()
    assert s.eval_judge_model == "gpt-4o-mini"
