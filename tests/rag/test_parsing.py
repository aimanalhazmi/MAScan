from mascan.rag import parsing


class _FakeResp:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeVision:
    def __init__(self, content: str) -> None:
        self._content = content
        self.seen: list = []

    def invoke(self, messages):
        self.seen = messages
        return _FakeResp(self._content)


def test_caption_figure_returns_model_text(tmp_path, monkeypatch):
    img = tmp_path / "fig.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    fake = _FakeVision("Bar chart: revenue by segment, 2025.")
    monkeypatch.setattr(parsing, "get_vision_model", lambda temperature=0.0: fake)

    caption = parsing.caption_figure(str(img))

    assert caption == "Bar chart: revenue by segment, 2025."
    # the image was actually attached as a data URL
    block = fake.seen[0].content[1]
    assert block["image_url"]["url"].startswith("data:image/png;base64,")


def test_caption_figure_falls_back_on_failure(tmp_path, monkeypatch):
    img = tmp_path / "fig.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")

    def _boom(temperature=0.0):
        raise RuntimeError("vision endpoint down")

    monkeypatch.setattr(parsing, "get_vision_model", _boom)

    assert parsing.caption_figure(str(img)) == ""  # never raises, empty on failure
