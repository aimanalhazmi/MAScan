from types import SimpleNamespace

from mascan.eval import pdf


class _FakePage:
    def __init__(self, text):
        self._text = text

    def extract_text(self):
        return self._text


def test_extract_joins_page_text(mocker):
    reader = mocker.Mock()
    reader.pages = [_FakePage("Hello"), _FakePage("World")]
    mocker.patch.object(pdf, "PdfReader", return_value=reader)

    out = pdf.extract_pdf_text("whatever.pdf")

    assert "Hello" in out and "World" in out


def test_extract_returns_empty_for_image_only_pdf(mocker):
    reader = mocker.Mock()
    reader.pages = [_FakePage(""), _FakePage(None)]
    mocker.patch.object(pdf, "PdfReader", return_value=reader)

    assert pdf.extract_pdf_text("scan.pdf") == ""


def test_extract_falls_back_to_pymupdf(monkeypatch):
    class _FakeDocument:
        def __enter__(self):
            return [SimpleNamespace(get_text=lambda _kind: "Fallback text")]

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(pdf, "PdfReader", None)
    monkeypatch.setitem(
        __import__("sys").modules,
        "fitz",
        SimpleNamespace(open=lambda _path: _FakeDocument()),
    )

    assert pdf.extract_pdf_text("whatever.pdf") == "Fallback text"
