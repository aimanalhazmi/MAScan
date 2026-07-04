from mascan.contracts.retrieval import Citation, RetrievedChunk
from mascan.rag.answer import collect_image_blocks, select_cited


def _chunk(images: list[str], page: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        content="x",
        source="upload",
        citation=Citation(document="d", page=page),
        metadata={"images": images},
        score=0.5,
    )


def testselect_cited_keeps_only_referenced_and_renumbers():
    chunks = [_chunk([], page=1), _chunk([], page=2), _chunk([], page=3)]
    # model cited the 2nd and 3rd chunks (not the 1st), 3rd twice
    answer, citations = select_cited("Margin rose [2]. Revenue grew [3] and [3].", chunks)

    assert [c.page for c in citations] == [2, 3]  # only referenced, in order
    assert answer == "Margin rose [1]. Revenue grew [2] and [2]."  # renumbered


def testselect_cited_no_markers_gives_no_citations():
    chunks = [_chunk([], page=1)]
    answer, citations = select_cited("I don't have enough information.", chunks)
    assert citations == []
    assert answer == "I don't have enough information."


def testcollect_image_blocks_dedups_and_skips_missing(tmp_path):
    img = tmp_path / "fig.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")  # bytes are enough; not decoded here
    missing = tmp_path / "gone.png"

    blocks = collect_image_blocks(
        [_chunk([str(img), str(img)]), _chunk([str(missing)])]
    )

    assert len(blocks) == 1  # deduped, missing skipped
    assert blocks[0]["type"] == "image_url"
    assert blocks[0]["image_url"]["url"].startswith("data:image/png;base64,")


def testcollect_image_blocks_empty_when_no_images():
    assert collect_image_blocks([_chunk([])]) == []
