import json

from mascan.app.api import sse_from_events


def decode_sse(line: str) -> dict:
    return json.loads(line.removeprefix("data: ").strip())


def test_validator_update_is_exposed_over_sse_before_done() -> None:
    events = iter(
        [
            {
                "node": "validator",
                "update": {
                    "final_markdown": "# Report\n\n## Sources\n\n1. Source",
                    "validation": {
                        "status": "passed",
                        "summary": {"total": 0, "passed": 0, "issues": 0, "failed": 0},
                        "issues": [],
                        "checks": [],
                        "markdown": "## Citation Validation",
                    },
                },
            }
        ]
    )

    payloads = [decode_sse(line) for line in sse_from_events(events, "thread-1")]

    assert payloads[0]["event"] == "node"
    assert payloads[0]["node"] == "validator"
    assert "## Fact Check" not in payloads[0]["update"]["final_markdown"]
    assert payloads[0]["update"]["validation"]["markdown"] == "## Citation Validation"
    assert payloads[1] == {"event": "done"}
