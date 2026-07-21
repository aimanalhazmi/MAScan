import json

from mascan.app.api import sse_from_events


def decode_sse(line: str) -> dict:
    return json.loads(line.removeprefix("data: ").strip())


def test_validator_update_is_exposed_over_sse_before_done(mocker) -> None:
    mocker.patch(
        "mascan.app.api.time.perf_counter",
        side_effect=[10.0, 11.25],
    )
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
    assert payloads[1] == {"event": "done", "duration_seconds": 1.25}


def test_clarification_reports_only_active_execution_duration(mocker) -> None:
    mocker.patch(
        "mascan.app.api.time.perf_counter",
        side_effect=[20.0, 20.5],
    )
    events = iter(
        [
            {
                "node": "__interrupt__",
                "question": "Which market?",
            }
        ]
    )

    payloads = [decode_sse(line) for line in sse_from_events(events, "thread-1")]

    assert payloads == [
        {
            "event": "clarification",
            "question": "Which market?",
            "thread_id": "thread-1",
            "duration_seconds": 0.5,
        }
    ]
