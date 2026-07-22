from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphRecursionError

from mascan.agents.environmental.agent import EnvironmentalAgent


def test_react_fallback_drops_trailing_unanswered_tool_calls(mocker: Any) -> None:
    agent = EnvironmentalAgent()
    pending_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "web_search",
                "args": {"query": "water stress"},
                "id": "call-web",
                "type": "tool_call",
            },
            {
                "name": "world_bank_environmental_indicators",
                "args": {"country_code": "DEU"},
                "id": "call-world-bank",
                "type": "tool_call",
            },
            {
                "name": "web_search",
                "args": {"query": "German manufacturing emissions"},
                "id": "call-emissions",
                "type": "tool_call",
            }
        ],
    )
    original_user_message = HumanMessage(content="Analyze water stress.")

    def stream(*args: Any, **kwargs: Any) -> Any:
        yield {"messages": [original_user_message, pending_call]}
        raise GraphRecursionError("iteration limit")

    react_agent = mocker.Mock()
    react_agent.stream.side_effect = stream
    llm = mocker.Mock()
    final = AIMessage(content="Evidence is limited, but here is the final analysis.")
    llm.invoke.return_value = final

    result = agent.invoke_react_with_fallback(
        react_agent,
        llm,
        "Analyze water stress.",
    )

    fallback_messages = llm.invoke.call_args.args[0]
    assert pending_call not in fallback_messages
    assert fallback_messages[1] is original_user_message
    assert isinstance(fallback_messages[-1], HumanMessage)
    assert "Do not call any more tools" in fallback_messages[-1].content
    assert result == {"messages": [original_user_message, final]}
