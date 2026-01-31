"""LangGraph room selector compatible with AgentLightning APO."""

from typing import Literal
from rich.console import Console
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode

from agentlightning.litagent import rollout
from agentlightning.types import PromptTemplate

from examples.openai_baseline import (
    AvailableRooms,
    RoomStatus,
    overlaps,
    ROOMS,
    JudgeResponse,
    RoomSelectionTask,
)
from dotenv import load_dotenv

load_dotenv()

console = Console()


class AgentState(MessagesState):
    pass


@tool
def get_rooms_and_availability(
    date: str, time_str: str, duration_min: int
) -> AvailableRooms:
    """Return meeting rooms with capacity, equipment, accessibility, distance, and booked time slots.

    Args:
        date: Date in YYYY-MM-DD format
        time_str: Time in HH:MM 24h format
        duration_min: Meeting duration in minutes
    """
    availability: list[RoomStatus] = []
    for r in ROOMS:
        free = all(
            not (b_date == date and overlaps(time_str, duration_min, b_time, b_dur))
            for (b_date, b_time, b_dur) in r["booked"]
        )
        item: RoomStatus = {
            **r,
            "free": free,
        }
        availability.append(item)
    return {"rooms": availability}


# Initialize the LLM with tools
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
tools = [get_rooms_and_availability]
llm_with_tools = llm.bind_tools(tools)


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """Determine whether to continue or end."""
    messages = state["messages"]
    last_msg = messages[-1]

    # If there are tool calls, continue to tools
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return "end"


def call_model(state: AgentState) -> AgentState:
    """Call the LLM."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def create_room_selector_graph():
    """Create a LangGraph agent for room selection."""
    # Build the graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools))

    # Add edges
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent", should_continue, path_map={"tools": "tools", "end": END}
    )
    workflow.add_edge("tools", "agent")
    return workflow.compile()


def room_selection_grader_langgraph(final_message: str, expected_choice: str) -> float:
    """Grade the room selection using LLM judge."""
    llm = ChatOpenAI(model="gpt-5-mini", temperature=0.0)

    system_content = (
        "You are a strict grader of exact room choice. "
        "Score the match on a 0-1 scale. Be critical. "
        "Bear in mind that the score can be partially correct (between 0 and 1)."
    )

    user_content = (
        f"Task output:\n{final_message}\n\nTask expected answer:\n{expected_choice}"
    )

    response = llm.with_structured_output(JudgeResponse).invoke(
        [SystemMessage(content=system_content), HumanMessage(content=user_content)]
    )

    console.print("[bold yellow]=== Judge ===[/bold yellow]")
    console.print(f"Reason: {response.reason}")
    console.print(f"Score: {response.score}")

    return response.score


@rollout
def room_selector_langgraph(task: RoomSelectionTask, prompt_template: PromptTemplate):
    """
    LangGraph-based room selector with Agent Lightning APO optimization.
    This version uses LangGraph for agent execution and can be optimized with APO.
    """

    # Create the graph
    app = create_room_selector_graph()

    # Format the user message using the provided prompt template
    user_message = prompt_template.format(**task["task_input"])

    console.print("[bold green]=== Task ===[/bold green]")
    console.print(task)
    console.print("\n[bold yellow]=== User Message ===[/bold yellow]")
    console.print(user_message)

    # Run the agent with system prompt
    result = app.invoke(
        [
            SystemMessage(content="You are a scheduling assistant."),
            HumanMessage(content=user_message),
        ]
    )

    # Get the final messsage
    final_message = result["messages"][-1].content

    console.print("\n[bold yellow]=== Final Response ===[/bold yellow]")
    console.print(final_message)

    # Grade the response
    score = room_selection_grader_langgraph(final_message, task["expected_choice"])

    console.print("\n[bold purple]=== Final Score ===[/bold purple]")
    console.print(score)

    return score
