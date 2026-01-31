# LangGraph Example

This directory contains a LangGraph implementation of the room scheduling agent that is compatible with Agent Lightning's APO optimization framework.

## Files

- **room_selector_langgraph.py** - LangGraph-based agent implementation compatible with APO

## Architecture

This implementation uses **LangGraph** instead of direct OpenAI API calls to build the agentic workflow, while maintaining compatibility with Agent Lightning for prompt optimization.

### Reused Components (Imported from `openai_baseline`)

The following components are imported and reused from the OpenAI baseline:

- **Data structures**: `RoomSelectionTask`, `RoomStatus`, `AvailableRooms`, `JudgeResponse`
- **Room data**: `ROOMS` constant and `overlaps()` utility function
- **Template**: `prompt_template_baseline()` for initial prompt
- **Data loading**: `load_room_tasks()` to load JSONL task data
- **Debug infrastructure**: `debug_agent()` function for testing and trace analysis

### Custom LangGraph Components

The following are custom implementations specific to this LangGraph version:

- **`get_rooms_and_availability` tool** - LangGraph `@tool` decorator implementation
- **`create_room_selector_graph()`** - LangGraph StateGraph workflow definition with:
  - `AgentState` (MessagesState)
  - `call_model` node
  - `should_continue` conditional logic
  - `ToolNode` for tool execution
- **`room_selection_grader_langgraph()`** - LangChain-based grader using:
  - `ChatOpenAI` with `.with_structured_output()`
  - `SystemMessage` and `HumanMessage` classes
- **`room_selector_langgraph()`** - Main agent function decorated with `@rollout`

## Key Differences from OpenAI Baseline

1. **Agent Framework**: LangGraph workflow vs. manual OpenAI API calls
2. **Tool Handling**: Automatic via `ToolNode` vs. manual tool call processing
3. **Message Format**: LangChain message classes (`SystemMessage`, `HumanMessage`) vs. dict format
4. **State Management**: LangGraph `MessagesState` vs. manual message list management
5. **Model**: Uses `gpt-4o-mini` (can be changed to match baseline's `gpt-4.1-nano`)

## Usage

Run from this directory:

```bash
# Debug single task execution
uv run python room_selector_langgraph.py
```

This will run the LangGraph agent on a single task with postmortem trace analysis.

## Integration with APO

The `room_selector_langgraph` function is decorated with `@rollout` and returns a float reward score, making it compatible with Agent Lightning's APO training pipeline. The prompt template can be optimized using APO just like the OpenAI baseline.

## Benefits of LangGraph Implementation

- **Graph visualization**: LangGraph workflows can be visualized
- **Streaming support**: Built-in support for streaming responses
- **State persistence**: Can persist conversation state between runs
- **Easier debugging**: Graph structure makes the agent flow explicit
- **LangChain ecosystem**: Access to LangChain tools and integrations
