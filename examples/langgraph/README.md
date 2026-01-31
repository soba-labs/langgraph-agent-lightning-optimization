# LangGraph Example

This directory contains a LangGraph implementation of the room scheduling agent that is compatible with Agent Lightning's APO optimization framework.

## Files

- **room_selector_langgraph.py** - LangGraph-based agent implementation with debug mode
- **room_selector_langgraph_apo.py** - APO training script for LangGraph agent
- **__init__.py** - Module initialization

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
- **`create_room_selector_graph()`** - LangGraph StateGraph workflow definition
- **`room_selection_grader_langgraph()`** - LangChain-based grader
- **`room_selector_langgraph()`** - Main agent function decorated with `@rollout`

**Note on Tool Execution**: This implementation uses a custom `call_tools()` function instead of LangGraph's `ToolNode`. The `ToolNode` causes a conflict with AgentOps instrumentation (`TypeError: descriptor '__call__' for 'type' objects doesn't apply to a 'ToolNode' object`). The custom function manually invokes tools and produces the same behavior without triggering the instrumentation issue.

## Key Differences from OpenAI Baseline

1. **Agent Framework**: LangGraph workflow vs. manual OpenAI API calls
2. **Tool Handling**: Custom `call_tools()` function vs. manual tool call processing (note: cannot use LangGraph's `ToolNode` due to AgentOps instrumentation conflict)
3. **Message Format**: LangChain message classes (`SystemMessage`, `HumanMessage`) vs. dict format
4. **State Management**: LangGraph `MessagesState` vs. manual message list management
5. **Model**: Uses `gpt-4o-mini` (can be changed to match baseline's `gpt-4.1-nano`)

## Usage

Run from this directory:

```bash
# Debug single task execution
uv run python room_selector_langgraph.py

# Train with APO to optimize prompts
uv run python room_selector_langgraph_apo.py
```

The debug script runs the LangGraph agent on a single task with postmortem trace analysis. The APO script performs iterative prompt optimization using the same APO configuration as the OpenAI baseline but with the LangGraph agent framework.

## Integration with APO

The `room_selector_langgraph` function is decorated with `@rollout` and returns a float reward score, making it compatible with Agent Lightning's APO training pipeline. The `room_selector_langgraph_apo.py` script demonstrates this integration:

- Reuses `load_train_val_dataset()`, `prompt_template_baseline()`, and other utilities from `openai_baseline`
- Uses the same APO configuration (beam_width=2, branch_factor=2, beam_rounds=2)
- Logs optimization progress to `apo_langgraph.log`
- Uses `SharedMemoryExecutionStrategy` for macOS compatibility

## Benefits of LangGraph Implementation

- **Graph visualization**: LangGraph workflows can be visualized
- **Streaming support**: Built-in support for streaming responses
- **State persistence**: Can persist conversation state between runs
- **Easier debugging**: Graph structure makes the agent flow explicit
- **LangChain ecosystem**: Access to LangChain tools and integrations
