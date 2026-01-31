# OpenAI Baseline Example

This directory contains a baseline implementation of a room scheduling agent that demonstrates prompt optimization using the Agent Lightning framework.

## Files

- **room_selector.py** - Baseline agent implementation with debug mode for single task execution
- **room_selector_apo.py** - APO (Automatic Prompt Optimization) training script
- **__init__.py** - Exports reusable components (data structures, tools, utilities) for use by other implementations

## Source

These files are adapted from the [Microsoft Agent Lightning repository](https://github.com/microsoft/agent-lightning/tree/main/examples/apo).

## Modifications

- **AgentOps initialization** added to both `room_selector.py` and `room_selector_apo.py` - Without proper agentops initialization, traces were dummy and not properly recorded.
- **POML dependency** - The APO algorithm requires `poml` package: `uv add poml`
- **macOS compatibility** in `room_selector_apo.py` - Uses `SharedMemoryExecutionStrategy` (from `agentlightning.execution`) instead of default multiprocessing to avoid pickling errors on macOS. This limits parallel execution (`n_runners=1`) but allows the code to run on macOS despite official Linux-only support.
- **Reusable debug function** - Created generic `debug_agent()` function that can be used by other implementations (e.g., LangGraph). This function handles agent execution, trace collection, and postmortem analysis.

## Usage

Run from this directory:

```bash
# Debug single task execution
uv run python room_selector.py

# Train with APO to optimize prompts
uv run python room_selector_apo.py
```

The APO script will iteratively improve the prompt template to work better with smaller models like `gpt-4.1-nano`, while the baseline demonstrates the initial implementation.
