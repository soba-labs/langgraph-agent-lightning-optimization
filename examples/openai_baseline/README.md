# OpenAI Baseline Example

This directory contains a baseline implementation of a room scheduling agent that demonstrates prompt optimization using the Agent Lightning framework.

## Files

- **room_selector.py** - Baseline agent implementation with a simple prompt template
- **room_selector_apo.py** - APO (Automatic Prompt Optimization) training script

## Source

These files are adapted from the [Microsoft Agent Lightning repository](https://github.com/microsoft/agent-lightning/tree/main/examples/apo).

## Modifications

- **AgentOps initialization** added to `room_selector.py` - Without proper agentops initialization, traces were dummy and not properly recorded.

## Usage

Run from this directory:

```bash
# Debug single task execution
uv run python room_selector.py

# Train with APO to optimize prompts
uv run python room_selector_apo.py
```

The APO script will iteratively improve the prompt template to work better with smaller models like `gpt-4.1-nano`, while the baseline demonstrates the initial implementation.
