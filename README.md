# LangGraph Agent Lightning Optimization

This project demonstrates prompt optimization for LLM agents using the [Agent Lightning](https://github.com/microsoft/agent-lightning) framework with APO (Automatic Prompt Optimization).

## Overview

Agent Lightning is a framework for optimizing LLM agent prompts through iterative training. This repository contains examples showing how to:

- Define agents with traceable rollouts
- Use APO algorithm to automatically improve prompts
- Evaluate agent performance with LLM-based judges
- Work with both OpenAI and LangGraph implementations

## Project Structure

```
├── data/                           # Task datasets in JSONL format
│   └── room_tasks.jsonl           # Room scheduling task examples
├── examples/
│   ├── openai_baseline/           # OpenAI-based agent examples
│   │   ├── room_selector.py       # Baseline room scheduling agent
│   │   └── room_selector_apo.py   # APO training script
│   └── langgraph/                 # LangGraph implementations (planned)
└── CLAUDE.md                      # Development guide for Claude Code
```

## Setup

This project uses `uv` for package management. Python 3.12+ required.

```bash
# Install dependencies
uv sync

# Set up environment variables
cp .env.example .env  # Add your OpenAI API key and AgentOps API key
```

## Required Modifications

The original Agent Lightning examples require the following additional setup:

### Installing POML

The APO algorithm requires the `poml` package which is not included in Agent Lightning's dependencies:

```bash
uv add poml
```

Without this, `room_selector_apo.py` will fail with `ModuleNotFoundError: No module named 'poml'`.

### AgentOps Initialization

Both `room_selector.py` and `room_selector_apo.py` have been updated to include proper AgentOps initialization:

```python
import os
import agentops
from dotenv import load_dotenv

load_dotenv()

AGENTOPS_API_KEY = os.getenv("AGENTOPS_API_KEY")
agentops.init(api_key=AGENTOPS_API_KEY, default_tags=["openai agents sdk"])
```

Without this initialization, AgentOps traces are dummy and not properly recorded. Make sure to set your `AGENTOPS_API_KEY` in the `.env` file.

### Platform and Hardware Requirements

**Official Support**: Agent Lightning is officially supported on Linux distributions (Ubuntu 22.04 or later recommended). macOS and Windows (outside of WSL2) are not officially supported.

**Python**: Python 3.10 or newer required. We recommend using the latest patch release of Python 3.10, 3.11, or 3.12.

**GPU**: Optional—only needed for fine-tuning model weights or GPU-accelerated workloads. CPU-only environments are fully supported for evaluation and inference.

#### Running on macOS (Workaround)

Despite the lack of official support, it is possible to run Agent Lightning on macOS with modifications. The default multiprocessing strategy causes pickling errors (`AttributeError: Can't get local object`).

To fix this, `room_selector_apo.py` uses `SharedMemoryExecutionStrategy`:

```python
from agentlightning.execution import SharedMemoryExecutionStrategy

trainer = Trainer(
    algorithm=algo,
    # Use shared memory strategy to avoid multiprocessing pickling issues on macOS
    strategy=SharedMemoryExecutionStrategy(n_runners=1),
    initial_resources={"prompt_template": prompt_template_baseline()},
    adapter=TraceToMessages(),
)
```

This modification is already applied in the repository.

Note: This limits parallel execution (`n_runners=1`) which may impact training performance.

## Usage

### Running the Baseline Agent

```bash
cd examples/openai_baseline
uv run python room_selector.py
```

This runs a single task execution with the baseline prompt template for debugging.

### Training with APO

```bash
cd examples/openai_baseline
uv run python room_selector_apo.py
```

APO will iteratively optimize the prompt template across multiple rounds, improving the agent's ability to correctly select meeting rooms with a smaller model.

## How It Works

1. **Agent Definition**: The `room_selector` agent is decorated with `@rollout` to enable execution tracing
2. **Task Execution**: Agent processes room selection tasks using OpenAI function calling
3. **Evaluation**: An LLM judge scores the output against expected answers (0-1 scale)
4. **Optimization**: APO uses beam search to generate and test improved prompt variations
5. **Result**: Better prompts that work with smaller, cheaper models

## Room Scheduling Task

The example task is a meeting room scheduler that must:
- Check room availability during requested time slots
- Verify capacity meets attendee count
- Ensure required equipment is available
- Meet accessibility requirements
- Select optimal room using tie-breaking rules (capacity slack, extra equipment, distance, booking count)

Dataset contains 57 test cases with varying complexity.

## Development

```bash
# Lint code
uv run ruff check .

# Format code
uv run ruff format .
```

See [CLAUDE.md](./CLAUDE.md) for detailed development guidance.

## Acknowledgments

Examples adapted from [Microsoft Agent Lightning](https://github.com/microsoft/agent-lightning/tree/main/examples/apo).
