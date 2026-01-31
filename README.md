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
