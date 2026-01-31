# Copyright (c) Microsoft. All rights reserved.

"""This sample code demonstrates how to use an existing APO algorithm to tune the prompts."""

import logging
import os
from typing import Tuple, cast

import agentops
from dotenv import load_dotenv
from openai import AsyncOpenAI
from .room_selector import (
    RoomSelectionTask,
    load_room_tasks,
    prompt_template_baseline,
    room_selector,
)

from agentlightning import Trainer, setup_logging
from agentlightning.adapter import TraceToMessages
from agentlightning.algorithm.apo import APO
from agentlightning.execution import SharedMemoryExecutionStrategy
from agentlightning.types import Dataset

load_dotenv()

AGENTOPS_API_KEY = os.getenv("AGENTOPS_API_KEY")
agentops.init(api_key=AGENTOPS_API_KEY, default_tags=["openai agents sdk"])


def load_train_val_dataset() -> Tuple[
    Dataset[RoomSelectionTask], Dataset[RoomSelectionTask]
]:
    dataset_full = load_room_tasks()
    train_split = len(dataset_full) // 2
    dataset_train = [dataset_full[i] for i in range(train_split)]
    dataset_val = [dataset_full[i] for i in range(train_split, len(dataset_full))]
    return cast(Dataset[RoomSelectionTask], dataset_train), cast(
        Dataset[RoomSelectionTask], dataset_val
    )


def setup_apo_logger(file_path: str = "apo.log") -> None:
    """Dump a copy of all the logs produced by APO algorithm to a file."""

    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] (Process-%(process)d %(name)s)   %(message)s"
    )
    file_handler.setFormatter(formatter)
    logging.getLogger("agentlightning.algorithm.apo").addHandler(file_handler)


def main() -> None:
    setup_logging()
    setup_apo_logger()

    openai_client = AsyncOpenAI()

    algo = APO[RoomSelectionTask](
        openai_client,
        val_batch_size=10,
        gradient_batch_size=4,
        beam_width=2,
        branch_factor=2,
        beam_rounds=2,
        _poml_trace=True,
    )

    trainer = Trainer(
        algorithm=algo,
        strategy=SharedMemoryExecutionStrategy(n_runners=1),
        # APO algorithm needs a baseline
        initial_resources={"prompt_template": prompt_template_baseline()},
        # APO algorithm needs an adapter to process the traces produced by rollouts
        adapter=TraceToMessages(),
    )
    dataset_train, dataset_val = load_train_val_dataset()
    trainer.fit(
        agent=room_selector, train_dataset=dataset_train, val_dataset=dataset_val
    )

    # Get the optimized prompt and save it
    best_resources = trainer.best_resources
    optimized_prompt = best_resources["prompt_template"]

    # Save to file
    output_file = "optimized_prompt.txt"
    with open(output_file, "w") as f:
        f.write(str(optimized_prompt))

    # Display results
    print("\n" + "=" * 80)
    print("APO OPTIMIZATION COMPLETE")
    print("=" * 80)
    print(f"\nOptimized prompt saved to: {output_file}")
    print(f"\nBaseline prompt:\n{prompt_template_baseline()}")
    print(f"\nOptimized prompt:\n{optimized_prompt}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
