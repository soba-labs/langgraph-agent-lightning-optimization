# Copyright (c) Microsoft. All rights reserved.

"""This sample code demonstrates how to use an existing APO algorithm to tune the prompts."""

import logging
import os
import platform
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

    # Detect system and configure trainer accordingly
    trainer_kwargs = {
        "algorithm": algo,
        # APO algorithm needs a baseline
        # Set it either here or in the algo
        "initial_resources": {
            # The resource key can be arbitrary
            "prompt_template": prompt_template_baseline()
        },
        # APO algorithm needs an adapter to process the traces produced by rollouts
        # Use this adapter to convert spans to messages
        "adapter": TraceToMessages(),
    }

    if platform.system() == "Linux":
        # Linux: use multiple runners for parallel execution
        trainer_kwargs["n_runners"] = 8
    else:
        # macOS/Windows: use shared memory strategy to avoid multiprocessing pickling issues
        trainer_kwargs["strategy"] = SharedMemoryExecutionStrategy(n_runners=1)

    trainer = Trainer(**trainer_kwargs)
    dataset_train, dataset_val = load_train_val_dataset()
    trainer.fit(
        agent=room_selector, train_dataset=dataset_train, val_dataset=dataset_val
    )


if __name__ == "__main__":
    main()
