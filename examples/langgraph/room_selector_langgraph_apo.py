from openai import AsyncOpenAI
from agentlightning import Trainer, setup_logging
from agentlightning.adapter import TraceToMessages
from agentlightning.algorithm.apo import APO
from agentlightning.execution import SharedMemoryExecutionStrategy
from room_selector_langgraph import room_selector_langgraph

# import reusable functions from openai_baseline
from examples.openai_baseline import (
    load_train_val_dataset,
    setup_apo_logger,
    RoomSelectionTask,
    prompt_template_baseline,
)
from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    setup_logging()
    setup_apo_logger(file_path="apo_langgraph.log")

    openai_client = AsyncOpenAI()

    # APO (Automatic Prompt Optimization) algorithm
    algo = APO[RoomSelectionTask](
        openai_client,
        val_batch_size=10,
        gradient_batch_size=4,
        beam_width=2,
        branch_factor=2,
        beam_rounds=2,
        _poml_trace=True,
    )

    # Trainer orchestrates the optimization process
    trainer = Trainer(
        algorithm=algo,
        strategy=SharedMemoryExecutionStrategy(n_runners=1),
        initial_resources={"prompt_template": prompt_template_baseline()},
        adapter=TraceToMessages(),
    )

    # Load the dataset
    dataset_train, dataset_val = load_train_val_dataset()

    # Train the agent
    trainer.fit(
        agent=room_selector_langgraph,
        train_dataset=dataset_train,
        val_dataset=dataset_val,
    )


if __name__ == "__main__":
    main()
