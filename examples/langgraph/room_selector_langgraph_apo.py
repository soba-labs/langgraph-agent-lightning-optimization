from openai import AsyncOpenAI
from agentlightning import Trainer, setup_logging
from agentlightning.algorithm.apo import APO
from agentlightning.execution import SharedMemoryExecutionStrategy
from room_selector_langgraph import room_selector_langgraph
from langgraph_adapter import LangGraphAdapter

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

    trainer = Trainer(
        algorithm=algo,
        strategy=SharedMemoryExecutionStrategy(n_runners=1),
        initial_resources={"prompt_template": prompt_template_baseline()},
        adapter=LangGraphAdapter(),
    )

    # Load the dataset
    dataset_train, dataset_val = load_train_val_dataset()

    # Train the agent
    trainer.fit(
        agent=room_selector_langgraph,
        train_dataset=dataset_train,
        val_dataset=dataset_val,
    )

    # Get the optimized prompt and save it
    best_resources = trainer.best_resources
    optimized_prompt = best_resources["prompt_template"]

    # Save to file
    output_file = "optimized_prompt_langgraph.txt"
    with open(output_file, "w") as f:
        f.write(str(optimized_prompt))

    # Display results
    print("\n" + "=" * 80)
    print("APO OPTIMIZATION COMPLETE (LangGraph)")
    print("=" * 80)
    print(f"\nOptimized prompt saved to: {output_file}")
    print(f"\nBaseline prompt:\n{prompt_template_baseline()}")
    print(f"\nOptimized prompt:\n{optimized_prompt}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
