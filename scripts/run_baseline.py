import json
import pandas as pd
import argparse

from pathlib import Path
from app.llm_client import generate_response


ROOT = Path(__file__).resolve().parents[1]

POLICY_FILE = ROOT / "data/policies/acme_travel_policy.md"
DATASET_FILE = ROOT / "data/evaluation/evaluation_dataset.json"
PROMPT_DIR = ROOT / "prompts"
RESULT_DIR = ROOT / "results"

STRATEGIES = [
    "structured",
    "few_shot",
    "decomposition"
]


def load_text(path):

    return path.read_text(encoding="utf-8")


def run_experiment(limit=3):

    policy = load_text(POLICY_FILE)

    with open(DATASET_FILE, encoding="utf-8") as file:
        dataset = json.load(file)

    RESULT_DIR.mkdir(exist_ok=True)

    results = []

    for strategy in STRATEGIES:

        prompt_template = load_text(
            PROMPT_DIR / f"{strategy}.txt"
        )

        print(f"\nRunning strategy: {strategy}")

        for item in dataset[:limit]:

            prompt = prompt_template.format(
                policy=policy,
                question=item["query"]
            )

            try:

                answer = generate_response(
                    system_prompt=(
                        "You are an enterprise T&E policy assistant. "
                        "Use only the supplied policy."
                    ),
                    user_prompt=prompt
                )

                status = "SUCCESS"

            except Exception as error:

                answer = str(error)
                status = "FAILED"

            results.append({
                "id": item["id"],
                "category": item["category"],
                "strategy": strategy,
                "question": item["query"],
                "gold_answer": item["gold_answer"],
                "model_answer": answer,
                "api_status": status
            })

            print(f"Completed: {item['id']} - {status}")

    output = pd.DataFrame(results)

    output.to_csv(
        RESULT_DIR / "baseline_results.csv",
        index=False
    )

    print("\nBASELINE EXPERIMENT COMPLETED")

    print(f"Total experiments: {len(results)}")

    print(
        f"Results saved to: "
        f"{RESULT_DIR / 'baseline_results.csv'}"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Number of evaluation questions to run"
    )

    args = parser.parse_args()

    run_experiment(limit=args.limit)

