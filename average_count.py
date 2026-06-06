import json
import re

def extract_score(match_info):
    """
    Extract score from strings like:
    - 'No match found, Max Score: 0.171875'
    - 'Match found, Score: 0.9234'
    """

    # First try normal Score:
    match = re.search(r"Score:\s*([0-9.]+)", match_info)

    # If not found, try Max Score:
    if not match:
        match = re.search(r"Max Score:\s*([0-9.]+)", match_info)

    if match:
        return float(match.group(1))

    return None


def compute_average_scores(json_path):
    # Load JSON
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Separate scores by label
    scores_label_0 = []
    scores_label_1 = []

    # Iterate through all categories
    for category, entries in data.items():

        for item in entries:

            label = item.get("label")
            match_info = item.get("match_info", "")

            score = extract_score(match_info)

            if score is None:
                continue

            if label == 0:
                scores_label_0.append(score)

            elif label == 1:
                scores_label_1.append(score)

    # Compute averages
    avg_label_0 = (
        sum(scores_label_0) / len(scores_label_0)
        if scores_label_0 else 0
    )

    avg_label_1 = (
        sum(scores_label_1) / len(scores_label_1)
        if scores_label_1 else 0
    )

    # Print results
    print("===== RESULTS =====")
    print(f"Label 0 count: {len(scores_label_0)}")
    print(f"Average score for label 0: {avg_label_0:.6f}")

    print()

    print(f"Label 1 count: {len(scores_label_1)}")
    print(f"Average score for label 1: {avg_label_1:.6f}")


# Example path
json_file = "/rds/user/bs816/hpc-work/exp_diss/HarmBench/results/eval/DirectRequest/qwen3.5_35B_A3B_base/eval.json"

compute_average_scores(json_file)
