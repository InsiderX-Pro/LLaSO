import json
import re
import os

def extract_scores(text):
    """
    Extract pronunciation scores from text, expecting three scores: accuracy, prosodic, fluency.
    """
    pattern1 = r"(accuracy|proso|fluency).*?(?:is|:|=|as).*?(\d+)"
    pattern2 = r"(\d+)\s*?(?:for)\s*?(accuracy|proso|fluency)"
    matches = re.findall(pattern1, text.lower())
    scores = {("prosodic" if key == "proso" else key): int(value) for key, value in matches}
    if len(scores) == 3:
        return scores
    else:
        matches = re.findall(pattern2, text.lower())
        scores = {("prosodic" if key == "proso" else key): int(value) for value, key in matches}
        return scores if len(scores) == 3 else None

if __name__ == "__main__":
    # List of JSON file paths
    json_paths = [
        '../model1/pronunciation_scoring_sentence_level_test.json',
        '../model2/pronunciation_scoring_sentence_level_test.json'
    ]

    # Output directory
    output_dir = '../results'
    os.makedirs(output_dir, exist_ok=True)

    # Dictionary to store final results
    final_results = {}

    for i, json_path in enumerate(json_paths):
        success_entries = []
        failed_entries = []

        # Load JSON data
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Iterate over each data entry
        for entry in data:
            gpt_text = next(c["value"] for c in entry["conversations"] if c["from"] == "gpt")
            gpt_scores = extract_scores(gpt_text)

            if 'answer' in entry.keys():
                answer_scores = extract_scores(entry["answer"])
            elif 'prediction' in entry.keys():
                answer_scores = extract_scores(entry["prediction"])
            else:
                answer_scores = None

            if gpt_scores and answer_scores:
                success_entries.append({
                    "file": entry["voice"][0],
                    "gpt_value": gpt_scores,
                    "answer": answer_scores
                })
            else:
                failed_entries.append({
                    "file": entry["voice"][0],
                    "gpt_value": gpt_text,
                    "answer": entry["answer"] if 'answer' in entry.keys() else entry.get("prediction", "N/A")
                })

        # Create a folder named by the sequence number of json_path
        folder_name = f"output_{i}"
        folder_path = os.path.join(output_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        # Save success and failed entries to corresponding files in the folder
        success_file_path = os.path.join(folder_path, "success_entries.json")
        failed_file_path = os.path.join(folder_path, "failed_entries.json")

        with open(success_file_path, "w") as f:
            json.dump(success_entries, f, indent=2)

        with open(failed_file_path, "w") as f:
            json.dump(failed_entries, f, indent=2)

        # Statistics
        total = len(success_entries)
        accuracy = (
            sum(1 for e in success_entries if e["gpt_value"]["accuracy"] == e["answer"]["accuracy"]) / total
            if total > 0 else 0
        )
        prosodic = (
            sum(1 for e in success_entries if e["gpt_value"]["prosodic"] == e["answer"]["prosodic"]) / total
            if total > 0 else 0
        )
        fluency = (
            sum(1 for e in success_entries if e["gpt_value"]["fluency"] == e["answer"]["fluency"]) / total
            if total > 0 else 0
        )

        # Add results to the final results dictionary
        final_results[os.path.basename(json_path)] = {
            "json_path": json_path,
            "total_entries": len(data),
            "matched_entries": total,
            "accuracy": accuracy,
            "prosodic_accuracy": prosodic,
            "fluency_accuracy": fluency,
            "folder_name": folder_name
        }

    # Save the final results to a single JSON file
    output_file = os.path.join(output_dir, "pronunciation_scoring_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)

    print(f"All results have been saved to: {output_file}")