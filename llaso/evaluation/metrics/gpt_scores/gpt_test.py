import os
import json
import time
from langchain_openai import AzureChatOpenAI

# ==============================
# Configurable Parameters Section
# ==============================
openai_config = {
    "key1": "",  # <REQUIRED: fill in your OpenAI API key>
    "key2": "",  # <optional: secondary key if needed>
    "deploymentName": "gpt-4o-mini",  # <your deployment/model name>
    "instanceName": "",               # <your Azure OpenAI instance name>
    "apiVersion": "2024-08-01-preview",
    "region": "",                     # <your Azure region>
    "endpoint": ""                    # <your Azure endpoint URL>
}


llm = AzureChatOpenAI(
    openai_api_key=openai_config['key1'],
    azure_endpoint=openai_config['endpoint'],
    deployment_name=openai_config['deploymentName'],
    openai_api_version=openai_config['apiVersion'],
    model_name=openai_config['deploymentName'],
    temperature=0.1,
    max_tokens=4096
)
import os, json, time

# ====== Directory Setup ======
# Directory containing .json files to be evaluated in gpt scores
eval_dir = "../model1_results_for_gpt"
# Output directory for evaluation results (.jsonl files)
result_dir = os.path.join(eval_dir, "gpt_result")
os.makedirs(result_dir, exist_ok=True)

# ========================

# ====== GPT Evaluation Prompt Templates ======
system_prompt = (
    "You are evaluating the performance of an AI assistant in an audio question answering task.\n"
    "Given a **Reference Answer** and a **Predicted Answer**, assign a score from 1 to 5 based on Relevance & Accuracy.\n"
    "Output format (exactly, no other text):\n"
    "Score: <integer 1-5>\n"
    "Explanation: <concise justification focusing on both relevance and accuracy>\n"
)
user_template = (
    "**Reference Answer:**\n{reference}\n\n"
    "**Predicted Answer:**\n{predicted}\n\n"
    "Please produce the evaluation."
)
# ===========================

# ====== Evaluation Loop ======
for fname in sorted(os.listdir(eval_dir)):
    if not fname.lower().endswith(".json"):
        continue
    input_path = os.path.join(eval_dir, fname)
    output_path = os.path.join(result_dir, fname[:-5] + ".jsonl")

    print(f"Processing {fname} => {os.path.basename(output_path)}")
    try:
        data = json.load(open(input_path, "r", encoding="utf-8"))
    except Exception as e:
        print(f"Could not read {fname}: {e}")
        continue

    # Open the .jsonl result file for this input .json
    with open(output_path, "w", encoding="utf-8") as fout:
        for idx, entry in enumerate(data):
            predicted = entry.get("answer", "").strip()
            convs = entry.get("conversations", [])
            reference = convs[1].get("value","").strip() if len(convs) > 1 else ""

            # Build the evaluation prompt
            messages = [
                ("system", system_prompt),
                ("user", user_template.format(reference=reference, predicted=predicted))
            ]
            try:
                resp = llm.invoke(messages)
                eval_text = resp.content.strip()
            except Exception as e:
                eval_text = f"ERROR: {e}"
                print(f"Evaluation failed for {fname} idx={idx}: {e}")

            # Write result immediately as a line in .jsonl
            record = {
                "file": fname,
                "index": idx,
                "predicted": predicted,
                "reference": reference,
                "evaluation": eval_text
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()           # Flush to disk after each record
            time.sleep(0.5)        # Throttle API call rate

    print(f"Finished evaluating {fname}, results saved in {output_path}")
