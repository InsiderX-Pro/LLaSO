import json
from evaluate import load

# Load the WER and CER evaluation modules ; borrow from huggingface metrics.
# === to load these metrics off-line ===
wer = load("/data/yirongsun/SpeechLLM/syr/repo_upload_result/Metrics/error_rate/wer/wer.py")
cer = load("/data/yirongsun/SpeechLLM/syr/repo_upload_result/Metrics/error_rate/cer/cer.py")

# Path to the JSON  /
json_path = '/code/syr/LLaSO/test/after_align_asr_test_100.json'

# Load the JSON file
with open(json_path, 'r') as f:
    data = json.load(f)

# Extract the 'answer' and the second dictionary's 'value' in 'conversations'
predictions = []
references = []

for entry in data:
    # The second 'value' in 'conversations' is the GPT's response
    predictions.append(entry['conversations'][1]['value'])
    # The 'answer' field is the reference transcription
    references.append(entry['answer'])

# Compute the WER score
wer_score = wer.compute(predictions=predictions, references=references)
print("WER Score:", wer_score)

# Compute the CER score
cer_score = cer.compute(predictions=predictions, references=references)
print("CER Score:", cer_score)
