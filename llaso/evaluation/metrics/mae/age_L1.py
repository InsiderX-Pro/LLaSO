import json
import numpy as np
import re
import os

def parse_midi_value(value_str):
    """
    Parse a string representing an age value and return it as an integer.
    """
    s = value_str.strip()
    try:
        return int(s)
    except ValueError:
        pass
    
    numbers = re.findall(r'\d+', s)
    if numbers:
        if '-' in s and len(numbers) >= 2:
            try:
                nums = [int(num) for num in numbers[:2]]
                return int(round(sum(nums) / 2.0))
            except Exception as e:
                raise ValueError(f"Failed to parse range numbers: {s}") from e
        else:
            try:
                return int(numbers[0])
            except Exception as e:
                raise ValueError(f"Failed to parse number: {s}") from e
    
    if 'adult' in s.lower():
        return 22
    
    raise ValueError(f"Unable to parse value: {s}")

def calculate_mae(json_path, output_dir):
    """
    Calculate the Mean Absolute Error (MAE) and save the result to a JSON file.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    y_true = []
    y_pred = []
    errors = []

    for entry in data:
        try:
            if 'prediction' in entry.keys():
                answer = parse_midi_value(entry["prediction"])
            elif 'answer' in entry.keys():
                answer = parse_midi_value(entry["prediction"])
            # Parse the model prediction from the "conversations" list, where "from" == "gpt"
            gpt_value = None
            for conv in entry["conversations"]:
                if conv["from"] == "gpt":
                    gpt_value = parse_midi_value(conv["value"])
                    break
            
            if gpt_value is None:
                raise KeyError("Missing prediction from 'gpt'")
            
            
            y_true.append(answer)
            y_pred.append(gpt_value)
        
        except (KeyError, ValueError, TypeError) as e:
            errors.append(f"Error in entry: {entry.get('voice', 'Unknown')} - {str(e)}")
            continue

    if len(y_true) == 0:
        result = {"message": "No valid data available for MAE calculation.", "errors": errors}
    else:
        mae = np.mean(np.abs(np.array(y_pred) - np.array(y_true)))
        result = {
            "json_path": json_path,
            "mae": round(mae, 2),
            "errors": errors
        }
    
    # save results to output_file
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"mae_results_{os.path.basename(json_path).split('.')[0]}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    print(f"Result saved to: {output_file}")

json_paths = [
    "../model1/NSynth_pitch_recognition_test.json",
    "../model2/NSynth_pitch_recognition_test.json",
]

output_dir = "../results"

for json_path in json_paths:
    calculate_mae(json_path, output_dir)