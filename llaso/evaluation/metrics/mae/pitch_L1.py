import json
import numpy as np
import re

def parse_pitch_value(value_str):
    """
    Parse a string representing a MIDI pitch value and return an integer (0-127).

    Parsing steps:
    1. Strip leading and trailing spaces and try to convert directly to int(); if successful, return the integer.
    2. If direct int() fails, try converting to float(); if successful, round and return as integer.
    3. If both above fail, use regex to extract all numbers (supporting integers and decimals):
       - If the string contains a hyphen '-' and at least two numbers are matched, treat as a range,
         and return the average of the first two numbers (rounded).
       - Otherwise, take the first matched number, convert to float, round, and return as integer.
    4. If no numbers can be extracted but the string contains keywords (such as "midi", "pitch", "hz", "fundamental", "dominant"),
       it indicates that a value should exist in the description but cannot be extracted; raise an exception.
    5. If none of the above methods succeed, raise a ValueError.

    Args:
      value_str: The string representing MIDI pitch value, e.g. "The pitch in the given audio sample is 60.", "440 Hz", or "60-65", etc.

    Returns:
      Integer MIDI pitch value.

    Raises:
      ValueError if a valid number cannot be parsed.
    """
    s = value_str.strip()
    # Try to convert directly to integer
    try:
        return int(s)
    except ValueError:
        pass
    # Try to convert directly to float
    try:
        return int(round(float(s)))
    except ValueError:
        pass
    # Use regex to extract all numbers (including decimals)
    numbers = re.findall(r'\d+\.?\d*', s)
    if numbers:
        # If there is a hyphen '-' and at least two numbers, treat as a range and take the average
        if '-' in s and len(numbers) >= 2:
            try:
                nums = [float(num) for num in numbers[:2]]
                return int(round(sum(nums) / 2.0))
            except Exception as e:
                raise ValueError(f"Failed to parse range numbers: {s}") from e
        else:
            try:
                return int(round(float(numbers[0])))
            except Exception as e:
                raise ValueError(f"Failed to parse number: {s}") from e
    # If no number extracted but string contains possible value-related keywords, raise an exception
    keywords = ['midi', 'pitch', 'hz', 'fundamental', 'dominant']
    if any(keyword in s.lower() for keyword in keywords):
        raise ValueError(f"Failed to extract a valid value: {s}")
    # Other situations: raise exception
    raise ValueError(f"Unable to parse value: {s}")

def calculate_mae(json_path):
    """
    Calculate the Mean Absolute Error (MAE).

    This function reads data from a given JSON file. The data includes ground-truth ("result") and model predictions
    (records from "conversations" list where "from" is "gpt"). The values in the data can be plain numbers, intervals
    (e.g. "60-65"), or descriptive text (e.g. "The pitch in the given audio sample is 60.").
    Thus, the parse_pitch_value function is used for smart parsing, converting all inputs to integers before calculating MAE.

    Args:
      json_path: Path to the JSON file. The file should contain a list of data dicts.
                 Each dict has "result" for the ground-truth value, and a "conversations" list containing records
                 where a record with "from" == "gpt" gives the predicted value ("value" field).

    Returns:
      If valid data exists, MAE value is computed and printed; if no valid data, a message is returned.

    Processing steps:
      1. Read and parse the JSON file.
      2. For each entry, use parse_pitch_value to parse both ground-truth and prediction values;
         check that the parsed numbers are valid (must be in the range 0~127).
      3. Collect all valid data and keep error logs for parsing failures.
      4. Compute and print MAE and error logs.
    """
    # Read JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    y_true = []  # Store list of ground-truth values
    y_pred = []  # Store list of predicted values
    errors = []  # Store list of error messages

    # Iterate over each data entry
    for entry in data:
        try:
             # Parse ground-truth and GPT prediction values using parse_pitch_value
            answer = parse_pitch_value(entry["result"])
            gpt_value = None
            for conv in entry["conversations"]:
                if conv["from"] == "gpt":
                    gpt_value = parse_pitch_value(conv["value"])
                    break
            
            # If GPT prediction is missing, raise an exception
            if gpt_value is None:
                raise KeyError("Missing prediction from 'gpt'")
            
            # Check that both ground-truth and prediction are in valid MIDI range (0-127)
            if not (0 <= answer <= 127 and 0 <= gpt_value <= 127):
                raise ValueError("MIDI value is out of valid range (0-127)")
            
            # Collect valid data
            y_true.append(answer)
            y_pred.append(gpt_value)
        
        except (KeyError, ValueError, TypeError) as e:
            errors.append(f"Error in entry: {entry.get('voice', 'Unknown')} - {str(e)}")
            continue

    # If no valid data, return message
    if len(y_true) == 0:
        return "No valid data available for MAE calculation."
    
    # Compute Mean Absolute Error (MAE)
    mae = np.mean(np.abs(np.array(y_pred) - np.array(y_true)))
    
    # Print MAE and error log
    print(f"MAE: {mae:.2f}")
    if errors:
        print("\nEncountered the following errors:")
        for error in errors:
            print(f"  - {error}")

calculate_mae("../model1/NSynth_pitch_recognition_test.json")
