import os
import json

def read_json(file_path):
    """Read a JSON file and return its Python object (list or dict)."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(file_path, data):
    """Write the given Python object to a JSON file (UTF-8, pretty-printed)."""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Merged file saved: {file_path}")

def merge_subdir_jsons(base_path, subdirs):
    """
    Traverse and merge all .json files under specified subdirectories.

    Each JSON file may contain:
      - a list of samples, or
      - a single sample object.

    All contents are appended into one big Python list.

    Args:
        base_path (str): Absolute path to the parent directory (e.g., /LLaSO-Instruct).
        subdirs   (list): Subdirectory names to search (e.g., ["audio_text", "pure_audio", "text_audio"]).

    Returns:
        list: Merged list of all samples across subdirs.
    """
    merged_data = []
    for sub in subdirs:
        sub_path = os.path.join(base_path, sub)
        if not os.path.isdir(sub_path):
            print(f"⚠️ Subdirectory does not exist, skipping: {sub_path}")
            continue

        # Traverse all .json files in the subdirectory
        for fname in os.listdir(sub_path):
            if fname.endswith('.json'):
                fpath = os.path.join(sub_path, fname)
                try:
                    print(f"→ Reading: {fpath}")
                    data = read_json(fpath)

                    # Allow both list and single object formats
                    if isinstance(data, list):
                        merged_data.extend(data)
                    else:
                        merged_data.append(data)
                except Exception as e:
                    print(f"❌ Failed to read {fpath}: {e}")
    return merged_data

if __name__ == "__main__":
    # === Configuration ===
    # Base directory where different modality subdirs are stored.
    BASE_PATH = "/LLaSO-Instruct"  # Change to your own base path

    # Subdirectories corresponding to modality configs:
    #   - "audio_text"   : audio instruction + text input
    #   - "pure_audio"   : audio only
    #   - "text_audio"   : text instruction + audio input
    # Adjust the list if you only want a subset.
    SUBDIRS = ["audio_text", "pure_audio", "text_audio"]

    # Output path for merged JSON file
    OUTPUT_JSON_PATH = "/LLaSO-Instruct.json"  # Change to your output path
    # =====================

    # Warm reminder about runtime
    print("⏳ Merging may take some time since the dataset contains many large files...")
    
    # Merge all selected subdirs
    merged_data = merge_subdir_jsons(BASE_PATH, SUBDIRS)
    print(f"🎯 Total merged samples: {len(merged_data)}")

    # Save merged dataset
    write_json(OUTPUT_JSON_PATH, merged_data)
    print(f"🌟 All data merged and saved.")
