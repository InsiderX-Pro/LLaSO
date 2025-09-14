#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Program Overview:
-----------------
This script evaluates instruction-following behavior (abstention rate) for speech or audio classification model outputs stored as .json files in a given directory.

Main workflow:
1. Find all .json files in a specified directory.
2. For a user-defined set of keywords, check that exactly one .json file matches each keyword (by filename substring).
3. For each matched file:
    a. Extract the set of all distinct label values from its prediction outputs.
    b. Compute the "abstention rate": the proportion of predictions that contain *exactly one* valid label from the set (case-insensitive, strict matching, avoiding partial matches).
    c. Report sample cases where predictions are invalid (zero or more than one label present).
4. Save the evaluation results as JSONL for further analysis.

"""


import re
import os
import sys
import json
import datetime   

def find_json_files(base_path):
    """
    Traverse base_path and return a list of all filenames ending with .json.

    Args:
        base_path (str): Directory path
    Returns:
        List[str]: List of JSON filenames
    """
    try:
        entries = os.listdir(base_path)
    except Exception as e:
        sys.exit(f"Unable to read directory {base_path}: {e}")
    jsons = [f for f in entries if f.lower().endswith('.json')]
    return jsons

def check_keyword_files(base_path, json_files, keywords):
    """
    For each keyword, check if there is exactly one file in json_files containing that keyword.

    Args:
        base_path (str): Directory path
        json_files (List[str]): List of JSON filenames
        keywords (List[str]): List of keywords
    Returns:
        Dict[str,str]: Mapping from keyword to matched filename (for keywords matched successfully)
    """
    matches_map = {}
    all_ok = True
    print(f"\nChecking JSON file matches in directory `{base_path}`\n")
    for kw in keywords:
        matches = [f for f in json_files if kw in f]
        if len(matches) == 1:
            fname = matches[0]
            print(f"Keyword `{kw}`: found unique matching file `{fname}`")
            matches_map[kw] = fname
        elif len(matches) == 0:
            print(f"Keyword `{kw}`: no matching file found!")
            all_ok = False
        else:
            print(f"Keyword `{kw}`: multiple matching files found: {matches}!")
            all_ok = False

    extra = [f for f in json_files if not any(kw in f for kw in keywords)]
    if extra:
        print(f"\n\n Note: The following JSON files in the directory do not match any keyword (total {len(extra)}):")
        for f in extra:
            print("   •", f)

    print("Checking complete" + (" - all keywords matched correctly!" if all_ok else " - some keywords mismatched, please check above errors."))
    return matches_map

def read_json_list(file_path):
    """
    Read a .json file (expected to be a JSON array) and return its parsed list.
    If the file is not an array but is JSONL (one JSON object per line), try parsing line by line.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            # Try to load as a single JSON array
            data = json.loads(content)
            if isinstance(data, list):
                return data
    except json.JSONDecodeError:
        pass
     # Fallback: parse line by line as JSONL
    items = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Skipping unparsable line: {line[:80]}...")
    return items

def get_json_labels(file_path):
    """
    Traverse a JSON array and extract entry['conversations'][1]['value'] as a label for each entry,
    and return the set of unique labels.

    Args:
        file_path (str): Path to JSON file
    Returns:
        Set[str]: Set of unique labels (deduplicated)
    """
    data = read_json_list(file_path)
    labels = set()
    for entry in data:
        conv = entry.get('conversations')
        if isinstance(conv, list) and len(conv) >= 2:
            val = conv[1].get('value', '')
            if isinstance(val, str):
                labels.add(val.strip())
    print(f"\nFile `{os.path.basename(file_path)}` extracted {len(labels)} unique labels:")
    for lbl in sorted(labels):
        print("   •", lbl)
    return labels


def _normalize_label(lbl: str) -> str:
    """Normalize label: remove trailing period, lower-case, strip spaces."""
    return lbl.lower().rstrip(". ").strip()

def abstention_rate_cal(file_path, valid_labels, debug_samples=10):
    """
    Compute the "instruction-following rate": the proportion of predictions that contain **exactly one and only one valid label**.
    Label matching uses the following rule:
        (?<![a-z0-9])  label_norm  (?![a-z0-9])
    -- The label must not be adjacent to letters or numbers on either side. This avoids 'male' matching 'female',
       while still supporting labels like '20-24'.
    """
    data = read_json_list(file_path)

    # 1) Normalize labels and compile regex patterns
    norm_labels = [_normalize_label(lab) for lab in valid_labels]
    patterns = {
        lab: re.compile(rf"(?<![a-z0-9]){re.escape(lab)}(?![a-z0-9])", flags=re.I)
        for lab in norm_labels
    }

    total, followed, bad_examples = 0, 0, []

    for i, entry in enumerate(data):
        conv = entry.get("conversations")
        if not (isinstance(conv, list) and len(conv) >= 2):
            continue

        total += 1
        pred_raw = str(entry.get("answer", "")).strip().lower()

        # 2) Count the number of matched labels
        hit = [lab for lab, pat in patterns.items() if pat.search(pred_raw)]
        hit_cnt = len(hit)

        if hit_cnt == 1:
            followed += 1
        else:
            bad_examples.append((entry.get("index", i), pred_raw, hit_cnt, hit))

    abstained = total - followed
    rate = abstained / total * 100 if total else 0.0
    fname = os.path.basename(file_path)

    # ----------- Result dictionary -----------
    result_dict = {
        "file": os.path.basename(file_path),
        "total": total,
        "one_hit": followed,
        "others": abstained,
        "abstention_rate": round(rate, 4),   # 0-1  
        "checked_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "bad_examples": bad_examples[:debug_samples]  # Only keep the first N
    }
    
    # 3) Print statistics
    print(f"\n Instruction-following statistics  <{fname}>")
    print(f"   Total samples               : {total}")
    print(f"   Samples with 1 matched label: {followed}")
    print(f"   Others (0 or >1)            : {abstained}")
    print(f"   Abstention rate             : {rate:.2f}%")

    # 4) 打印部分异常样本
    if bad_examples:
        shown = bad_examples[:debug_samples]
        print(f"\nFound {len(bad_examples)} non-conforming samples, showing first {len(shown)}:")
        for k, (idx, pred, cnt, hit) in enumerate(shown, 1):
            print(
                f"  #{k:<2} entry[index]={idx:<5} | matches={cnt:<2} | matched labels={hit or '-'} | prediction='{pred}'"
            )
    else:
        print("\n All samples contain exactly one valid label.")
        
    return result_dict

def main():
    # ========== User Config Section ==========
    OUTPUT_JSONL = "your output jsonl path"
    
    BASE_PATH = ""   # TODO: change to actual directory
    KEYWORDS = [
        "NSynth_music_source_classification_test",
        "vctk_gender_classification_test",
        "vctk_age_classification_test",
        "vocalsound_gender_classification_test",
        "accentdb_AI_classification_test",
        "common-voice_accent_classification_test",
        "meld_SV_classification_test",
        "vctk_accent_classification_test",
        "voxceleb_gender_classification_test",
        "CREMAD_EIE_classification_test",
        "common-voice_age_classification_test",
        "Synthetic_Audio_Classification_test",
        "NSynth_instrument_classification_test",
        "common-voice_gender_classification_test",
        "vocalsound_vocal_classification_test",
        "meld_ER_classification_test",
        "velocity_classification_test",
        "CREMAD_ER_classification_test",
        "meld_EIE_classification_test"
    ]
    
    
    # ================================

    # 1. Get all JSON files in directory
    json_files = find_json_files(BASE_PATH)
    if not json_files:
        sys.exit(f"Nope .json files found in directory `{BASE_PATH}`.")

     # 2. Check keywords and get mapping to matched filenames
    matches = check_keyword_files(BASE_PATH, json_files, KEYWORDS)

    # 3. For each matched file, extract labels and compute abstention rate
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:
        for kw, fname in matches.items():
            path = os.path.join(BASE_PATH, fname)
            labels = get_json_labels(path)
            stats = abstention_rate_cal(path, labels)   
            fout.write(json.dumps(stats, ensure_ascii=False) + "\n")

    print(f"\nStatistics saved to: {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()
