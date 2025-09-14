from collections import namedtuple
import json
import leven  # pip install leven
import numpy as np

SparseTensor = namedtuple('SparseTensor', 'indices vals shape')

PHN_MAPPING = {
    'iy': 'iy', 'ix': 'ix', 'ih': 'ix', 'eh': 'eh', 'ae': 'ae', 'ax': 'ax', 'ah': 'ax',
    'ax-h': 'ax', 'uw': 'uw', 'ux': 'uw', 'uh': 'uh', 'ao': 'ao', 'aa': 'ao', 'ey': 'ey',
    'ay': 'ay', 'oy': 'oy', 'aw': 'aw', 'ow': 'ow', 'er': 'er', 'axr': 'er', 'l': 'l', 'el': 'l',
    'r': 'r', 'w': 'w', 'y': 'y', 'm': 'm', 'em': 'm', 'n': 'n', 'en': 'n', 'nx': 'n', 'ng': 'ng',
    'eng': 'ng', 'v': 'v', 'f': 'f', 'dh': 'dh', 'th': 'th', 'z': 'z', 's': 's', 'zh': 'zh',
    'sh': 'zh', 'jh': 'jh', 'ch': 'ch', 'b': 'b', 'p': 'p', 'd': 'd', 'dx': 'dx', 't': 't',
    'g': 'g', 'k': 'k', 'hh': 'hh', 'hv': 'hh', 'bcl': 'h#', 'pcl': 'h#', 'dcl': 'h#', 'tcl': 'h#',
    'gcl': 'h#', 'kcl': 'h#', 'q': 'h#', 'epi': 'h#', 'pau': 'h#', 'h#': 'h#'
}

IDX_MAPPING = {
    0: 3, 1: 1, 2: 5, 3: 3, 4: 4, 5: 5, 6: 5, 7: 22, 8: 8, 9: 9, 10: 27, 11: 11, 12: 12, 13: 27,
    14: 14, 15: 15, 16: 16, 17: 36, 18: 37, 19: 38, 20: 39, 21: 27, 22: 22, 23: 23, 24: 24, 25: 25,
    26: 27, 27: 27, 28: 28, 29: 28, 30: 31, 31: 31, 32: 32, 33: 33, 34: 34, 35: 27, 36: 36, 37: 37,
    38: 38, 39: 39, 40: 38, 41: 41, 42: 42, 43: 43, 44: 27, 45: 27, 46: 27, 47: 47, 48: 48, 49: 60,
    50: 50, 51: 27, 52: 52, 53: 53, 54: 54, 55: 54, 56: 56, 57: 57, 58: 58, 59: 59, 60: 60
}


def calc_PER(pred, ground_truth, normalize=True, merge_phn=True):
    """
    Calculate Phoneme Error Rate (PER), based on the python package leven for edit distance.
    This function converts the sparse tensor format input into sequence lists,
    then transforms each sequence into a string where each character represents a phoneme.
    It computes the error rate between prediction and reference using the edit distance algorithm.
    By default, the edit distance is normalized (divided by the reference sequence length).

    Args:
        pred: Sparse tensor containing predictions, type is SparseTensor (namedtuple with indices, vals, shape).
        ground_truth: Sparse tensor containing ground truth, type is SparseTensor (namedtuple with indices, vals, shape).
        normalize: Boolean, whether to normalize the edit distance. If True, the error rate is between 0 and 1.
        merge_phn: Boolean, whether to map 61 phonemes to 39 phonemes before computing edit distance (default True).

    Returns:
        Returns the average Phoneme Error Rate (PER).

    Notes:
    1. If the reference sequence is empty (length 0):
        - If the prediction sequence is also empty, it is considered a correct match and the error rate is 0;
        - If the prediction sequence is not empty, it is considered a complete error and the error rate is 1.
    2. The function requires the number of prediction and reference samples to be equal, otherwise an exception will be thrown.
    """
    # Convert sparse tensor to sequence lists, then to string representations (each character for one phoneme)
    pred_seq_list = seq_to_single_char_strings(sparse_tensor_to_seq_list(pred, merge_phn=merge_phn))
    truth_seq_list = seq_to_single_char_strings(sparse_tensor_to_seq_list(ground_truth, merge_phn=merge_phn))
    
    # Ensure the numbers of prediction and reference samples are the same
    assert len(truth_seq_list) == len(pred_seq_list), "The number of prediction and reference samples does not match!"
    
    distances = []  # Used to store edit distances for each sample (possibly normalized)
    for i in range(len(truth_seq_list)):
        # Handle empty reference sequence to avoid division by zero
        if len(truth_seq_list[i]) == 0:
            # If reference is empty:
            # - If prediction is also empty, set error rate to 0 (correct);
            # - If prediction is not empty, set error rate to 1 (completely wrong).
            if len(pred_seq_list[i]) == 0:
                distances.append(0)
            else:
                distances.append(1)
            continue

        # Compute the edit distance for the current sample
        dist_i = leven.levenshtein(pred_seq_list[i], truth_seq_list[i])
        # Normalize by reference length if needed
        if normalize:
            dist_i /= float(len(truth_seq_list[i]))
        distances.append(dist_i)
    
    # Return the average phoneme error rate across all samples
    return np.mean(distances)

def seq_to_single_char_strings(seq):
    strings = []
    for s in seq:
        strings.append(''.join([chr(65 + p) for p in s]))
    return strings

def string_to_int_sequence(s):
    """
    Convert a string to an integer sequence, assuming each character represents a phoneme.
    Converts to uppercase and only keeps characters in A-Z, so that ord(ch)-65 is in 0-25.
    """
    s = s.upper()
    return [ord(ch) - 65 for ch in s if 'A' <= ch <= 'Z']

def sparse_tensor_to_seq_list(sparse_seq, merge_phn=True):
    phonemes_list = []
    it = 0
    num_samples = np.max(sparse_seq.indices, axis=0)[0] + 1
    for n in range(num_samples):
        cur_sample_indices = sparse_seq.indices[sparse_seq.indices[:, 0] == n, 1]
        if len(cur_sample_indices) == 0:
            seq_length = 0
        else:
            seq_length = np.max(cur_sample_indices) + 1
        seq = sparse_seq.vals[it:it+seq_length]
        _seq = [IDX_MAPPING[p] for p in seq] if merge_phn else seq
        phonemes_list.append(_seq)
        it += seq_length
    return phonemes_list

def list_to_sparse_tensor(seq_list):
    """
    Convert a list of integer sequences to SparseTensor format.
    """
    indices = []
    vals = []
    for i, seq in enumerate(seq_list):
        for j, token in enumerate(seq):
            indices.append([i, j])
            vals.append(token)
    if seq_list:
        max_len = max(len(seq) for seq in seq_list)
    else:
        max_len = 0
    shape = np.array([len(seq_list), max_len], dtype=np.int64)
    indices = np.array(indices, dtype=np.int64)
    vals = np.array(vals, dtype=np.int32)
    return SparseTensor(indices, vals, shape)

if __name__ == "__main__":
    # Paths to JSON files

    json_paths = [
        '/code/syr/LLaSO/test/phoneme_recognition_test.json',
        #'.../model2/phoneme_recognition_test.json'
        ]
    for json_path in json_paths:
        # read JSON 
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract prediction and reference strings
        # Note: predictions are from conversations[1]['value'], references from answer
        pred_strings = []
        ref_strings = []
        for entry in data:
            pred_strings.append(entry['conversations'][1]['value'])
            ref_strings.append(entry['answer'])
        
        # Convert strings to integer sequences (one per sample)
        pred_seqs = [string_to_int_sequence(s) for s in pred_strings]
        ref_seqs = [string_to_int_sequence(s) for s in ref_strings]
        
        # Convert to SparseTensor format
        pred_sparse = list_to_sparse_tensor(pred_seqs)
        ref_sparse = list_to_sparse_tensor(ref_seqs)
        
        # calculate PER
        per_score = calc_PER(pred_sparse, ref_sparse, normalize=True, merge_phn=True)
        print(json_path, "PER Score:", per_score)

 

    
