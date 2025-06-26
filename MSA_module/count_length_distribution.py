import numpy as np
from collections import Counter

# Settings
BIN_SIZE = 100
MAX_LENGTH = 4000
BINS = list(range(0, MAX_LENGTH + BIN_SIZE, BIN_SIZE))  # 0–100, 100–200, ..., 1900–2000
BINS.append(float('inf'))  # Catch-all bin for > MAX_LENGTH

def get_length(array):
    if array.ndim == 1:
        return len(array)
    elif array.ndim == 2:
        return array.shape[1]
    else:
        raise ValueError("Unsupported array shape: only 1D or 2D allowed.")

def get_bin_label(length):
    for i in range(len(BINS) - 1):
        if length < BINS[i + 1]:
            lower = BINS[i]
            upper = int(BINS[i + 1]) if BINS[i + 1] != float('inf') else "+"
            return f"{lower}-{upper}"
    return f"{MAX_LENGTH}+"

def analyze_lengths(npz_file):
    data = np.load(npz_file, allow_pickle=True)
    bin_counter = Counter()

    # Count lengths into bins
    for key in data:
        try:
            arr = data[key]
            length = get_length(arr)
            label = get_bin_label(length)
            bin_counter[label] += 1
        except Exception as e:
            print(f"Failed to process {key}: {e}")

    # Sort bins numerically
    def bin_sort_key(label):
        return int(label.split("-")[0]) if '-' in label else float('inf')

    sorted_bins = sorted(bin_counter.keys(), key=bin_sort_key)

    # Print summary with cumulative total
    print(f"\nLength distribution in '{npz_file}':\n")
    cumulative = 0
    for label in sorted_bins:
        count = bin_counter[label]
        cumulative += count
        print(f"{label:<10} Count: {count:<6}  Cumulative: {cumulative}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_file", help="Path to the .npz file")
    args = parser.parse_args()
    analyze_lengths(args.npz_file)

