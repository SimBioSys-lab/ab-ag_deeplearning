import numpy as np

def calculate_class_weights(npz_file):
    """
    Calculate class weights for a binary classification problem 
    based on the distribution of labels in an .npz file.

    Args:
        npz_file (str): Path to the .npz file containing labels.

    Returns:
        dict: Class weights for each class {0: weight_0, 1: weight_1}.
    """
    # Load the data from the .npz file
    data = np.load(npz_file, allow_pickle=True)

    # Initialize counts for each class
    positive_count = 0
    negative_count = 0

    # Count positive and negative samples
    for key in data.keys():
        labels = data[key].flatten()
        positive_count += np.sum(labels == 1)
        negative_count += np.sum(labels == 0)

    # Total number of samples
    total_samples = positive_count + negative_count

    # Avoid division by zero
    if positive_count == 0 or negative_count == 0:
        raise ValueError("One of the classes has zero samples. Adjust your dataset.")

    # Calculate class weights
    weight_0 = total_samples / (2 * negative_count)
    weight_1 = total_samples / (2 * positive_count)

    class_weights = {0: weight_0, 1: weight_1}

    print(f"Total Samples: {total_samples}")
    print(f"Class 0 Samples: {negative_count}, Class 1 Samples: {positive_count}")
    print(f"Class Weights: {class_weights}")

    return class_weights

# Example Usage
npz_file_path = "padded_interfaces_train_3000.npz"
class_weights = calculate_class_weights(npz_file_path)
print(f"Calculated Class Weights: {class_weights}")

