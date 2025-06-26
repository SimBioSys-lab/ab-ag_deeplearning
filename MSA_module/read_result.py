import pandas as pd

# Initialize an empty list to store the results
results = []

# Read the file line by line
with open("NM4.5.out", "r") as file:
    current_model = None
    current_fold = None
    model_structure = None
    dropout = None
    category = None
    auc_pr = None
    auc_roc = None
    lines = file.readlines()

    for i in range(len(lines)):
        line = lines[i].strip()

        # Check if the line contains the model name
        if line.startswith("Testing model:"):
            current_model = line.split(":")[1].strip()
            current_fold = current_model.split("_")[1][-1]  # Extract fold number
            model_structure = "_".join(current_model.split("_")[2:5])  # Extract model structure
            dropout = float(current_model.split("_")[-1][2:-4])  # Extract dropout value
            category = None  # Reset the category when starting a new model

        # Check for category (e.g., Lchain, Hchain, Antibody, Antigen)
        if line.startswith("Metrics for "):
            category = line.split("Metrics for ")[1][:-1]  # Extract category (e.g., Lchain, Hchain, etc.)

        # Skip AGchain_0, AGchain_1, and AGchain_2
        if category in {"AGchain_0", "AGchain_1", "AGchain_2", "Lchain", "Hchain"}:
            continue

        # Extract AUC-ROC and AUC-PR values
        if "AUC-ROC:" in line:
            auc_roc = float(line.split("AUC-ROC:")[-1].strip())
        if "AUC-PR:" in line:
            auc_pr = float(line.split("AUC-PR:")[-1].strip())

        # Append the results only if a valid category and metrics are detected
        if current_model and category and auc_pr is not None and auc_roc is not None:
            results.append({
                "Model": current_model,
                "Fold": current_fold,
                "Structure": model_structure,
                "Dropout": dropout,
                "Category": category,
                "AUC-ROC": auc_roc,
                "AUC-PR": auc_pr
            })
            # Reset metrics after saving them
            auc_pr = None
            auc_roc = None

# Convert the results to a DataFrame
df = pd.DataFrame(results)

# Calculate the average AUC-ROC and AUC-PR for each structure and dropout
avg_metrics = df.groupby(["Structure", "Dropout", "Category"])[["AUC-ROC", "AUC-PR"]].mean().reset_index()

# Calculate the maximum AUC-ROC and AUC-PR for each structure and dropout
max_metrics = df.groupby(["Structure", "Dropout", "Category"])[["AUC-ROC", "AUC-PR"]].max().reset_index()

# Save the detailed, averaged, and maximum metrics to CSV files
df.to_csv("metrics_summary.csv", index=False)
avg_metrics.to_csv("avg_metrics_summary.csv", index=False)
max_metrics.to_csv("max_metrics_summary.csv", index=False)

# Print the DataFrames for validation
print("Detailed Metrics:")
print(df)
print("\nAverage Metrics:")
print(avg_metrics)
print("\nMaximum Metrics:")
print(max_metrics)

