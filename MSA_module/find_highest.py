import pandas as pd

df = pd.read_csv("ihd10_metrics.csv")          # <-- your CSV path
ag = df[df["subset"] == "AG"].copy()
# drop NaNs just in case
ag = ag.dropna(subset=["auc_pr"])

top = ag.sort_values("auc_pr", ascending=False).iloc[0]
print(f"Top AG AUC-PR: PDBID={top['PDBID']}  auc_pr={top['auc_pr']:.6f}")

bot = ag.sort_values("auc_pr", ascending=False).iloc[-1]
print(f"Bot AG AUC-PR: PDBID={bot['PDBID']}  auc_pr={bot['auc_pr']:.6f}")


