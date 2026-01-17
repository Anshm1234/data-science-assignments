import pandas as pd
from imblearn.over_sampling import SMOTE

data = pd.read_csv("Creditcard_data.csv")

print(data.head())
print(data["Class"].value_counts())

data = pd.read_csv("Creditcard_data.csv")

X = data.drop("Class", axis=1)
y = data["Class"]

smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

balanced_data = pd.DataFrame(X_balanced, columns=X.columns)
balanced_data["Class"] = y_balanced

balanced_data.to_csv("Creditcard_data_balanced.csv", index=False)

print("Balanced Class Distribution:\n")
print(balanced_data["Class"].value_counts())
