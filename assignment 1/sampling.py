import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.combine import SMOTETomek

data = pd.read_csv("Creditcard_data.csv")

X = data.drop("Class", axis=1)
y = data["Class"]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Sampling methods dictionary
sampling_methods = {
    "Sampling1": RandomUnderSampler(random_state=42),
    "Sampling2": RandomOverSampler(random_state=42),
    "Sampling3": SMOTE(random_state=42),
    "Sampling4": NearMiss(),
    "Sampling5": SMOTETomek(random_state=42)
}
