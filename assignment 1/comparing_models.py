import sampling as s
import models
from sklearn.metrics import accuracy_score
import pandas as pd

# Access dictionary inside models.py
results = pd.DataFrame(
    index=models.models.keys(),
    columns=s.sampling_methods.keys()
)

for samp_name, sampler in s.sampling_methods.items():
    X_resampled, y_resampled = sampler.fit_resample(
        s.X_train, s.y_train
    )

    for model_name, model in models.models.items():
        model.fit(X_resampled, y_resampled)
        y_pred = model.predict(s.X_test)
        acc = accuracy_score(s.y_test, y_pred) * 100
        results.loc[model_name, samp_name] = round(acc, 2)

print("Model training and evaluation completed.")
print(results)

print("\nBest Sampling Technique for Each Model:\n")

for model in results.index:
    best_sampling = results.loc[model].astype(float).idxmax()
    best_accuracy = results.loc[model].astype(float).max()
    print(f"{model} → {best_sampling} ({best_accuracy}%)")
