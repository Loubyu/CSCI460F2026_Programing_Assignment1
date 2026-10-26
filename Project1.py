import csv
import statistics
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score

df = pd.read_csv('bank-full.csv', sep=',', quoting=csv.QUOTE_NONE)
df.columns = df.columns.str.strip('"')
df["age"] = df["age"].str.strip('"').astype(int)

for column in df.columns:
    if pd.api.types.is_string_dtype(df[column]):
        df[column] = df[column].str.strip('"')

        unique_values = df[column].unique()
        mapping = {}

        for i, value in enumerate(unique_values):
            mapping[value] = i

        df[column] = df[column].map(mapping)

df = df.dropna()

y = df["y"]
X = df.drop(columns=["y"])

trial_results = []
summary_results = []

for training_percentage in range(10, 100, 10):
    test_percentage = 1 - (training_percentage / 100)

    accuracies = []
    f1_scores = []

    print(f"\nTraining data: {training_percentage}%")

    for trial in range(10):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_percentage,
            stratify=y
        )

        model = GaussianNB()
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(predictions, y_test)
        f1 = f1_score(predictions, y_test, average="weighted")

        accuracies.append(accuracy)
        f1_scores.append(f1)

        print(f"Trial {trial + 1}: Accuracy = {accuracy:.4f}, F1 = {f1:.4f}")

        trial_results.append({
            "Training Percentage": training_percentage,
            "Trial": trial + 1,
            "Accuracy": accuracy,
            "F1 Score": f1
        })

    average_accuracy = statistics.mean(accuracies)
    average_f1 = statistics.mean(f1_scores)

    accuracy_std = statistics.stdev(accuracies)
    f1_std = statistics.stdev(f1_scores)

    print(f"Average Accuracy: {average_accuracy:.4f}")
    print(f"Average F1 Score: {average_f1:.4f}")
    print(f"Accuracy Standard Deviation: {accuracy_std:.4f}")
    print(f"F1 Standard Deviation: {f1_std:.4f}")

    summary_results.append({
        "Training Percentage": training_percentage,
        "Mean Accuracy": average_accuracy,
        "Accuracy Std Dev": accuracy_std,
        "Mean F1 Score": average_f1,
        "F1 Std Dev": f1_std
    })

trial_df = pd.DataFrame(trial_results)
summary_df = pd.DataFrame(summary_results)

trial_df.to_csv("trial_results.csv", index=False)
summary_df.to_csv("summary_results.csv", index=False)

print("\nSummary Table:")
print(summary_df.to_string(index=False))

plt.figure(figsize=(10, 6))

plt.plot(
    summary_df["Training Percentage"],
    summary_df["Mean Accuracy"],
    marker="o",
    label="Average Accuracy"
)

plt.plot(
    summary_df["Training Percentage"],
    summary_df["Mean F1 Score"],
    marker="s",
    label="Average F1 Score"
)

plt.xlabel("Training Data Percentage")
plt.ylabel("Classification Score")
plt.title("Gaussian Naive Bayes Classification Performance")
plt.xticks(range(10, 100, 10))
plt.legend()
plt.grid(True)

plt.savefig("performance_graph.png", dpi=300, bbox_inches="tight")
plt.close()