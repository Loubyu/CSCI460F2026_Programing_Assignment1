import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
accuracy_score,
confusion_matrix,
ConfusionMatrixDisplay,
f1_score,
classification_report,
)

df = pd.read_csv('bank-full.csv', sep=',',quoting=csv.QUOTE_NONE)
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

for training_percentage in range(10, 100, 10):
    test_percentage = 1 - (training_percentage / 100)
    accuracies = []
    f1_scores = []

    for trial in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percentage, stratify=y)
        model = GaussianNB()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(predictions, y_test)
        f1 = f1_score(predictions, y_test, average="weighted")
        accuracies.append(accuracy)
        f1_scores.append(f1)
    
    average = sum(accuracies) / 10
    print("Average accuracy for", training_percentage, "% training data:", average)
    average_f1scores = sum(f1_scores) / 10
    print("Average F1 score for", training_percentage, "% training data:", average_f1scores)
