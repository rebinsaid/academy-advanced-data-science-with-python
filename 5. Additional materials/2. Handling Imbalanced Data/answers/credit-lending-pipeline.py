from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import make_scorer, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import LabelEncoder


# Encode the labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, stratify=y, random_state=0)


model = HistGradientBoostingClassifier(
    categorical_features="from_dtype", random_state=0
)

model.fit(X_train, y_train)
model

scores = {
    "accuracy": make_scorer(accuracy_score),
    "precision": make_scorer(precision_score),
    "recall": make_scorer(recall_score)
}

print(f"Accuracy: {scores['accuracy'](model, X_test, y_test)}")
print(f"Precision: {scores['precision'](model, X_test, y_test)}")
print(f"Recall: {scores['recall'](model, X_test, y_test)}")