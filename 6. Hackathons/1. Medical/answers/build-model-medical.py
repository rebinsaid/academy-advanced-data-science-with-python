from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

model_lr = LogisticRegression()

pipeline_lr = Pipeline(steps = [
    ('scaler', StandardScaler()),
    ('model', model_lr)
])


# Fit the pipeline to X_train and y_train

pipeline_lr.fit(X_train, y_train)