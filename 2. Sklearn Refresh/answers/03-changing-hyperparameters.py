
# With default model parameters
new_model = DecisionTreeClassifier(
    max_depth=None,
    min_samples_leaf=1,
    min_samples_split=2,
    class_weight='balanced', 
    random_state=42)

new_pipeline = Pipeline(steps=[
    ('preprocessing', preprocessing),
    ('model', new_model)
])

new_pipeline.fit(X_train,y_train)

fig, ax = plt.subplots()
RocCurveDisplay.from_estimator(new_pipeline, X_train, y_train, ax=ax, name='Train')
RocCurveDisplay.from_estimator(new_pipeline, X_test, y_test, ax=ax, name='Test')

# With default parameters, the model is massively overfitting!