
# Instantiate model
better_model = GradientBoostingClassifier(
    # class_weight='balanced',
    random_state=42,
    )

# Model pipeline
better_pipeline = Pipeline(steps=[
    ('preprocessing', preprocessing),
    ('model', better_model)
])

# GridSearchCV
better_params = {
    'model__n_estimators': [100, 200],
    'model__max_depth': range(2,6),
    'model__max_features': ['sqrt', 10]
}

better_grid = GridSearchCV(better_pipeline, better_params, scoring='roc_auc', cv = 3)
better_grid.fit(X_train, y_train)
better_results = pd.DataFrame(better_grid.cv_results_).sort_values('rank_test_score')
display(better_results.head())
best_gbf = better_grid.best_estimator_

# Plot
fig, ax = plt.subplots(1, 1, figsize=(8,6))
RocCurveDisplay.from_estimator(best_gbf, X_train, y_train, ax=ax, name='Train')
RocCurveDisplay.from_estimator(best_gbf, X_test, y_test, ax=ax, name='Test')
ax.set(title='Tuned model');