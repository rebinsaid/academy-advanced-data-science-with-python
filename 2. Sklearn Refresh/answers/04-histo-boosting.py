
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

# Define model
model_histboost = HistGradientBoostingClassifier() # Play around with parameters if you want


# Define pipeline
preprocessing_hist = Pipeline(steps=[
    ('onehot', onehot), # No imputation necessary for HistGradientBoosting
])

pipeline_histboost = Pipeline(steps=[
    ('preprocessing', preprocessing_hist),
    ('model', model_histboost)
])


# Tune pipeline
params = {
    'model__max_depth': range(20, 41 ,10),
    'model__max_leaf_nodes': range(20, 41, 10),
    'model__learning_rate': [0.2, 0.1, 0.01],
}

grid = GridSearchCV(pipeline_histboost, params, scoring='roc_auc', cv = 3)
grid.fit(X_train, y_train)
model_best = grid.best_estimator_

cv_results = pd.DataFrame(grid.cv_results_).sort_values('rank_test_score')
display(cv_results.head())


#Graph AUC metrics
print('\n')
fig, ax = plt.subplots(1, 2, figsize=(16,6))

RocCurveDisplay.from_estimator(pipeline_boosting, X_train, y_train, ax=ax[0], name='Train')
RocCurveDisplay.from_estimator(pipeline_boosting, X_test, y_test, ax=ax[0], name='Test')
ax[0].set(title='Gradient Boosting')

RocCurveDisplay.from_estimator(model_best, X_train, y_train, ax=ax[1], name='Train')
RocCurveDisplay.from_estimator(model_best, X_test, y_test, ax=ax[1], name='Test')
ax[1].set(title='GridCV Histogram boosting');