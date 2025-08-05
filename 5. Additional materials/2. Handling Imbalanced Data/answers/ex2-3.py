
model = SVC()

over = RandomOverSampler(sampling_strategy=0.5)

pipeline = Pipeline(steps=[('over', over),
                          ('model', model)])

cv = StratifiedKFold(n_splits=10)
scores_ros = cross_val_score(pipeline, X_train, y_train, scoring=ftwo_scorer, cv=cv, n_jobs=-1)

print(f'Mean score (RandomOverSampler): {(np.mean(scores_ros)):.3f}')