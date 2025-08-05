
from imblearn.over_sampling import RandomOverSampler

over = RandomOverSampler(sampling_strategy=0.3)
under = RandomUnderSampler(sampling_strategy=0.5)

pipeline_comb = Pipeline(steps=[
    ('over', over),
    ('tomek', TomekLinks()),
    ('under', under),
    ('model', model)])

scores_comb = cross_val_score(pipeline_comb, X_train, y_train, scoring=ftwo_scorer, cv=cv, n_jobs=-1)

print(f'Mean score: {(np.mean(scores_comb)):.3f}')

pipeline.fit(X_train, y_train)
ftwo_scorer(pipeline, X_test, y_test)
