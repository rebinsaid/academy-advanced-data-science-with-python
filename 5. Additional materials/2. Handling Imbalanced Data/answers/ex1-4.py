
pipeline_tl = Pipeline(steps=[('under', tl), ('model', model)])
pipeline_nm = Pipeline(steps=[('under', nm), ('model', model)])
pipeline_oss = Pipeline(steps=[('under', oss), ('model', model)])

cv = StratifiedKFold(n_splits=5)
ftwo_scorer = make_scorer(fbeta_score, beta=2)

scores_tl = cross_val_score(pipeline_tl, X_train, y_train, scoring=ftwo_scorer, cv=cv, n_jobs=-1)
scores_nm = cross_val_score(pipeline_nm, X_train, y_train, scoring=ftwo_scorer, cv=cv, n_jobs=-1)
scores_oss = cross_val_score(pipeline_oss, X_train, y_train, scoring=ftwo_scorer, cv=cv, n_jobs=-1)

print(f"Mean score (Tomek Links): {(np.mean(scores_tl)):.3f}")
print(f"Mean score (Near-Miss V3): {(np.mean(scores_nm)):.3f}")
print(f"Mean score (One-sided Selection): {(np.mean(scores_oss)):.3f}")
