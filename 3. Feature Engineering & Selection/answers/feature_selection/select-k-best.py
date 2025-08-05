skb_knn_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('feature_selection', SelectKBest(f_classif, k=10)),
    ('model', knn_model)
])

skb_forest_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('feature_selection', SelectKBest(f_classif, k=10)),
    ('model', forest_model)
])

skb_knn_pipeline.fit(X_train_corr, y_train)
skb_forest_pipeline.fit(X_train_corr, y_train)

y_skb_train_probs = skb_knn_pipeline.predict_proba(X_train_corr)[:,1]
y_skb_test_probs = skb_knn_pipeline.predict_proba(X_test_corr)[:,1]

print(f'Train AUC: {round(roc_auc_score(y_train, y_skb_train_probs),4)}',
      f'Test AUC: {round(roc_auc_score(y_test, y_skb_test_probs),4)}',
      sep='\n'
      )

y_skb_train_probs = skb_forest_pipeline.predict_proba(X_train_corr)[:,1]
y_skb_test_probs = skb_forest_pipeline.predict_proba(X_test_corr)[:,1]

print(f'Train AUC: {round(roc_auc_score(y_train, y_skb_train_probs),4)}',
      f'Test AUC: {round(roc_auc_score(y_test, y_skb_test_probs),4)}',
      sep='\n'
      )