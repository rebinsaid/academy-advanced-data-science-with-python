from sklearn.linear_model import LogisticRegression

logistic_model = LogisticRegression(
                                    class_weight='balanced', 
                                   random_state=42)

logistic_model.fit(X_train, y_train)

y_pred_test = logistic_model.predict_proba(X_test)[:,1]
y_pred_train = logistic_model.predict_proba(X_train)[:,1]

print(f'AUC score: {roc_auc_score(y_test, y_pred_test), roc_auc_score(y_train, y_pred_train)}')
