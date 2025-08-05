
preds = new_pipeline.predict(X_test)

print(classification_report(y_test, preds))

# For a medical dataset, recall as it is better to maximize since you typically want to capture as many 
# true positive cases as possible.