
preprocessing = Pipeline(steps=[
    ('ct', ct),
    ('impute', SimpleImputer(strategy='mean')),
])

preprocessing.set_output(transform='pandas')
display(preprocessing.fit_transform(X_train))

model = DecisionTreeClassifier(max_depth=3, 
                               class_weight='balanced',
                               random_state=42)

pipeline = Pipeline(steps=[
    ('preprocessing', preprocessing),
    ('model', model)
])
pipeline