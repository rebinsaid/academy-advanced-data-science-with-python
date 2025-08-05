# Import GridSearchCV() from sklearn.model_selection and use it to 
# search across the different scalers that you can use.

from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV

parameters = {
    'scaler': [RobustScaler(), MinMaxScaler(), StandardScaler()]
}

grid = GridSearchCV(pipeline_lr, parameters)


# Fit your grid to (X_train, y_train).

grid.fit(X_train, y_train)


# What was the best scaler according to the GridSearchCV and what was the average accuracy score?

print(grid.best_params_, grid.best_score_)


# Use grid.score(X, y) to find the accuracy score on your train and test data.

print(grid.score(X_train, y_train), grid.score(X_test, y_test))


# Make a dataframe from the cv_results_ from your grid and sort_values() 
# by rank_test_score to see the best models and their parameters.

cv_results = pd.DataFrame(grid.cv_results_)
cv_results.sort_values('rank_test_score')