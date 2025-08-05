# Perform the train test split on the data to create X_train, X_test, y_train, y_test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=111)


# Check the shape of X_train, X_test, y_train and y_test

print('Shape of X_train and y_train', X_train.shape, y_train.shape)
print('Shape of x_test and y_test', X_test.shape, y_test.shape)