# First let's try out the one hot encoding. Import the OneHotEncoder from sklearn.preprocessing.
from sklearn.preprocessing import OneHotEncoder

onehot = OneHotEncoder(drop='first')

onehot.fit_transform(X_train).shape


# Rerun .fit_transform() on only the categorical columns from X_train

onehot.fit_transform(X_train[categorical_columns]).shape


# Now we need to build a column transformer so that we can only encode the categorical columns.

from sklearn.compose import ColumnTransformer

column_transformer = ColumnTransformer(
    [
        ('onehot', OneHotEncoder(drop='first', sparse=False), categorical_columns)
    ], remainder='passthrough'
)

X_train_encoded = column_transformer.fit_transform(X_train)
X_train_encoded.shape


# Now let's try out a scaler. You can only do this on the encoded data since you cannot scale categorical features!!

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

pd.DataFrame(scaler.fit_transform(X_train_encoded), columns = column_transformer.get_feature_names())