from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer

new_ct = ColumnTransformer([
    ('onehot', OneHotEncoder(drop='if_binary'), categorical_cols)
], remainder='passthrough')
