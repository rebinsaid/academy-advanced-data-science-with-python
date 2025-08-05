# +
import matplotlib.pyplot as plt
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# -

def create_Xy(df, target, categorical_columns):
    
    ct = ColumnTransformer(
        [('onehot', OneHotEncoder(drop='first'),
          categorical_columns)], 
        remainder='passthrough'
    ) 
    
    df = df.reset_index(drop=True)
    X = df.drop(columns=[target])
    X = ct.fit_transform(X)
    
    y = df[target]
    
    return X, y


def plot_majority_minority_class(X, y):
    for label in np.unique(y):
        plt.scatter(X[y==label, 0], X[y==label, 1], label=label)
    plt.legend()
    plt.show()

