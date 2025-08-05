from sklearn.base import BaseEstimator, TransformerMixin

class DropColumnsContaining(BaseEstimator, TransformerMixin):
    def __init__(self, cols_containing):
        self.cols_containing=cols_containing #Use regex to search for multiple terms, e.g. "first_term|second_term"
    
    def fit(self, X, y=None):
        #regex_filter = "|".join(self.cols_containing) 
        regex_filter = self.cols_containing
        
        self.cols = X.filter(regex=regex_filter).columns
        return self
    
    def transform(self, X):
        return X.drop(columns=self.cols)
    
remove_perimeter = DropColumnsContaining(cols_containing='perimeter')

remove_perimeter.fit_transform(X)