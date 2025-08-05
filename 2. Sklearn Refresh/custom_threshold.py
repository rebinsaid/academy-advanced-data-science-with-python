from sklearn.tree import DecisionTreeClassifier

class CustomThreshold(DecisionTreeClassifier):
    """ Custom threshold wrapper for binary classification"""
    def __init__(self, model, threshold=0.5):
        super().__init__()
        self.model = model
        self.threshold = threshold
        self._estimator_type == 'classifier'
    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)
        return self
    def predict(self, X):
        return (self.model.predict_proba(X)[:, 1] > self.threshold).astype(int)
