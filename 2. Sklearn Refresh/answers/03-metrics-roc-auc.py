
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay

fig, ax = plt.subplots(1, 2, figsize=(16,6))

RocCurveDisplay.from_estimator(base_model, X_train, y_train, ax=ax[0], name='Train')
RocCurveDisplay.from_estimator(base_model, X_test, y_test, ax=ax[0], name='Test')
ax[0].set(title='Baseline model');

RocCurveDisplay.from_estimator(pipeline_rf, X_train, y_train, ax=ax[1], name='Train')
RocCurveDisplay.from_estimator(pipeline_rf, X_test, y_test, ax=ax[1], name='Test')
ax[1].set(title='Random Forest model')

