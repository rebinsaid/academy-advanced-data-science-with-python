
from imblearn.over_sampling import RandomOverSampler

rand_over = RandomOverSampler(sampling_strategy=0.5)
X_train_res, y_train_res = rand_over.fit_resample(X_train, y_train)

print(f"Original dataset shape: {Counter(y_train)}\nResampled dataset shape: {Counter(y_train_res)}")