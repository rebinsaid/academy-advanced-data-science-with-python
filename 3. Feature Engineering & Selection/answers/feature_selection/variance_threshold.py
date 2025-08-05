# +
from sklearn.feature_selection import VarianceThreshold


num_before = len(food_df.columns)
print(f'Features before: {num_before}')

selector = VarianceThreshold(0.05)
selected = selector.fit_transform(food_df.drop(columns=['title']))

num_after = selected.shape[1]
print(f'Features after: {num_after}')

print(f'Number of features eliminated: {num_before - num_after}')

pd.DataFrame(columns=selector.get_feature_names_out(),
             data=selected)

