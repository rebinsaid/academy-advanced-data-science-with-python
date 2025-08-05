
X_bmi, _ = (
    stroke
    ### your code here
    .assign(bmi = lambda df: df['weight'] / df['height']**2)
    .pipe(create_Xy,
          drop_cols=drop_cols,
          target_col=target)
)
