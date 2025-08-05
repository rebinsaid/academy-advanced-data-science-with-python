
# new feature matrix
X_state, _ = (
    stroke
    .assign(bmi = lambda df: df['height'] / df['weight']**2)
    
    # add new state column
    .assign(state = lambda df: df['address'].str.split().str[-2])
    
    .pipe(create_Xy, 
          drop_cols=drop_cols, 
          target_col=target,
          )
)

# Train-test split
X_train_state, X_test_state, _, _ = train_test_split(X_state,
                                                    y,
                                                    test_size = 0.25,
                                                    random_state = 123,
                                                    stratify = y,
                                                    )