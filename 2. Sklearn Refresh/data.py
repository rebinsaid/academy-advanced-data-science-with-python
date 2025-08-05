def create_Xy(df, drop_cols, target_col):
    df = df.drop(columns=drop_cols)
    return (
        df.drop(columns=target_col),
        df[target_col]
    )


def get_train_test_params(y):
    return {
        'test_size':0.25, 
        'random_state':123, 
        'stratify':y
    }
