
# prune CDC data to 2021 and only state and rate columns
state_rate2021 = state_mortality.loc[lambda df: df['year'] == 2021, ['state','rate']]

X_rate, _ = (
    stroke
    .assign(bmi = lambda df: df['height'] / df['weight']**2)
    .assign(state = lambda df: df['address'].str.split().str[-2])

    # merge dfs
    .merge(state_rate2021, on = 'state', how = 'left')
    .rename(columns={'rate':'mortality_rate'})
    
    .pipe(create_Xy, 
          drop_cols=drop_cols, 
          target_col=target,
          )
)