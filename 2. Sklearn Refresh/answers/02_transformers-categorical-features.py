stroke.select_dtypes('O').nunique()

# None of the columns have an ordinal relationship (a certain 
# smoking_status isn't higher or lower than another)
# We should use OneHotEncoder.
