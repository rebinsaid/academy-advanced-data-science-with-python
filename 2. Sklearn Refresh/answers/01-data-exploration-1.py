
# a) How many missing values are there in each column?
display(stroke.isnull().sum())

# b) Use seaborn's heatmap (on df.isnull()) to determine if there is a pattern to the missing values
sns.heatmap(stroke.isnull())