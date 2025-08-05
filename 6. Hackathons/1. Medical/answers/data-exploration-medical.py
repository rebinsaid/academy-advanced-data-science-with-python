# 1. How many patients is there data for?

medical['id'].nunique()


# 2. What datatypes does the dataset contain? Are there any missing values?

print(f'Datatypes in the data:\n {medical.dtypes.value_counts()}\n')

print(f'There are {medical.isnull().sum().sum()} missing values in the data')


# 3. How many predictive features are there?

print(medical.columns)

# id and diagnosis are not suitable feature (unique identifier and target)
print(f'There are {medical.shape[1]-2} features')


# 4. Run a pair plot on the features you think are the most correlated, using hue as the diagnosis.
    
import seaborn as sns

columns = ['diagnosis', 'perimeter_mean', 'area_mean', 'perimeter_se', 'area_se', 'perimeter_worst', 'area_worst']

sns.pairplot(medical[columns], hue='diagnosis')