# Decision trees looks for the most optimal split in ordered data.
# If it were to do this for categories, it would need to look at all the permutations
# and decide which is the best splitter.

# If you take the who column as an example:

stroke['who'].value_counts()

# In the entire data, there are 2576 women, 1677 men and 856 children. 
# Right away the Decision tree would need to compute what the best split would be
# when considering all permutations of these groups 
# I.e. is the best split Man+Woman | Child, or Child+Woman | Man etc.

# After each split the tree makes, a new subset of data gets seen by the model
# The Decision tree needs to repeat the above process with the new data

# This already can be highly costly, however imagine a feature with 5+ categories and a tree with 7+ splits
