# +
# Scaled
# Scale the data.
scaler = StandardScaler() 
X_penguins_scaled = scaler.fit_transform(X_penguins)

# PCA. 
pca_transformer = PCA(n_components=2)
pca_data = pca_transformer.fit_transform(X_penguins_scaled)

# Create dataframe. 
df_penguins = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])

# Visualisation
targets = y_penguins.unique()
colors = ['r', 'b', 'g'] 
visualise(df_penguins , targets, colors, y_penguins)

retained_info = pca_transformer.explained_variance_ratio_.sum()*100
n_components = pca_transformer.components_.shape[0]
print(f'In total we managed to preserve {retained_info:.2f}% of information with {n_components} component(s)')
