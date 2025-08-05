
pca = PCA(n_components=2)
X_train_res_pca = pca.fit_transform(X_train_res)

plot_majority_minority_class(X_train_res_pca, y_train_res)

# Visually it does not look different, since it only created random copies of existing points