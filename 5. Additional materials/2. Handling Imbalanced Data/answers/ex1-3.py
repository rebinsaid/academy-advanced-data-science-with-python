
pca = PCA(n_components=2)
X_train_tl_pca = pca.fit_transform(X_train_tl)
X_train_nm_pca = pca.fit_transform(X_train_nm)
X_train_oss_pca = pca.fit_transform(X_train_oss)

plot_majority_minority_class(X_train_tl_pca, y_train_tl, title = "Tomek Links")
plot_majority_minority_class(X_train_nm_pca, y_train_nm, title = "Near-Miss V3")
plot_majority_minority_class(X_train_oss_pca, y_train_oss, title = "One-sided Selection")