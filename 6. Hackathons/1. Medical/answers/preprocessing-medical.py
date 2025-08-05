from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

pd.DataFrame(scaler.fit_transform(X_train), columns=features)