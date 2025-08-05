
X_train_tl, y_train_tl = tl.fit_resample(X_train, y_train)
X_train_nm, y_train_nm = nm.fit_resample(X_train, y_train)
X_train_oss, y_train_oss = oss.fit_resample(X_train, y_train)