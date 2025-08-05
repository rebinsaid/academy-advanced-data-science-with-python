from sklearn.cluster import KMeans

K = range(1, 10)
score = []
for k in K:
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(X_scaled)
    score.append(kmeans.inertia_)

plt.plot(K, score, 'bx-')
plt.title('The Elbow Method')
plt.xlabel('k')
plt.ylabel('Inertia');
