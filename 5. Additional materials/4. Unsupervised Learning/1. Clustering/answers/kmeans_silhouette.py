from sklearn.metrics import silhouette_score

K = range(2, 10)
score = []
for k in K: 
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(X_scaled)
    labels = kmeans.labels_
    sil_score = silhouette_score(X_scaled, labels, metric='euclidean')
    score.append(sil_score)

plt.plot(K, score, 'bx-')
plt.title('The Sihouette Score')
plt.xlabel('k')
plt.ylabel('Silhouette Score');
