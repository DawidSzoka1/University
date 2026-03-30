import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
import seaborn as sns

dataset = pd.read_csv('../docs/diabetes.csv')
X = dataset.drop('Outcome', axis=1)
X = X.dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
K = range(2, 8)
fits = []
score = []
for k in K:
    # Utworzenie modelu metodą k-średnich w postaci rodziny skupień
    model = KMeans(n_clusters=k, random_state=0, n_init='auto').fit(X_pca)
    # Dodanie modelu do fits
    fits.append(model)
    # Wyznaczenie wartości silhouette score dla utworzonego modelu
    score.append(silhouette_score(X_pca, model.labels_,
                                  metric='euclidean'))
print(score)

sns.lineplot(x=K, y=score).set(title='Wykres Silhouette score --Diabetes', xlabel='Liczba skupień (k)',
                               ylabel='Silhouette score')
plt.show()
wcss = []
for i in range(2, 8):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300,
                    n_init=10, random_state=0)
    kmeans.fit(X_pca)
    # kmeans.inertia_ zwraca sumę kwadratów odległości od środka skupienia
    wcss.append(kmeans.inertia_)
# Wykreślenie wykresu dla metody łokcia
plt.plot(range(2, 8), wcss)
plt.title('Wykres dla metody "łokcia" -- Diabetes')
plt.xlabel('Liczba skupień (k)')
plt.ylabel('WCSS (inercja)')
plt.grid(True)
plt.show()

sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=fits[0].labels_)
plt.title("Rozkład obiektów (PCA) dla k=2")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=fits[1].labels_)
plt.title("Rozkład obiektów (PCA) dla k=3")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()
kmeans = KMeans(n_clusters=3)
kmeans.fit(X_pca)
labels = kmeans.labels_
print(labels)
